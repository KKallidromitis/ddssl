from lib.data.utils import build_dataset
import time
import os.path as osp
from mmaction.utils import collect_env
from torch.utils.data import DataLoader
from mmcv.parallel import collate
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import SGD
import torch.distributed as dist
import argparse
import wandb
import time
import datetime
from typing import Optional
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.scheduler import CosineAnnealing
import wandb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from lib.loss import bn2,bn
from lib.logging import wandb_dump_img
import kornia
import numpy as np
from kornia.augmentation import VideoSequential,ImageSequential
import wandb
import yaml
from torchvision import transforms as TF
from torchvision.models import resnet18
from lib.featurizer import ResNetFeatuerizer
from lib.solver import kernel_ridge,svm_forward
from lib.loss import match_loss_fn
from lib.fpn import FCNHead,Generator
import os
import pathlib
from torchvision.models._utils import IntermediateLayerGetter
from lib.data.utils import sample_patches
def get_parser():
    parser = argparse.ArgumentParser(description='Detcon-BYOL Training')
    parser.add_argument("--local_rank", metavar="Local Rank", type=int, default=0, 
                        help="Torch distributed will automatically pass local argument")
    parser.add_argument("--cfg", metavar="Config Filename", default="train_imagenet_300", 
                        help="Experiment to run. Default is Imagenet 300 epochs")
    parser.add_argument("--name", metavar="Log Name", default="", 
                        help="Name of wandb entry")
    parser.add_argument("--skip_eval", action="store_true", 
                        help="skip Davis Evaluation")
    parser.add_argument("--random", action="store_true", 
                        help="use random initalized ResNet") # Broken, to be fixed
    parser.add_argument("--sample", action="store_true", 
                        help="Use real image on init") # Use real image init, broken (image do not update)
    parser.add_argument("--label", action="store_true", 
                        help="Use learnable fake labels")
    parser.add_argument("--gan", action="store_true", 
                        help="Add (pseudo) GAN loss for syn images")
    parser.add_argument("--cosine_decay", action="store_true", 
                        help="Cosine Lr Schedule")
    parser.add_argument("--style", action="store_true", 
                        help="Include Image-Level Gram Loss")
    parser.add_argument("--eval_interval",type=int,default=1, 
                        help="eval intervals") # not used
    parser.add_argument("--batch_size",type=int,default=64, 
                        help="batch size per gpu")
    parser.add_argument("--num_workers",type=int,default=8, 
                        help="num of workers")
    parser.add_argument("--inner_loop",type=int,default=10, 
                        help="num of workers")
    parser.add_argument("--epochs",type=int,default=100, 
                        help="num of workers")
    parser.add_argument("--blr",type=float,default=0.05, 
                        help="base learning rate of vfs")
    parser.add_argument("--seed",type=int,default=0, 
                        help="Random Seed")
    parser.add_argument("--weight_decay",type=float,default=0.0001, 
                        help="weight decay of vfs")
    parser.add_argument("--jit",type=float,default=1e-3, 
                        help="regulizer value in KRR")
    
    parser.add_argument("--config",type=str,default= 'configs/imagenet-1k.yaml', 
                        help="config file")
    parser.add_argument("--loss_type",type=str,default= 'ours', 
                        help="see lib, can be byol, simclr, or others"),
    parser.add_argument("--mm",type=float,default= 0.6, 
                        help="momentum")
    parser.add_argument("--method",type=str,default='pixels', 
                        help="pixels or FCN, with FCN, a generative model is employed")
                        
    return parser

                    

# Parse Args 
parser = get_parser()
args = parser.parse_args()
print(args)

np.random.seed(args.seed)

torch.manual_seed(args.seed)


timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
env_info_dict = collect_env()
env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])

#Setup CFG
meta = dict()
meta['env_info'] = env_info
meta['seed'] = args.seed
meta['exp_name'] = osp.basename(args.config)


with open(args.config, 'r') as f:
    config = yaml.safe_load(f)






# init random dataset
lr_scheduler = CosineAnnealing(**config['lr_config'])
SIZE = 2350   # 1p
LATENT_DIM = 3584
IM_SIZE = 64
indices = list(range(SIZE))
if args.method == 'pixels':
    video_syn = torch.randn(size=(SIZE, 3* IM_SIZE* IM_SIZE + LATENT_DIM ), dtype=torch.float, requires_grad=False)
elif args.method == 'FCN':
    video_syn = torch.randn(size=(SIZE, (7 * 7+ 14*14) * 512+ LATENT_DIM ), dtype=torch.float, requires_grad=False)
else:
    raise AssertionError("Invalid Method")

video_syn_momentum = torch.zeros_like(video_syn)

# label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,
def apply_kornia(aug,v,parms=None):
    if parms is not None:
        v = aug_list(v,params=parms)
    else:
        v = aug_list(v)
    return v,aug._params


world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ['RANK'])# or args.local_rank
local_rank = int(os.environ.get('LOCAL_RANK', '0'))    
dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
device = 'cuda'
torch.cuda.set_device(local_rank)
torch.cuda.manual_seed(args.seed)
print(f"init done {rank}")

# Init Models
model = ResNetFeatuerizer()
model.to(device)
model = DDP(model)
model.eval()

model_random = ResNetFeatuerizer(pretrained=None)
model_random.to(device)
model_random = DDP(model)
model_random.eval()


k = torch.tensor(0.).to(device) #parameter used for platt scaling
k.requires_grad = True




results = []
# HPS
CROP_SIZE = 64
BATCH_SYN = 1024
BATCH_SIZE = args.batch_size

GLOBAL_BATCH_SIZE = BATCH_SIZE * world_size
LR = args.blr * GLOBAL_BATCH_SIZE / 256 # default 0.05
epoches = args.epochs
EVAL_INTERVAL = args.eval_interval
SKIP_EVAL = args.skip_eval
INNER_LOOP = args.inner_loop
now = datetime.datetime.now()
ROOT_DIR = f'./ckpt/{now.strftime("%m-%d-%Y")}'
pathlib.Path(ROOT_DIR).mkdir(parents=True,exist_ok=True)

if args.method == 'FCN':
    fcn = Generator().to(device)
    encoder = resnet18(pretrained=False)
    encoder.avgpool = nn.Identity()
    encoder.fc = nn.Identity()
    encoder = IntermediateLayerGetter(encoder,return_layers=dict(layer4='out')).to(device)
    encoder = DDP(encoder)
    fcn = DDP(fcn)
    fcn.train()
    encoder.train()
    optimizer_model = SGD([*fcn.parameters(),*encoder.parameters()],lr=LR, momentum=0.9, weight_decay=args.weight_decay)

optimizer_k = SGD([k],lr=1e-2,momentum=0.8)
    
# Init dataset
aug_list = ImageSequential(
    kornia.augmentation.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
    kornia.color.BgrToRgb(),
    kornia.augmentation.RandomGrayscale(p=0.2),
    #kornia.augmentation.RandomAffine(360, p=1.0),
    kornia.augmentation.RandomResizedCrop((CROP_SIZE,CROP_SIZE),scale=(0.2,1.0)),
    #kornia.augmentation.CenterCrop(CROP_SIZE),
    kornia.augmentation.PadTo((CROP_SIZE,CROP_SIZE)),
    kornia.augmentation.RandomHorizontalFlip(),
    kornia.augmentation.RandomGaussianBlur(kernel_size=(23,23),sigma=(0.1, 0.2),p=0.5),
    kornia.augmentation.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    random_apply=False)

torch_vision_transforms = TF.Compose(
    [TF.Resize((IM_SIZE,IM_SIZE)),
    TF.ToTensor(),
])
dataset = build_dataset(config,torch_vision_transforms)
sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
)
dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,sampler=sampler,num_workers=args.num_workers,drop_last=True)
if args.sample:
    video_syn[:,LATENT_DIM:]=sample_patches(dataset,IM_SIZE,SIZE).reshape(SIZE,-1)
    video_syn = video_syn.detach().cuda()
    dist.broadcast(video_syn,src=0)
    video_syn = video_syn.detach().cpu()

# time tracking
total_iter = len(dataset) // (BATCH_SIZE * world_size)
start_time = time.time()
end = time.time()

if rank == 0:
    wandb.init()

def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)

new_lr = LR
for epoch in range(epoches):
    sampler.set_epoch(epoch)
    iter = 0
    
    
    if epoch % EVAL_INTERVAL == 0 and not SKIP_EVAL:
        pass
    print(f"EPOCH : {epoch}")
    
    for batch,label in dataloader:
        #idx = np.random.choice(indices,BATCH_SYN * world_size,replace=False)
        # print(idx[:10]) sanity check to make sure they are in sync
        if args.cosine_decay:
            new_lr = lr_scheduler.get_lr(LR,epoch*total_iter+iter,epoches*total_iter)
        # for param_group in optimizer.param_groups:
        #      param_group['lr'] = new_lr
        idx = np.random.choice(indices,BATCH_SYN)
        all_idx = idx
        syn_batch = video_syn[idx].detach().clone().to(device)
        syn_batch_momentum = video_syn_momentum[idx].detach().clone().to(device)
        syn_batch.requires_grad=True
        optimizer = SGD([syn_batch,],lr=new_lr, momentum=0.6, weight_decay=0)
        #breakpoint()
        optimizer.state[syn_batch]['momentum_buffer'] = syn_batch_momentum
        real_video_aug,parms = apply_kornia(aug_list,batch.to(device))
        real_video_aug = real_video_aug.detach()
        syn_video = syn_batch[...,LATENT_DIM:]
        if args.method == 'pixels':
            syn_video = syn_video.reshape(-1,3,IM_SIZE,IM_SIZE)
        elif args.method == 'FCN':
            syn_video = fcn(syn_video[:,:7*7*512].reshape(-1,512,7,7))
            #syn_video = fcn([syn_video[:,:7*7*512].reshape(-1,512,7,7),syn_video[:,7*7*512:].reshape(-1,512,14,14)])
            
        syn_label = syn_batch[...,:LATENT_DIM]
        syn_video_aug = syn_video
        if args.random:
            reset_all_weights(model_random)
        else:
            model_random = model
        #model_randopm.reset_parms()
        #syn_video_aug,_ = apply_kornia(aug_list,syn_video_aug)
        with torch.no_grad():
            feature_real = model_random(real_video_aug).detach()
            feature_real_gt =  model(real_video_aug).detach()
        feature_syn = model_random(syn_video_aug)
        bn = nn.Identity()
        if args.label:
            solved = kernel_ridge(bn(feature_real),syn_label)
            pred = svm_forward(bn(feature_real),bn(feature_syn),solved)
        else:
            solved = kernel_ridge(bn(feature_real) @ bn(feature_syn.T),bn(feature_syn))
            pred = svm_forward(bn(feature_real),bn(feature_syn),solved)
        
        losses = dict()
        match_loss = match_loss_fn(bn(pred),bn(feature_real_gt),k,args.loss_type) 
        dm_loss = ((feature_syn.mean(0)-feature_real.mean(0))**2).sum()
        losses.update(dict(match_loss=match_loss.item(),dm_loss =dm_loss.item()))
        loss = match_loss + 0.2* dm_loss
        if args.gan:
            all_features = torch.cat([feature_real,feature_syn],dim=0)
            gt_labels = torch.FloatTensor([1,]*len(feature_real)+[-1]*len(feature_syn)).reshape(-1,1).to(device) * 10
            solved = kernel_ridge(bn(all_features),gt_labels)
            pred_labels = svm_forward(bn(all_features),bn(all_features),solved)
            loss_gan = nn.functional.binary_cross_entropy_with_logits(pred_labels,(gt_labels > 0 ).float())
            losses.update(dict(loss_gan=loss_gan.item()))
            loss += 0.5 * (-loss_gan)
        if args.style:
            ga = torch.softmax(bn(feature_real),dim=-1)
            gb = torch.softmax(bn(feature_syn),dim=-1)
            ga = ga.T @ ga / ga.shape[0] * 1024
            gb = gb.T @ gb / gb.shape[0] * 1024
            loss_style = 100000 *nn.functional.mse_loss(ga,gb)
            loss += loss_style
            losses.update(dict(loss_style=loss_style.item()))

            # Color Dist
            color_img = real_video_aug.mean(dim=(2,3)) # N X 3
            color_syn = syn_video_aug.mean(dim=(2,3)) # N X 3
            dist_loss = nn.functional.mse_loss(color_img.mean(0),color_syn.mean(0)) \
                        + nn.functional.mse_loss(
                            color_img.T @ color_img / color_img.shape[0],
                            color_syn.T @ color_syn / color_syn.shape[0],
                        )
            loss += dist_loss
            losses.update(dict(dist_loss=dist_loss.item()))

            # smmoothness N X C X H X W
            syn_dx = color_syn[...,:,1:]-color_syn[...,:,:-1]
            syn_dy = color_syn[...,1:,:]-color_syn[...,:-1,:]
            gradient = (syn_dx ** 2).mean() + (syn_dy ** 2).mean()
            loss += 1e-5 * dist_loss
            losses.update(dict(gradient=gradient.item()))
            

            

        reconst_loss = torch.FloatTensor([0]).mean().to(device)
        if args.method == 'FCN':
            reconst = fcn(encoder(real_video_aug)['out'])
            reconst_loss = nn.functional.mse_loss(reconst,real_video_aug) 
            loss+= reconst_loss
            losses.update(dict(reconst_loss=reconst_loss.item()))
            optimizer_model.zero_grad()
        optimizer.zero_grad()
        optimizer_k.zero_grad()
        loss.backward()
        dist.all_reduce(syn_batch.grad.data, op=dist.reduce_op.SUM)
        syn_batch.grad.data = syn_batch.grad.data / world_size
        optimizer.step()
        optimizer_k.step()
        if args.method == 'FCN':
            optimizer_model.step()
        syn_batch_momentum = optimizer.state[syn_batch]['momentum_buffer'].detach().cpu()
        # manual SGD
    
        # Actual update of model
        # TODO: Check manual DDP is equivalent to actual DDP
        if iter % 10 == 0 and rank == 0:
            if iter % 200 == 0 :
                # log image # B X N X C X T X H X W
                img = syn_video[:].detach().cpu().numpy().transpose(0,2,3,1) * 0.22 + 0.45
                wandb_dump_img(img[:4],'img_syn')
                img = real_video_aug[:].detach().cpu().numpy().transpose(0,2,3,1) * 0.22 + 0.45
                wandb_dump_img(img[:4],'img_real')
                if args.method == 'FCN':
                    img = reconst[:].detach().cpu().numpy().transpose(0,2,3,1) * 0.22 + 0.45
                    wandb_dump_img(img[:4],'img_reconst')
            wandb.log(
                dict(
                    epoch=epoch,
                    lr=new_lr,
                    iter=iter,
                    loss=loss.item(),
                    **losses,
                )
            )
            eta_seconds =(time.time()-start_time) * ((epoches*total_iter) / (epoch*total_iter+iter+1) -1 )
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(f'ETA {eta_string}||{epoch}:{iter}/{total_iter}\t Match Loss {match_loss.item()} ')
        # Save sampled data
        #tensor_list = [torch.zeros_like(syn_batch, dtype=syn_batch.dtype,device=syn_batch.device) for _ in range(world_size)]
        #dist.all_gather(tensor_list, syn_batch)
        video_syn[all_idx] = syn_batch.detach().clone().to(video_syn.device)
        video_syn_momentum[all_idx] =syn_batch_momentum.detach().clone().to(video_syn.device)
        iter += 1

        torch.cuda.empty_cache()
    # todo: Save model state parms, and add eval pipeline
    if epoch % 50 == 0:
        torch.save(model.state_dict(),os.path.join(ROOT_DIR,f'checkpoint_{epoch}.pth'))
        torch.save(video_syn,os.path.join(ROOT_DIR,f'syn_videos_{epoch}.pth'))