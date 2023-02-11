from mmaction.datasets import build_dataset
from mmaction.apis import train_model
from mmaction.models import build_model
from mmcv import Config, DictAction
import time
import os.path as osp
from mmaction.utils import collect_env, get_root_logger
from mmcv.runner import init_dist, set_random_seed
from torchreparam import ReparamModule
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import SGD
import torch.distributed as dist
import argparse
import wandb
import time
import datetime

from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='Detcon-BYOL Training')
parser.add_argument("--local_rank", metavar="Local Rank", type=int, default=0, 
                    help="Torch distributed will automatically pass local argument")
parser.add_argument("--cfg", metavar="Config Filename", default="train_imagenet_300", 
                    help="Experiment to run. Default is Imagenet 300 epochs")
parser.add_argument("--name", metavar="Log Name", default="", 
                    help="Name of wandb entry")


args = parser.parse_args()

import enum
import wandb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def wandb_dump_img(imgs,category):
    n_imgs = len(imgs)
    fig, axes = plt.subplots(1,n_imgs,figsize=(5*n_imgs, 5))
    #raw, kmeans on 
    fig.tight_layout()
    for idx,img in enumerate(imgs):
        axes[idx].imshow(img)
    wandb.log({category:wandb.Image(fig)}) 
    plt.close(fig)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=0,
        num_clips=2,
        out_of_bound_opt='loop'),
    dict(type='DecordDecode'),
    dict(
        type='RandomResizedCrop',
        area_range=(0.2, 1.),
        same_across_clip=False,
        same_on_clip=False),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(
        type='Flip',
        flip_ratio=0.5,
        same_across_clip=False,
        same_on_clip=False),
    # dict(
    #     type='ColorJitter',
    #     brightness=0.4,
    #     contrast=0.4,
    #     saturation=0.4,
    #     hue=0.1,
    #     p=0.8,
    #     same_across_clip=False,
    #     same_on_clip=False),
    # dict(
    #     type='RandomGrayScale',
    #     p=0.2,
    #     same_across_clip=False,
    #     same_on_clip=False),
    # dict(
    #     type='RandomGaussianBlur',
    #     p=0.5,
    #     same_across_clip=False,
    #     same_on_clip=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

class arguments:
    pass
args=arguments()

args.config = 'configs/r50_sgd_cos_100e_r5_1xNx2_k400.py'
distributed = False
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
env_info_dict = collect_env()
env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
cfg = Config.fromfile(args.config)
cfg.gpu_ids = range(1)
cfg.seed = 0
cfg.work_dir = './work'

meta = dict()
meta['env_info'] = env_info
meta['seed'] = cfg.seed
meta['exp_name'] = osp.basename(args.config)


import kornia
import numpy as np
from kornia.augmentation import VideoSequential
import wandb
CROP_SIZE = 100
aug_list = VideoSequential(
    kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
    kornia.color.BgrToRgb(),
    kornia.augmentation.RandomAffine(360, p=1.0),
    kornia.augmentation.CenterCrop(CROP_SIZE),
    kornia.augmentation.PadTo((CROP_SIZE,CROP_SIZE)),
    random_apply=10,
    data_format="BCTHW",
    same_on_frame=True)

# cfg.data.train['pipeline']=train_pipeline
SIZE = 2350   # 1p
indices = list(range(SIZE))
video_syn = torch.randn(size=(SIZE, 2, 3, 1, 224, 224), dtype=torch.float, requires_grad=False)
# label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,
def apply_kornia(aug,v,parms=None):
    v = v.permute(0,2,1,3,4,5)
    b,c,n,t,h,w = v.shape
    v = v.view(b,c,n*t,h,w)
    if parms is not None:
        v = aug_list(v,params=parms)
    else:
        v = aug_list(v)
    v = v.view(b,c,n,t,*v.shape[-2:]).permute(0,2,1,3,4,5)
    return v,aug._params

import os
cfg.data.train.dataset.pipeline=train_pipeline
datasets = [build_dataset(cfg.data.train)]
model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
#model = ReparamModule(model)
# model = DDP(model,delay_allreduce=True)

world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ['RANK'])# or args.local_rank
local_rank = int(os.environ.get('LOCAL_RANK', '0'))    
dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
print(f"init done {rank}")
sampler = torch.utils.data.DistributedSampler(
            datasets[0], num_replicas=world_size, rank=rank, shuffle=True
)


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis

def calc_match_loss(gw_syn, gw_real, dis_metric='ours'):
    dis = torch.tensor(0.0).to(gw_real[0].device)

    if dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'%dis_metric)

    return dis

#flat_param = model.flat_param
net_parameters = list(model.parameters())

optimizer_model = SGD(model.parameters(),lr=0.05, momentum=0.9, weight_decay=0.0001)
BATCH_SIZE = 64

dataloader = DataLoader(datasets[0],batch_size=BATCH_SIZE,sampler=sampler,num_workers=8,drop_last=True)

epoches = 100
device = 'cuda'
torch.cuda.set_device(local_rank)
model.to(device)
model.train()
np.random.seed(0)
total_iter = len(datasets[0]) // (BATCH_SIZE * world_size)


def set_batch_norm(model,on=True):
    for module in model.modules():
        if 'BatchNorm' in module._get_name():  #BatchNorm
            module.eval() # fix mu and sigma of every BatchNorm layer
start_time = time.time()
end = time.time()
assert BATCH_SIZE * world_size < len(indices) # Cannot have larger global batch size than syn dataset size
if local_rank == 0:
    wandb.init()

for epoch in range(epoches):
    sampler.set_epoch(epoch)
    iter = 0
    
    for batch in dataloader:
        model.train()
        #TODO: see if there is a smarter way
        idx = np.random.choice(indices,BATCH_SIZE * world_size,replace=False)
        # print(idx[:10]) sanity check to make sure they are in sync
        all_idx = idx
        idx = idx[BATCH_SIZE*local_rank:BATCH_SIZE*(local_rank+1)]
        syn_batch = video_syn[idx].detach().clone().to(device)
        syn_batch.requires_grad=True
        real_video_aug,parms = apply_kornia(aug_list,batch['imgs'].to(device))
        syn_video_aug,_ = apply_kornia(aug_list,syn_batch,parms)
        r = model(real_video_aug)
        # disable bn
        for module in model.modules():
            if 'BatchNorm' in module._get_name():  #BatchNorm
                module.eval() # fix mu and sigma of every BatchNorm layer
        r_syn = model(syn_video_aug)

        optimizer = SGD([syn_batch,],lr=0.05, momentum=0.9, weight_decay=0.0001)

        vfs_loss_real = r['img_head.0.loss_feat' ].mean()
        vfs_loss_syn = r_syn['img_head.0.loss_feat' ].mean()
        gw_real = torch.autograd.grad(vfs_loss_real, net_parameters,retain_graph=True)
        gw_real = list((x.detach() for x in gw_real))
        gw_syn = torch.autograd.grad(vfs_loss_syn, net_parameters, create_graph=True)

        # do match SGD
        match_loss = calc_match_loss(gw_real,gw_syn)
        
        optimizer.zero_grad()
        match_loss.backward()
        # syn_batch.grad.data = syn_batch.grad.data / world_size
        optimizer.step()
        
        # Actual update of model
        optimizer_model.zero_grad()
        # Manual DDP
        for (pm,pm_grad) in zip(net_parameters, gw_real):
            dist.all_reduce(pm_grad, op=dist.ReduceOp.SUM,async_op=True) # reduce and average gradient
            pm_grad /= world_size
            pm.grad = pm_grad

        optimizer_model.step()
        if iter % 10 == 0 and local_rank == 0:
            if iter % 200 == 0 :
                # log image # B X N X C X T X H X W
                img = syn_batch[0,:,:,0].detach().cpu().numpy().transpose(0,2,3,1)
                wandb_dump_img([img[0],img[1]],'img0')
            wandb.log(
                dict(
                    epoch=epoch,
                    iter=iter,
                    match_loss=match_loss.item(),
                    vfs_loss_real=vfs_loss_real.item(),
                    vfs_loss_syn = vfs_loss_syn.item()
                )
            )
            eta_seconds =(time.time()-start_time) * ((epoches*total_iter) / (epoch*total_iter+iter+1) -1 )
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(f'ETA {eta_string}||{epoch}:{iter}/{total_iter}\t Match Loss {match_loss.item()}\t VFS Loss {vfs_loss_real.item():2f}({vfs_loss_syn.item():2f})')
        # Save sampled data
        tensor_list = [torch.zeros_like(syn_batch, dtype=syn_batch.dtype,device=syn_batch.device) for _ in range(world_size)]
        dist.all_gather(tensor_list, syn_batch)
        video_syn[all_idx] = torch.cat(tensor_list,dim=0).detach().clone().to(video_syn.device)
        iter += 1

        torch.cuda.empty_cache()
    # todo: Save model state parms, and add eval pipeline
    if epoch % 1 == 0:
        torch.save(flat_param,f'checkpoint_{epoch}.pth')