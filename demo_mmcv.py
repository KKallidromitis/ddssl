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
from torch import nn
from torch.optim import SGD

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

cfg.data.train.dataset.pipeline=train_pipeline
datasets = [build_dataset(cfg.data.train)]
model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
model = ReparamModule(model)
flat_param = model.flat_param
optimizer_model = SGD([flat_param],lr=0.05, momentum=0.9, weight_decay=0.0001)
BATCH_SIZE = 16

dataloader = DataLoader(datasets[0],batch_size=BATCH_SIZE)
epoches = 1
device = 'cuda'

model.to(device)
model.train()

total_iter = len(datasets[0]) // BATCH_SIZE
for epoch in range(epoches):
    iter = 0
    for batch in dataloader:
    # batch = next(iter(dataloader))
    # ### DO kornia augmentation
        
        idx = np.random.choice(indices,BATCH_SIZE,replace=False)
        syn_batch = video_syn[idx].detach().clone().to(device)
        syn_batch.requires_grad=True
        real_video_aug,parms = apply_kornia(aug_list,batch['imgs'].to(device))
        syn_video_aug,_ = apply_kornia(aug_list,syn_batch,parms)
        r = model(real_video_aug,flat_param=flat_param)
        r_syn = model(syn_video_aug,flat_param=flat_param)
        optimizer = SGD([syn_batch,],lr=0.05, momentum=0.9, weight_decay=0.0001)
        vfs_loss_real = r['img_head.0.loss_feat' ].mean()
        vfs_loss_syn = r_syn['img_head.0.loss_feat' ].mean()
        gw_real = torch.autograd.grad(vfs_loss_real, flat_param)[0].detach()
        gw_syn = torch.autograd.grad(vfs_loss_syn, flat_param, create_graph=True)[0]
        match_loss = nn.functional.mse_loss(gw_syn,gw_real)
        optimizer.zero_grad()
        match_loss.backward()
        optimizer.step()
        
        optimizer_model.zero_grad()
        r = model(real_video_aug,flat_param=flat_param)
        vfs_loss_real = r['img_head.0.loss_feat' ].mean()
        vfs_loss_real.backward()
        optimizer_model.step()
        if iter % 10 == 0:
            print(f'{epoch}:{iter}/{total_iter}\t Match Loss {match_loss.item()}\t VFS Loss {vfs_loss_real.item()}({vfs_loss_syn.item()})')
        # Save sampled data
        video_syn[idx] = syn_batch.detach().clone().to(video_syn.device)
        iter += 1
    # todo: Save model state parms, and add eval pipeline
    if epoch % 50 == 0:
        torch.save(flat_param,f'checkpoint_{epoch}.pth')