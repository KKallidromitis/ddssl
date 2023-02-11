import torch
import os
import copy
import torch.nn as nn
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data.dataloader import default_collate
from utils import get_dataset, get_network, get_daparam,TensorDataset, epoch, ParamDiffAug
from lib.data.kinectics import KinecticsWrapper

from torch.utils.data import DataLoader
from torchreparam import ReparamModule


def collate_fn(batch):
    # remove audio from the batch
    batch = [[d[0],d[2]] for d in batch]
    return default_collate(batch)

import kornia
from kornia.augmentation import VideoSequential
aug_list = VideoSequential(
    kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
    kornia.color.BgrToRgb(),
    kornia.augmentation.RandomAffine(360, p=1.0),
    kornia.augmentation.CenterCrop(100),
    random_apply=10,
    data_format="BTCHW",
    same_on_frame=True)
class arguments:
    pass

args = arguments()
args.zca = False
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.dsa_param = ParamDiffAug()
args.buffer_path = './buffers'
args.dataset = 'kinetics400'
args.data_path = '/shared/group/kinetics/'
args.batch_real = 10
args.subset = 'imagenette'
args.model = 'ConvNet'
args.batch_train = 10
args.num_experts = 100
args.lr_teacher = 0.01
args.mom = 0
args.l2 = 0
args.train_epochs = 50
args.dsa = True
args.dsa_strategy = 'color_crop_cutout_flip_scale_rotate'

im_size = (128, 128)
mean=[0.43216, 0.394666, 0.37645]
std=[0.22803, 0.22145, 0.216989]
num_classes=1
class_map = {0:1}
cached = torch.load('cache_train.pth')
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Normalize(mean=mean, std=std),
#     transforms.Resize(im_size),
#     transforms.CenterCrop(im_size)])
dst_train = KinecticsWrapper(args.data_path,split='train_256',
    frames_per_clip=10,_precomputed_metadata=cached.metadata,transform=None)

dataloader = DataLoader(dst_train,batch_size=4,collate_fn=collate_fn)

batch = next(iter(dataloader))

breakpoint()
# dst_test = KinecticsWrapper(args.data_path,frames_per_clip=10)