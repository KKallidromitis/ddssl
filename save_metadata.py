import torch
import os
import copy
import torch.nn as nn
from tqdm import tqdm
from torchvision import datasets, transforms
from utils import get_dataset, get_network, get_daparam,TensorDataset, epoch, ParamDiffAug
from lib.data.kinectics import KinecticsWrapper
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
transform = transforms.Compose([transforms.ToPILImage(),transforms.Normalize(mean=mean, std=std),transforms.Resize(im_size),transforms.CenterCrop(im_size)])
#dst_train = KinecticsWrapper(args.data_path,split='train_256',frames_per_clip=10)
dst_val = KinecticsWrapper(args.data_path,split='val_256',frames_per_clip=10)
breakpoint()
dst_val.transform = None
torch.save(dst_val,'cache_val.pth')
# dst_train = KinecticsWrapper(args.data_path,split='train_256',frames_per_clip=10)

# dst_test = KinecticsWrapper(args.data_path,frames_per_clip=10)
