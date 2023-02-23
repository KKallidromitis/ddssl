import torch
from torch import nn
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter

class ResNetFeatuerizer(nn.Module):

    def __init__(self,pretrained='/shared/jacklishufan/mmcls/slotcon_imagenet_r50_200ep.pth') -> None:
        super().__init__()
        backbone = models.resnet50(pretrained=None)
        if pretrained is not None:
            v = torch.load(pretrained)
            if 'state_dict' in v:
                v = v['state_dict']
            r = backbone.load_state_dict(v,strict=False)
            print(r)
        self.backbone = IntermediateLayerGetter(backbone, return_layers=
            {'layer4':'c5','layer3':'c4','layer2':'c3'}
        )
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        

    def forward(self,x) -> torch.Tensor:
        features = self.backbone(x)
        features = torch.cat(
            [self.avgpool(v) for k,v in features.items()],dim=1
        )
        return features # N,3584