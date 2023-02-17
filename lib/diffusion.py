from templates import *
import torch
from diffusion.base import DummyModel
from torch import nn

class DiffusionAPI:

    @staticmethod
    def build_diffusion():
    # device = 'cuda'
        conf = ffhq256_autoenc()
        conf.fp16 = False
    # print(conf.name)
        model = LitModel(conf)
        return model

    @staticmethod
    def sample_noise(batch_size,device):
        pass
    
    @staticmethod
    def forward_loss(model,x,cond=None):
        t, weight = model.T_sampler.sample(len(x), x.device)
        if cond is None:
            cond = model.encode(x)
        losses = model.sampler.training_losses(model=model.model,x_start=x,t=t,model_kwargs=dict(cond=cond))
        losses.update(cond=cond)
        return losses
    @staticmethod
    def decode(model,cond,t=800):
        batch_size = cond.shape[0]
        xT = torch.randn(batch_size,3,256,256,device=cond.device)
        t = torch.LongTensor([t,]*batch_size).to(cond.device)
        r = model.model.forward(x=xT.detach(),
                      t=model.sampler._scale_timesteps(t),
                      cond=cond)
        rr = model.sampler.p_mean_variance(
                        model=DummyModel(pred=r.pred),
                        # gradient goes through x_t
                        x=xT,
                        t=t,
                        clip_denoised=False)
        return rr['pred_xstart'] # N X C X H X W
    @staticmethod
    def decode_step(model,cond,T=2):
        batch_size = cond.shape[0]
        sampler = model.conf._make_diffusion_conf(T).make_sampler()
        xT = torch.randn(batch_size,3,256,256,device=cond.device)
        for idx in reversed(range(T)):       
            t = torch.LongTensor([T,]*batch_size).to(cond.device)
            rr = sampler.p_sample(
                            model=model.model,
                            # gradient goes through x_t
                            x=xT,
                            model_kwargs=dict(cond=cond),
                            t=torch.LongTensor([idx,]*batch_size).to(cond.device),
                            clip_denoised=False)
            xT = rr['sample'] # N X C X H X W
        return xT

# Diffisuion VAE
class DiffusionModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model = DiffusionAPI.build_diffusion()
        self.linear = nn.Linear(512,512*2)

    def encode(self,x):
        cond = self.model.encode(x)
        mean_std = self.linear(cond)
        return mean_std[...,:512],mean_std[...,512:]

    def decode(self,sample):
        result = DiffusionAPI.decode_step(self.model,sample,3)
        return result

    def kl_divergence(self,mean,log_sigma2)-> torch.Tensor:
        return - 0.5 * (1 + 2 * log_sigma2 - mean**2-log_sigma2.exp())
    
    def forward_loss(self,x):
        mean,log_sigma2 = self.encode(x)
        p = torch.rand_like(mean,device=mean.device)
        sample = p * (log_sigma2 * 0.5 ).exp() + mean
        reconst = self.decode(sample)
        l2_loss = nn.functional.mse_loss(reconst,x)
        d_kl = self.kl_divergence(mean,log_sigma2).sum(-1).mean()
        diffusion_loss = DiffusionAPI.forward_loss(self.model,x,sample)
        vae_loss = l2_loss + d_kl
        diffusion_loss.update(dict(l2_loss=l2_loss,vae_loss=vae_loss,d_kl=d_kl,reconst=reconst))
        diffusion_loss['loss'] = diffusion_loss['loss'].mean()+ 0.5 * vae_loss 
        return diffusion_loss
        
# state = torch.load(f'last.ckpt', map_location='cpu')
# model.load_state_dict(state['state_dict'], strict=False)
# model.ema_model.eval()
# model.ema_model.to(device);
# x = torch.randn(2,3,256,256).to(device)
# cond = model.encode(x)
# t, weight = model.T_sampler.sample(len(x), x.device)
# xT = th.randn_like(x)
# model.to(device)
# losses = model.sampler.training_losses(model=model.model,x_start=x,t=t,model_kwargs=dict(cond=cond))
# # 
# model.eval()
# r = model.model.forward(x=xT.detach(),
#                       t=model.sampler._scale_timesteps(t),
#                       cond=cond)
# rr = model.sampler.p_mean_variance(
#                 model=DummyModel(pred=r.pred),
#                 # gradient goes through x_t
#                 x=xT,
#                 t=t,
#                 clip_denoised=False)
# rr['pred_xstart']
