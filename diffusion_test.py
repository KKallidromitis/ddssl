from templates import *
import torch
from diffusion.base import DummyModel
device = 'cuda'
conf = ffhq256_autoenc()
conf.fp16 = False
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'../diffae/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device);
x = torch.randn(2,3,256,256).to(device)
cond = model.encode(x)
t, weight = model.T_sampler.sample(len(x), x.device)
xT = th.randn_like(x)
model.to(device)
losses = model.sampler.training_losses(model=model.model,x_start=x,t=t,model_kwargs=dict(cond=cond))
# 
cond.requires_grad=True
model.model.eval()
r = model.model.forward(x=xT.detach(),
                      t=model.sampler._scale_timesteps(t),
                      cond=cond)
rr = model.sampler.p_mean_variance(
                model=DummyModel(pred=r.pred),
                # gradient goes through x_t
                x=xT,
                t=t,
                clip_denoised=False)
rr['pred_xstart']
g =  torch.autograd.grad(rr['pred_xstart'].sum(),cond)
sampler = model.conf._make_diffusion_conf(20).make_sampler()
from lib.diffusion import DiffusionAPI
