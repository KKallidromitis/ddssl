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

class Encoder(nn.Module):
    def __init__(self, z_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(784 + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 2 * z_dim),
        )

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        h = self.net(xy)
        # m, v = ut.gaussian_parameters(h, dim=1)
        #return m, v

class Decoder(nn.Module):
    def __init__(self, z_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 784)
        )

    def decode(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)
        
class VAE(nn.Module):
    def __init__(self, imgChannels=1, shape=(128, 8, 8), zDim=256):
        super(VAE, self).__init__()
        featureDim = int(np.product(shape))
        self.shape = shape
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encoder_blk = nn.Sequential(
            nn.Conv2d(3,1,1),
            nn.Flatten(),
            nn.Linear(32*32 , 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 2 * zDim),
        )
        # self.embed = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(featureDim, zDim*2),
        # )
        self.featureDim = featureDim
        self.zDim = zDim

        self.decoder_blk =  nn.Sequential(
            nn.Linear(zDim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 32*32),
        )

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.deconv = nn.Sequential(
            nn.Conv2d(1,3,1),
        )

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = self.encoder_blk(x)
        #print(x.shape)
        #x = self.embed(x)
        mu,logVar = x[...,:self.zDim],x[...,self.zDim:]
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = self.decoder_blk(z)
        #x = x.view(-1, *self.shape)
        x = x.view(-1, 1,32,32)
        x = self.deconv(x)
        #x = torch.sigmoid(x)
        return x

    def kl_divergence(self,mean,log_sigma2)-> torch.Tensor:
        return 0.5 * (-1 - log_sigma2 + mean.pow(2) + log_sigma2.exp())

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        if self.training:
            mu, logVar = self.encoder(x)
            z = self.reparameterize(mu, logVar)
            out = self.decoder(z)
            # breakpoint()
            d_kl = self.kl_divergence(mu,logVar).sum(-1).mean()
            #print(d_kl)
            l2_loss = nn.functional.mse_loss(out,x.detach())
            vae_loss =   l2_loss + 1 / (32*32*3)* d_kl
            return out, mu, logVar,vae_loss
        else:
            mu, logVar = x[...,:self.zDim],x[...,self.zDim:]
            #z = self.reparameterize(mu, logVar)
            d_kl = self.kl_divergence(mu,logVar).sum(-1).mean()
            out = self.decoder(mu)
            return out,1 * d_kl * 0

    def sample(self, n,device):
        z = torch.randn(n,self.zDim).to(device)
        out = self.decoder(z)
        return out

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
