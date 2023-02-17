from model.unet_autoenc import BeatGANsAutoencModel,BeatGANsAutoencConfig
from templates import ddpm
from experiment import *


def autoenc_base():
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf

# conf = BeatGANsAutoencConfig(
#                 attention_resolutions=(16, ),
#                 channel_mult=(1, 2, 4, 8),
#                 conv_resample=True,
#                 dims=2,
#                 dropout= 0.1,
#                 embed_channels=512,
#                 enc_out_channels=512,
#                 enc_pool='adaptivenonzero',
#                 enc_num_res_block=2,
#                 enc_channel_mult=(1, 2, 4, 8, 8),
#                 enc_grad_checkpoint=False,
#                 enc_attn_resolutions=None,
#                 image_size=224,
#                 in_channels=3,
#                 model_channels=self.net_ch,
#                 num_classes=None,
#                 num_head_channels=-1,
#                 num_heads_upsample=-1,
#                 num_heads=self.net_beatgans_attn_head,
#                 num_res_blocks=self.net_num_res_blocks,
#                 num_input_res_blocks=self.net_num_input_res_blocks,
#                 out_channels=self.model_out_channels,
#                 resblock_updown=self.net_resblock_updown,
#                 use_checkpoint=self.net_beatgans_gradient_checkpoint,
#                 use_new_attention_order=False,
#                 resnet_two_cond=self.net_beatgans_resnet_two_cond,
#                 resnet_use_zero_module=self.
#                 net_beatgans_resnet_use_zero_module,
#                 latent_net_conf=latent_net_conf,
#                 resnet_cond_channels=self.net_beatgans_resnet_cond_channels,
#             )

conf = autoenc_base()
breakpoint()
model = BeatGANsAutoencModel(conf.model_conf)
breakpoint()