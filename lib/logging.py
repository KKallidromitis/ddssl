import wandb
from matplotlib import pyplot as plt


def wandb_dump_img(imgs,category):
    n_imgs = len(imgs)
    fig, axes = plt.subplots(1,n_imgs,figsize=(2*n_imgs, 2))
    #raw, kmeans on 
    fig.tight_layout()
    for idx,img in enumerate(imgs):
        axes[idx].imshow(img)
    wandb.log({category:wandb.Image(fig)}) 
    plt.close(fig)