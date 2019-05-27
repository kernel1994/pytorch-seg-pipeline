import cfg
from model import unet
from torchsummary import summary


def creat():
    if cfg.model_name == 'unet':
        model = unet.UNet(cfg.num_class).to(cfg.device)
    else:
        raise RuntimeError(f'Error model name {cfg.model_name}')

    summary(model, input_size=(3, 224, 224))

    return model
