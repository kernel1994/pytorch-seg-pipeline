from torchvision import transforms
from torch.utils.data import DataLoader

import cfg
from dataset.SimDataset import SimDataset


def get_dataloader():
    # use same transform for train/val for this example
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = SimDataset(2000, transform=trans)
    val_set = SimDataset(200, transform=trans)

    dataloaders = {
        'train': DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    }

    return dataloaders
