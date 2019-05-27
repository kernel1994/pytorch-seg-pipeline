import copy
import time
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn.functional as F

import cfg
from loss import dice_loss


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice_loss_value = dice_loss(pred, target)

    loss = bce * bce_weight + dice_loss_value * (1 - bce_weight)

    metrics['bce_loss'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice_loss'] += dice_loss_value.data.cpu().numpy() * target.size(0)
    metrics['total_loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append('{}: {:4f}'.format(k, metrics[k] / epoch_samples))

    print('{}: {}'.format(phase, ', '.join(outputs)))


def train_model(model, dataloaders):
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(cfg.num_epochs):
        print('-' * 10)
        print(f'Epoch {epoch}/{cfg.num_epochs - 1}')

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print('LR', param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(cfg.device)
                labels = labels.to(cfg.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print('saving best model')
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # save best model weights
    torch.save(best_model_wts, cfg.weights_path)
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model
