import cfg
import torch
import numpy as np


def predict(model, test_data_loader):
    model.load_state_dict(torch.load(cfg.weights_path))

    # Set model to evaluate mode
    model.eval()

    for inputs, labels in next(iter(test_data_loader)):
        inputs = inputs.to(cfg.device)
        labels = labels.to(cfg.device)

        pred = model(inputs)
        pred = pred.data.cpu().numpy()

        np.save('./pred', pred)
