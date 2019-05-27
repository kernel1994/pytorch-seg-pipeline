import cfg
from dataset import dataset
from model import creat_model
from train import train_model
from prediction import predict


def main():
    # 1. data preparation
    dataloaders = dataset.get_dataloader()

    # 2. model creation
    model = creat_model.creat()

    # 3. model training
    model = train_model(model, dataloaders)

    # 4. data prediction
    predict(model, dataloaders['val'])

    # 5. index evaluation


if __name__ == "__main__":
    main()
