import one_gpu
# device, CPU or GPU
device = one_gpu.device

# contain intermediate and final outcomes of training and prediction
main_path = '/path/to/main/'
# model name
model_name = 'unet'
# number of class
num_class = 6
# number of epochs
num_epochs = 40
# batch size
batch_size = 32
# best weights file path
weights_path = './best_weights.pt'
# learning rate
lr = 1e-4
