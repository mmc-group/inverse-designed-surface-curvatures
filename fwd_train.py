import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
import torchvision
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from core import *
from collections import OrderedDict

# forward training may take long, be patient

cuda_device = 0
torch.cuda.set_device(cuda_device)
datatype = torch.float32

#=============import dataset====================
num_samples = 20000
test_samples = 2000
train_samples = int(num_samples-test_samples)

x_arrayset = np.load(os.getcwd() + '/data/x_arrayset.npy')
x_arrayset = curvature_normalization().normalize(x_arrayset)
x_arrayset = from_200_200_to_20100(x_arrayset) #(num_samples,20100) normalized curvature map
y_arrayset = np.load(os.getcwd() + '/data/y_arrayset.npy')#(20000,7)

# switch x and y, x: design parameters, y: curvature maps
y_train = torch.tensor(x_arrayset[:train_samples]).type(datatype).to(cuda_device)
y_test = torch.tensor(x_arrayset[train_samples:]).type(datatype).to(cuda_device)

# design parameters and normalization
all_label = torch.tensor(y_arrayset).type(datatype).to(cuda_device)
data = design_para_normalization(all_label)
all_label_normalization = data.normalize(all_label)
x_train = all_label_normalization[:train_samples]
x_test = all_label_normalization[train_samples:]

dataset = TensorDataset(x_train, y_train)
BATCH_SIZE = 128
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#==========free some CUDA memory==============
del x_arrayset
del y_arrayset
del all_label
del data
del all_label_normalization
del dataset

os.makedirs(os.getcwd() + '/model/trained_model', exist_ok=True)
os.makedirs(os.getcwd() + '/model/loss_history', exist_ok=True)
#=========forward model=========================
forward_model = forward()

# load partly trained model
resume = False

if resume:
    forward_path = os.getcwd() + '/model/trained_model/forward.pt'
    state_dict = torch.load(forward_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    forward_model.load_state_dict(new_state_dict)
    forward_model.to(cuda_device).to(datatype)
    forward_model.eval()

# train from scratch
else:
    forward_model.to(cuda_device).to(datatype)

forward_model = torch.nn.DataParallel(forward_model, device_ids=[0,1,2])

#============== start training ========================
N_EPOCHS = 300
lr = 0.0001
loss_history_train = []
loss_history_test = []
weight_decay = 1e-8
optimizer = optim.Adam(forward_model.parameters(), lr=lr, weight_decay = weight_decay)
for epoch in range(N_EPOCHS):

    for id_batch, (x_batch, y_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        y_batch_pred = forward_model(x_batch)
        y_test_pred = forward_model(x_test)
        loss = torch.mean((y_batch_pred - y_batch)**2)
        loss.backward()
        optimizer.step()

    y_train_epoch = forward_model(x_train)
    y_test_epoch = forward_model(x_test)
    loss_train_epoch = torch.mean((y_train_epoch- y_train)**2)
    loss_test_epoch = torch.mean((y_test_epoch- y_test)**2)
    loss_history_train.append(loss_train_epoch.detach().item())
    loss_history_test.append(loss_test_epoch.detach().item())

    print(f"Epoch {epoch + 1}\n-------------------------------")
    print('loss train={:3e}, test = {:3e}'.format(loss_train_epoch.item(),loss_test_epoch.item()))

    #------------free some CUDA memory-----------
    del loss
    del y_train_epoch
    del y_test_epoch
    del loss_train_epoch
    del loss_test_epoch
    del y_batch_pred
    del y_test_pred

    #----------- save model and loss data every 10 epochs-------
    if (epoch+1) % 10 == 0:
        torch.save(forward_model.state_dict(),os.getcwd() + '/model/trained_model/forward.pt')
        np.save(os.getcwd() +'/model/loss_history/loss_train_forward.npy',np.array(loss_history_train))
        np.save(os.getcwd() +'/model/loss_history/loss_test_forward.npy',np.array(loss_history_test))

print('Training finished')
torch.cuda.empty_cache()
