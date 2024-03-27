import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from core import *
from collections import OrderedDict


cuda_device = 0
torch.cuda.set_device(cuda_device)
datatype = torch.float32

# check data
num_samples = 20000
test_samples = 2000
train_samples = int(num_samples-test_samples)

# load dataset
x_arrayset = np.load(os.getcwd() + '/data/x_arrayset.npy')
x_arrayset = curvature_normalization().normalize(x_arrayset)
x_arrayset = from_200_200_to_20100(x_arrayset)#(num_samples,20100) normalized curvature map
y_arrayset = np.load(os.getcwd() +'/data/y_arrayset.npy')#(20000,7)

# switch x and y, x: design parameters, y: curvature maps
y_train = torch.tensor(x_arrayset[:train_samples]).type(datatype).to(cuda_device)#(18000,20100)
y_test = torch.tensor(x_arrayset[train_samples:]).type(datatype).to(cuda_device)#(18000,20100)

all_label = torch.tensor(y_arrayset).type(datatype).to(cuda_device)#(20000,7)
data = design_para_normalization(all_label)#(20000,7)
all_label_normalization = data.normalize(all_label)##(20000,7)
x_train = all_label_normalization[:train_samples]#(18000,7)
x_test = all_label_normalization[train_samples:]#(2000,7)

BATCH_SIZE = 128
dataset = TensorDataset(x_train, y_train)#(18000,7),(18000,20100)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# free some memory
del x_arrayset
del y_arrayset
del all_label
del data
del all_label_normalization
del dataset

# load well-trained forward model
forward_path = os.getcwd() + '/model/trained_model/forward.pt'
forward_model = forward()
state_dict = torch.load(forward_path)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_state_dict[k.replace("module.", "")] = v
forward_model.load_state_dict(new_state_dict)
forward_model.to(cuda_device)
forward_model.eval()

del state_dict
del new_state_dict

# load inverse model
inverse_model = inverse()

# load partly trained model
resume = False

if resume:
    inverse_path  = os.getcwd() + '/model/trained_model/inverse.pt'
    state_dict = torch.load(inverse_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    inverse_model.load_state_dict(new_state_dict)
    inverse_model.to(cuda_device)
    inverse_model.eval()
    del state_dict
    del new_state_dict
else:
    inverse_model.to(cuda_device)

inverse_model = torch.nn.DataParallel(inverse_model, device_ids=[0,1,2]) # if trained on multiple GPUs

lr = 0.00001
N_EPOCHS = 600
loss_history_train = []
loss_history_test = []
weight_decay = 1e-8
optimizer = optim.Adam(inverse_model.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in range(N_EPOCHS):

    for id_batch, (x_batch, y_batch) in enumerate(dataloader):#(128,7),(128,20100)
        optimizer.zero_grad()
        x_batch_pred = inverse_model(y_batch)
        y_batch_pred = forward_model(x_batch_pred)
        loss = torch.mean((y_batch_pred-y_batch)**2)

        loss.backward()
        optimizer.step()

    x_train_pred = inverse_model(y_train)
    x_test_pred = inverse_model(y_test)
    y_train_epoch = forward_model(x_train_pred)
    y_test_epoch = forward_model(x_test_pred)
    loss_train_epoch = torch.mean((y_train_epoch- y_train)**2)
    loss_test_epoch = torch.mean((y_test_epoch- y_test)**2)
    loss_history_train.append(loss_train_epoch.detach().item())
    loss_history_test.append(loss_test_epoch.detach().item())

    print(f"Epoch {epoch + 1}\n-------------------------------")
    print('loss train = {:3e}, test = {:3e}'.format(loss_train_epoch.item(),loss_test_epoch.item()))

    del loss
    del y_train_epoch
    del y_test_epoch
    del loss_train_epoch
    del loss_test_epoch
    del x_batch_pred
    del y_batch_pred
    del x_train_pred
    del x_test_pred

    if (epoch+1) % 10 == 0:
        torch.save(inverse_model.state_dict(),os.getcwd() + '/model/trained_model/inverse.pt')
        np.save(os.getcwd() +'/model/loss_history/loss_train_inverse.npy',np.array(loss_history_train))
        np.save(os.getcwd() +'/model/loss_history/loss_test_inverse.npy',np.array(loss_history_test))

torch.cuda.empty_cache()
