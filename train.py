import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.model import AccidentXai
from src.vid_dataloader import MyDataset, MySampler
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import glob
import numpy as np
from src.eval_tools import evaluation_P_R80, print_results, vis_results
# import argparse
import yaml
from natsort import natsorted


#defining the parameters
#===========================================
with open('config.yml','r') as yamlfile:
    data = yaml.load(yamlfile,Loader=yaml.FullLoader)

cfg= data['NETWORK']
directory = data['DIRECTORY']
num_classes = cfg['num_cls']
learning_rate = cfg['lr']
batch_size = cfg['batch_size']
h_dim = cfg['h_dim']
z_dim = cfg['z_dim']
n_layers = cfg['n_layers']
num_epochs = cfg['epoch']
input_dim = cfg['input_dim']
n_mean = cfg['n_mean']
n_std = cfg['n_std']
loss_type = cfg['loss_type']
network_type = cfg['network_type']
extractor = cfg['extractor']
fps = cfg['fps']

gpu_id = cfg['gpu_id']
dropout = cfg['dropout']

train_data_path = directory['train_dir']
test_data_path = directory['test_dir']
model_dir = directory['model_dir']
logs_dir = directory['logs_dir']

transform = transforms.Compose(
        [
            transforms.Resize((input_dim[0], input_dim[1])),
            transforms.ToTensor(),
            transforms.Normalize((n_mean[0], n_mean[1], n_mean[2]), (n_std[0], n_std[1] ,n_std[2])),
        ]
    )

device = ("cuda" if torch.cuda.is_available() else "cpu")

os.environ['CUDA_VISIBLE_DEVICES']= gpu_id




#--------------train data----------------------------------------
train_class_paths = [d.path for d in os.scandir(train_data_path) if d.is_dir]


train_class_image_paths = []
train_end_idx = []
for c, class_path in enumerate(train_class_paths):
    for d in os.scandir(class_path):
        if d.is_dir:
            paths = natsorted(glob.glob(os.path.join(d.path, '*.jpg')))
            paths = [(p, c) for p in paths]

            train_class_image_paths.extend(paths)
            train_end_idx.extend([len(paths)])

train_end_idx = [0, *train_end_idx]
train_end_idx = torch.cumsum(torch.tensor(train_end_idx), 0)
seq_length = 99

train_sampler = MySampler(train_end_idx,seq_length)

##-------------Test data-------------------------------
test_class_paths = [d.path for d in os.scandir(test_data_path) if d.is_dir]

test_class_image_paths = []
test_end_idx = []
for c, class_path in enumerate(test_class_paths):
    for d in os.scandir(class_path):
        if d.is_dir:
            paths = natsorted(glob.glob(os.path.join(d.path, '*.jpg')))
            # Add class idx to paths
            paths = [(p, c) for p in paths]
            test_class_image_paths.extend(paths)
            test_end_idx.extend([len(paths)])

test_end_idx = [0, *test_end_idx]
test_end_idx = torch.cumsum(torch.tensor(test_end_idx), 0)
seq_length = 99

test_sampler = MySampler(test_end_idx,seq_length)


train_data = MyDataset(image_paths= train_class_image_paths,
    seq_length=seq_length,
    transform=transform,
    length=len(train_sampler))

test_data = MyDataset(image_paths= test_class_image_paths,
    seq_length=seq_length,
    transform=transform,
    length=len(test_sampler))

train_dataloader = DataLoader(dataset= train_data, batch_size=batch_size,sampler=train_sampler)
test_dataloader = DataLoader(dataset= test_data, batch_size=batch_size, sampler=test_sampler)


def write_scalars(logger, epoch, loss):
    logger.add_scalars('train/loss',{'loss':loss}, epoch)

def write_test_scalars(logger, epoch, losses, metrics):
    # logger.add_scalars('test/loss',{'loss':loss}, epoch)
    logger.add_scalars('test/losses/total_loss',{'Loss': losses}, epoch)
    logger.add_scalars('test/accuracy/AP',{'AP':metrics['AP'], 'PR80':metrics['PR80']}, epoch)
    logger.add_scalars('test/accuracy/time-to-accident',{'mTTA':metrics['mTTA'], 'TTA_R80':metrics['TTA_R80']}, epoch)



def test(test_dataloader, model):
    all_pred = []
    all_labels = []
    losses_all = []
    all_toas = []

    with torch.no_grad():
        loop = tqdm(test_dataloader,total = len(test_dataloader), leave = True)
        for imgs, labels, toa in loop:
            imgs = imgs.to(device)
            labels = torch.squeeze(labels)
            labels = labels.to(device)
            # outputs = model(imgs)
            loss, outputs = model(imgs,labels,toa)
            loss = loss['total_loss'].item()
            losses_all.append(loss)
            num_frames = imgs.size()[1]
            batch_size = imgs.size()[0]
            pred_frames = np.zeros((batch_size,num_frames),dtype=np.float32)
            for t in range(num_frames):
                pred = outputs[t]
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)

            #gather results and ground truth
            all_pred.append(pred_frames)
            label_onehot = labels.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size,])
            all_labels.append(label)
            toas = np.squeeze(toa.cpu().numpy()).astype(np.int)
            all_toas.append(toas)
            # loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(val_loss = np.mean(losses_all))

    all_pred = np.vstack((np.vstack(all_pred[0][:-1]), all_pred[0][-1]))
    all_labels = np.hstack((np.hstack(all_labels[0][:-1]), all_labels[0][-1]))
    all_toas = np.hstack((np.hstack(all_toas[0][:-1]), all_toas[0][-1]))

    return all_pred, all_labels, all_toas, losses_all




def train():

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logger = SummaryWriter(logs_dir)
    # x_dim = 2048 #2048 for resnet50


    model = AccidentXai(num_classes, h_dim, z_dim,n_layers,dropout, extractor, loss_type, network_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # print(model)


    #for transfer learning uncomment the following
    #==============================================================================
    if extractor == 'resnet50':
        for name, param in model.features.named_parameters():
            if "fc.0.weight" in name or "fc.0.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif extractor == 'vgg16':
        #For vgg16
        for name, param in model.features.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Train the GRU
        for name, param in model.gru_net.named_parameters():
            if 'gru.weight' in name or 'gru.bias' in name:
                param.requires_grad = True
                # print(name)
            elif 'dense1' in name or 'dense2' in name:
                param.requires_grad = True
                # print(name)
            else:
                param.requires_grad = False
    else:
        raise NotImplementedError
    #==============================================================================



    model.train()
    loss_best=100

    for epoch in range(num_epochs):
        loop = tqdm(train_dataloader,total = len(train_dataloader), leave = True)

        for imgs, labels, toa in loop:
            loop.set_description(f"Epoch  [{epoch+1}/{num_epochs}]")
            imgs = imgs.to(device)
            labels = torch.squeeze(labels)
            labels = labels.to(device)
            # outputs = model(imgs)
            loss, outputs = model(imgs,labels,toa)
            # loss = custom_loss(outputs, labels)
            optimizer.zero_grad()
            loss['total_loss'].mean().backward()
            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            optimizer.step()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss = loss['total_loss'].item())
            lr = optimizer.param_groups[0]['lr']

        write_scalars(logger,epoch,loss['total_loss'])
        #test and evaluate the model
        print('-------------------------------')
        print('------Starting evaluation------')
        model.eval()
        all_pred, all_labels, all_toas, losses_all = test(test_dataloader, model)
        total_loss = np.mean(losses_all)
        metrics = {}
        metrics['AP'], metrics['mTTA'], metrics['TTA_R80'], metrics['PR80']= evaluation_P_R80(all_pred, all_labels, all_toas, fps)
        write_test_scalars(logger,epoch,total_loss, metrics)
        model.train()
        # save model

        best_model_file = os.path.join(model_dir, 'best_model.pth')
        model_file = os.path.join(model_dir, 'saved_model_%02d.pth'%(epoch))
        torch.save(model.state_dict(),model_file)
        if total_loss < loss_best:
            loss_best = total_loss
            torch.save(model.state_dict(),best_model_file)
    logger.close()

if __name__ == "__main__":


    train()
