from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import glob
import cv2
import os, sys
import os.path as osp
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from natsort import natsorted
import yaml


with open('config.yml','r') as yamlfile:
    data = yaml.load(yamlfile,Loader=yaml.FullLoader)

cfg= data['NETWORK']
directory = data['DIRECTORY']

video_dir = directory['demo_dir']
num_classes = cfg['num_cls']
n_mean = cfg['n_mean']
n_std = cfg['n_std']
h_dim = cfg['h_dim']
z_dim = cfg['z_dim']
input_dim = cfg['input_dim']
n_layers = cfg['n_layers']
dropout = cfg['dropout']
extractor = cfg['extractor']
loss_type = cfg['loss_type']
network_type = cfg['network_type']
model_file = directory['best_weight']
destination_folder = directory['destination_dir']


def init_accident_model(model_file, num_classes, h_dim, z_dim, n_layers, extractor, loss_type):
    # building model

    model = AccidentXai(num_classes, h_dim, z_dim,n_layers, dropout, extractor, loss_type, network_type)
    # print(model)

    model = model.to(device=device)

    model.eval()
    # load check point
    model = load_checkpoint(model, model_file, isTraining=False)
    return model


def load_checkpoint(model, filename, isTraining=False):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        # start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint)
        # if isTraining:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model


def get_input_video(video_dir, n_frames,  device):
    transform = transforms.Compose(
            [
                transforms.Resize((input_dim[0], input_dim[1])),
                transforms.ToTensor(),
                transforms.Normalize((n_mean[0], n_mean[1], n_mean[2]), (n_std[0], n_std[1] ,n_std[2])),
            ]
        )

    images =[]
    video_path = natsorted(glob.glob(os.path.join(video_dir, '*.jpg')))
    for i in video_path:
        image_path = i
        # print('image_path :', image_path)
        image = Image.open(image_path)
        image = transform(image)

        images.append(image)
    x = torch.stack(images).to(device)
    x = torch.unsqueeze(x,0)
    return x

def parse_results(all_outputs, batch_size=1, n_frames=100):
    # parse inference results
    pred_score = np.zeros((batch_size, n_frames), dtype=np.float32)

    # run inference
    for t in range(n_frames):
        pred = all_outputs[t]  # B x 2
        pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
        pred_score[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
    return pred_score

def building_cam(model,methods,extractor, use_cuda):
    if extractor == 'resnet50':
        target_layers = [model.features.resnet.layer4[-1]] #for resnet
    elif extractor == 'vgg16':
        target_layers = [model.features.resnet.features[-1]] #for vgg
    else:
        raise NotImplementedError

    cam_algorithm = methods['gradcam']
    cam = cam_algorithm(model=model, target_layers = target_layers, use_cuda= use_cuda)

    return cam



def saliency_map(cam, video_dir, destination_dir):
    video_path = natsorted(glob.glob(os.path.join(video_dir, '*.jpg')))
    target_category = 1
    dim= (512,384)
    for img in video_path:

        image_path = img
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]

        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        #==================================
        # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
        # gb = gb_model(input_tensor, target_category=target_category)
        # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        # cam_gb = deprocess_image(cam_mask * gb)
        # gb = deprocess_image(cam_image)
        #=================================
        file_name = img.split('/')[-1]
        file_name = file_name
        file_save = os.path.join(destination_dir,file_name)
        resized = cv2.resize(cam_image,dim)

        # cv2.imwrite(file_save,cam_image)

        cv2.imwrite(file_save,resized)
    print('-----finished------')
    return



if __name__ == '__main__':
    from src.model import AccidentXai
    num_frames = 100

    device=("cuda" if torch.cuda.is_available() else "cpu")
    input_data = get_input_video(video_dir,num_frames,device)

    model = init_accident_model(model_file, num_classes,h_dim,z_dim,n_layers, extractor, loss_type)

    # Uncomment the below code block if you want to see the prediction probablity:
    #===================================================
    # labels = torch.Tensor([[0,1]]).to(device) #useless
    # toa = torch.Tensor([[45]]).to(device) #useless
    # with torch.no_grad():
    #     loss,output = model(input_data,labels,toa)
    #
    # pred_score= parse_results(output)
    # print(pred_score)
    # #===================================================


    #for grad-cam
    from pytorch_grad_cam import GradCAM, \
        ScoreCAM, \
        GradCAMPlusPlus, \
        AblationCAM, \
        XGradCAM, \
        EigenCAM, \
        EigenGradCAM
        # LayerCAM, \
        # FullGrad
    from pytorch_grad_cam import GuidedBackpropReLUModel
    from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                             deprocess_image, \
                                             preprocess_image

    methods = \
    {"gradcam": GradCAM,
     "scorecam": ScoreCAM,
     "gradcam++": GradCAMPlusPlus,
     "ablationcam": AblationCAM,
     "xgradcam": XGradCAM,
     "eigencam": EigenCAM,
     "eigengradcam": EigenGradCAM}
     # "layercam": LayerCAM,
     # "fullgrad": FullGrad}

    use_cuda = torch.cuda.is_available()
    cam = building_cam(model,methods,extractor,use_cuda)

    # cam = building_cam(model,GradCAM,methods,use_cuda)

    video_name = video_dir.split('/')[-1]
    destination_dir = destination_folder+ video_name
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    saliency_map(cam, video_dir, destination_dir)
