# import torch
import torch.nn as nn
import torchvision.models as models
import torch

device = ("cuda" if torch.cuda.is_available() else "cpu")

class CrackClassifier(nn.Module):
    def __init__(self, num_classes, device):
        super(CrackClassifier,self).__init__()
        self.resnet = models.resnet50(pretrained=True) #for transfer learning
        for param in self.resnet.parameters():
            param.requires_grad= False


        self.resnet.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, num_classes)).to(device)


    def forward(self,x):
        x = self.resnet(x)
        return x


class CrackXai(nn.Module):
    def __init__(self, num_classes):
        super(CrackXai,self).__init__()

        self.num_classes= num_classes
        self.crack = CrackClassifier(num_classes, device)
        # self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self,x):
        output = self.crack(x)
        # y = torch.squeeze(y)
        # loss = self.loss(output,y)

        return output
