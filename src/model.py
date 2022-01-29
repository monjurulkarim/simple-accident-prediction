# import torch
import torch.nn as nn
import torchvision.models as models
import torch
from torch.autograd import Variable
import torch.nn.functional as F

device = ("cuda" if torch.cuda.is_available() else "cpu")


#Feature extract
class FeatureExtractor(nn.Module):
    def __init__(self, num_classes, device, output_dim, extractor):
        super(FeatureExtractor,self).__init__()
        self.feat_extractor = extractor
        if self.feat_extractor =='resnet50':
            self.resnet = models.resnet50(pretrained=True) #for transfer learning
            self.resnet.fc = nn.Sequential(
                           nn.Linear(2048, out_features= output_dim))
        elif self.feat_extractor == 'vgg16':
            self.resnet = models.vgg16(pretrained=True) #for transfer learning
            self.resnet.classifier[6] = nn.Linear(in_features=4096,out_features=output_dim)
        else:
            raise NotImplementedError

    def forward(self,x):
        x = self.resnet(x)
        return x


#Recurrent Neural Network
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers,dropout, network='gru'):
        super(GRUNet, self).__init__()
        self.network = network
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = dropout
        # self.dense1 = torch.nn.Linear(hidden_dim, 64) #64
        if self.network == 'gru':
            self.dense1 = torch.nn.Linear(hidden_dim, 64) #64
        elif self.network == 'cnn':
            self.dense1 = torch.nn.Linear(hidden_dim+hidden_dim, 64) #64
        self.relu = nn.ReLU()
        self.dense2 = torch.nn.Linear(64, output_dim) #64
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        if self.network == 'gru':
            out, h = self.gru(x, h)
        elif self.network == 'cnn':
            out = x
        out = F.dropout(out[:,-1],self.dropout[0])

        out = self.relu(self.dense1(out))
        out = F.dropout(out,self.dropout[1])
        out = self.dense2(out)
        out = self.logsoftmax(out)
        return out, h

class AccidentXai(nn.Module):
    def __init__(self, num_classes, h_dim, z_dim, n_layers, dropout, extractor, loss, network):
        super(AccidentXai,self).__init__()

        self.network = network
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers= n_layers
        self.num_classes= num_classes
        self.feat_extractor = extractor
        self.features = FeatureExtractor(num_classes, device, h_dim+h_dim, self.feat_extractor)
        self.gru_net = GRUNet(h_dim+h_dim, h_dim, 2,n_layers,dropout, self.network)
        self.loss_type = loss
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')


    def forward(self,x,y,toa):
        losses = {'total_loss': 0}
        all_output, all_hidden = [], []
        h = Variable(torch.zeros(self.n_layers, x.size(0), self.h_dim))
        h = h.to(x.device)
        for t in range(x.size(1)):
            x_t = self.features(x[:,t])
            x_t = torch.unsqueeze(x_t,1)
            # print('x_t shape: ', x_t.shape)
            output, h = self.gru_net(x_t,h)
            #computing losses
            if self.loss_type == 'exponential':
                L1 =self._exp_loss(output,y,t,toa=toa,fps=10.0)
            elif self.loss_type == 'crossentropy':
                target_cls = y[:, 1]
                target_cls = target_cls.to(torch.long)
                L1 = self.ce_loss(output, target_cls)
            else:
                raise ValueError('Select loss function correctly.')
            losses['total_loss']+=L1
            all_output.append(output) #TO-DO: all hidden
        return losses, all_output



    def _exp_loss(self, pred, target, time, toa, fps=20.0):
            '''
            :param pred:
            :param target: onehot codings for binary classification
            :param time:
            :param toa:
            :param fps:
            :return:
            '''

            target_cls = target[:, 1]
            target_cls = target_cls.to(torch.long)
            penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (toa.to(pred.dtype) - time - 1) / fps)
            pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
            # negative example
            neg_loss = self.ce_loss(pred, target_cls)

            loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
            return loss
