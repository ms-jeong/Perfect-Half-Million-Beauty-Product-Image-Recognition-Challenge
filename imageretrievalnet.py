import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import pretrainedmodels
from traindataset import myImageFloder
from pooling import MAC, SPoC, RMAC, RAMAC
from torchsummary import summary
import datetime


OUTPUT_DIM = {
    'alexnet'       :  256,
    'vgg11'         :  512,
    'vgg13'         :  512,
    'vgg16'         :  512,
    'vgg19'         :  512,
    'resnet18'      :  512,
    'resnet34'      :  512,
    'resnet50'      : 2048,
    'resnet101'     : 2048,
    'resnet152'     : 2048,
    'densenet121'   : 1024,
    'densenet161'   : 2208,
    'densenet169'   : 1664,
    'densenet201'   : 1920,
    'squeezenet1_0' :  512,
    'squeezenet1_1' :  512,
    'resnext101_64x4d' :  2048,
    'nasnetalarge' :  2048,
    'se_resnet101' :  2048,
}
class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'

class ImageRetrievalNet(nn.Module):
#class imageretrievalnet(nn.Module):
    def __init__(self, features, meta):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.norm = L2N()
        self.meta = meta

    
    def forward(self, x):
        # features -> pool -> norm
        x = self.features(x)
        feature_MAC = self.norm(MAC()(x)).squeeze(-1).squeeze(-1)
        feature_SPoC = self.norm(SPoC()(x)).squeeze(-1).squeeze(-1)
        feature_RMAC = self.norm(RMAC()(x)).squeeze(-1).squeeze(-1)
        feature_RAMAC = self.norm(RAMAC()(x)).squeeze(-1).squeeze(-1)

        return x,feature_MAC,feature_SPoC,feature_RMAC,feature_RAMAC
        #return x,feature_MAC.permute(1, 0), feature_SPoC.permute(1, 0), feature_RMAC.permute(1, 0), feature_RAMAC.permute(1, 0)





def init_network(model='vgg16'):
    
    net_in = pretrainedmodels.__dict__[model](num_classes=1000, pretrained='imagenet')
    mean=net_in.mean
    std=net_in.std

    if model.startswith('vgg'):
        net_in = getattr(torchvision.models, model)(pretrained=True)
        #features = list(list(net_in.children())[0][:-1])
        features = list(net_in.children())[0]
    elif model.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif model.startswith('resnext101_64x4d'):
        features = list(net_in.children())[:-2]
    elif model.startswith('se'):
        features = list(net_in.children())[:-2]
    else:
        raise ValueError('Unknown model: {}!'.format(model))
    
    dim = OUTPUT_DIM[model]

    # create meta information to be stored in the network
    meta = {'architecture':model, 'outputdim':dim, 'mean':mean, 'std':std}

    # create a generic image retrieval network
    net = ImageRetrievalNet(features, meta)

    return net

def extract_vectors(net, images, image_size, print_freq=100):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    
    normalize = torchvision.transforms.Normalize(mean=net.meta['mean'], std=net.meta['std'])
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

    # creating dataset loader
    batch_size=16
    loader = torch.utils.data.DataLoader(myImageFloder(images, transform, imsize=image_size), batch_size=batch_size, shuffle=False, num_workers=12)

    # extracting vectors
    name_list = []
    features=[]
    macs=[]
    rmacs=[]
    spocs=[]
    ramacs=[]
    for i, data in enumerate(loader):
        inputs, names = data
        input_var = Variable(inputs.cuda())

        feature,feature_MAC, feature_SPoC, feature_RMAC, feature_RAMAC = net(input_var)

        features.append(feature.data.cpu())
        macs.append(feature_MAC.data.cpu())
        rmacs.append(feature_RMAC.data.cpu())
        spocs.append(feature_SPoC.data.cpu())
        ramacs.append(feature_RAMAC.data.cpu())
        name_list.extend(names)


        if (i+1) % print_freq == 0 or (i+1) == len(images):
           print('\r>>>> {}/{} done...'.format((i+1)*batch_size, len(images)))



    #torch.save(net,'extractor_model.pth')



    return features,macs, rmacs, spocs, ramacs, name_list
