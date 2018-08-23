import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn

from torchvision import models
import pretrainedmodels
import os
from PIL import Image

#import sys
#sys.path.append('test')

from test import *

class Extractor:
    def __init__(self):
        # TODO
        #   - initialize and load model here
        code_path = os.path.dirname(os.path.abspath(__file__))

        #self.model = os.path.join(code_path, 'resnet50.pth.tar')
        self.model = os.path.join(code_path,'extractor_model.pth')
        self.result = None

        self.model = torch.load(self.model)
        self.model = torch.nn.DataParallel(self.model).cuda()



    def inference_by_path(self, image_path):
        result = []

        # TODO
        #   - Inference using image path

        Resize = trn.Compose([
            trn.Resize((224,224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = Image.open(image_path)
        img = img.convert("RGB")

        input_img = V(Resize(img).unsqueeze(0), volatile=True)

        result= self.model.forward(input_img)

        self.result={'Pool5':result[0].data.cpu().numpy().tobytes(),
                     'MAC':result[1].data.cpu().numpy().tobytes(),
                     'SPoc':result[2].data.cpu().numpy().tobytes(),
                     'RMAC':result[3].data.cpu().numpy().tobytes(),
                     'RAMAC':result[4].data.cpu().numpy().tobytes()}

        return self.result

if __name__=='__main__':




    a=Extractor()
    ret= a.inference_by_path('./444562.jpg')
    import numpy as np
    np.set_printoptions(threshold=np.inf)
    print(np.frombuffer(ret['Pool5'],dtype=np.float32))
