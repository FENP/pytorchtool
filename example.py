import os
import sys
sys.path.append('..')
import time
import torch
import pytorchtool
import numpy as np

from classes import class_names

from PIL import Image
from torchvision import models, transforms

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def process_img(path_img):
    # hard code
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # tensor
    img_tensor = inference_transform(Image.open(path_img).convert('RGB'))
    img_tensor.unsqueeze_(0)        # chw --> bchw
    
    return img_tensor

class model:
    def __init__(self, model_name, use_gpu=False):
        self.model_name = model_name
        self.x = process_img('./pandas.jpg')
        self.use_gpu = use_gpu

        if self.model_name in 'inception':
            self.model_name = 'inception'
            self.path = "./model_weight/inception_v3/inception_v3_google-1a9a5a14.pth"

            model = models.Inception3(aux_logits=False, transform_input=False, 
                                    init_weights=False)
            model.eval()
            self.model = model
        elif self.model_name in 'alexnet':
            self.model_name = 'alexnet'
            self.path = "./model_weight/alexnet/alexnet-owt-4df8aa71.pth"
            
            model = models.alexnet(False)
            model.eval()
            self.model = model 
        else:
            print("Wrong model name")

        if self.use_gpu:
            self.model = self.model.to(0)
            self.x = self.x.cuda()

    def load_weight(self):
        state_dict_read = torch.load(self.path)

        self.model.load_state_dict(state_dict_read, strict=False)

    def get_model(self):
        return self.model
    
    def get_input(self):
        return self.x
    
    def save_layers(self, depth=-1):
        pytorchtool.save_model(self.model, depth=depth)
    
    def inference(self):
        with torch.no_grad():
            outputs = self.model(self.x)

        print("result: " + class_names[torch.argmax(outputs, 1)[0]])

    def prof(self, depth=-1):
        with pytorchtool.Profile(self.model, use_cuda=self.use_gpu, 
                depth=depth) as prof:
            self.model(self.x)

        if self.use_gpu:
            prof.printCsv("./parameters/" + self.model_name + "GPU.csv")
        else:
            prof.printCsv("./parameters/" + self.model_name + "CPU.csv")


if __name__ == "__main__":
    name = "alex"
    m = model(name)
    m.prof(depth=-1)
    m.inference()
    #m.load_weight()
    #m.save_layers(depth=-1)
