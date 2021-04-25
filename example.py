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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
            self.depth = 2
        elif self.model_name in 'alexnet':
            self.model_name = 'alexnet'
            self.path = "./model_weight/alexnet/alexnet-owt-4df8aa71.pth"
            
            model = models.alexnet(False)
            model.eval()
            self.model = model
            self.depth = -1 
        else:
            print("Wrong model name")

        if self.use_gpu:
            self.model = self.model.to(0)
            # self.x = self.x.cuda()
            self.x = self.x.to(0)

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

        if not os.path.exists("./parameters/" + self.model_name):
            os.makedirs("./parameters/" + self.model_name)

        if self.use_gpu:
            prof.printCsv("./parameters/" + self.model_name + "/gpuPart.csv")
        else:
            prof.printCsv("./parameters/" + self.model_name + "/cpuPart.csv")


if __name__ == "__main__":
    torch.randn(4).to(0)

    name = "in"
    start_init = time.time()
    m = model(name, use_gpu=True)
    print("模型结构初始化时间: ", time.time() - start_init)
    start_load = time.time()
    m.load_weight()
    print("模型参数加载时间: ", time.time() - start_load)

    doPrepare = False
    doProf = True
    doInference = True
    doPartition = True

    if doPrepare:
        m.save_layers(depth=m.depth)
    elif doProf:
        m.prof(depth=m.depth)
        m.prof(depth=m.depth)
    elif doInference:
        start = time.time()
        m.inference()
        print("推理时间", time.time() - start)
    elif doPartition:
        '''
        使用Alexnet进行了切分测试
        '''
        cModel = pytorchtool.Surgery(m.model, 0, depth=m.depth)
        cModel.setLayerState({"input": 1, "features.0": 2, "features.1": 2, "features.2": 2, "features.3": 2,
                            "features.4": 2, "features.5": 2, "features.6": 2, "features.7": 2,
                            "features.8": 2, "features.9": 2, "features.10": 2, "features.11": 2,
                            "features.12": 2, "avgpool": 2, "classifier.0": 2, "classifier.1": 2,
                            "classifier.2": 2, "classifier.3": 2, "classifier.4": 2, "classifier.5": 2,
                            "classifier.6": 2, 'flatten': 2})
        cModel.clearMiddleResult()
        cModel(m.x)
        cModel.recover() # 恢复m的forward函数，避免sModel对同一个模型嵌套修改
        print(cModel.getMiddleResult())

        sModel = pytorchtool.Surgery(m.model, 2, depth=m.depth)
        sModel.setLayerState({"input": 1, "features.0": 2, "features.1": 2, "features.2": 2, "features.3": 2,
                            "features.4": 2, "features.5": 2, "features.6": 2, "features.7": 2,
                            "features.8": 2, "features.9": 2, "features.10": 2, "features.11": 2,
                            "features.12": 2, "avgpool": 2, "classifier.0": 2, "classifier.1": 2,
                            "classifier.2": 2, "classifier.3": 2, "classifier.4": 2, "classifier.5": 2,
                            "classifier.6": 2, 'flatten': 2})
        '''
        这里使用随机生成的相同size的数据代替原始输入数据，
        实际使用时若将计算全部卸载到了服务端，则需要传入原始数据
        '''
        sModel.setMiddleResult(cModel.getMiddleResult())
        outputs = sModel(torch.rand(224, 224).unsqueeze_(0))
        print("result: " + class_names[torch.argmax(outputs, 1)[0]])
