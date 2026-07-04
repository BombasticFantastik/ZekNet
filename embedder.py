import torch
from torch.nn import Module
import faiss
import onnxruntime as ort
import yaml
import cv2
import numpy as np

import PIL

option_path='config.yaml'
with open(option_path,'r') as file_option:
    files_option=yaml.safe_load(file_option)

class BuffaloModel(Module):
    def __init__(self, path,use_gpu=False):
        super().__init__()
        self.session=ort.InferenceSession(path)

        self.input_name=self.session.get_inputs()[0].name
        self.output_name=self.session.get_outputs()[0].name
    def forward(self,x):

        #разворот для буфало
        if x.ndim == 4:
            x = x[:, [2, 1, 0], :, :]
        else: # если пришла одна картинка [C, H, W]
            x = x[[2, 1, 0], :, :]

        x=x*255
        x = (x - 127.5) / 128.0
        x=x.numpy()
        output=self.session.run([self.output_name],{self.input_name:x})
        output=torch.from_numpy(output[0])
        output = torch.nn.functional.normalize(output, p=2, dim=1)
        return output
    
def compare_new_face(img,vectors,model,treshold=1.5):
    """
    ВХОД: Изображения лица,все вектора, модель для векторизации и порог отсечения фото
    ВЫХОД: Индекс наиболее схожего человека из переданного массива векторов
    """

    new_vector=model(img)
    new_vector=new_vector.numpy()

    indexer=faiss.IndexFlatL2(512)
    indexer.add(vectors)

    similarities, indices=indexer.search(x=new_vector,k=1)

    if similarities[0].item()<treshold:
        return indices[0].item()
    else:
        print("ТАКОЙ ЧЕЛОВЕК НЕ НАЙДЕН")
        return 0
        

def get_vector_from_face(img,model):
    """
    ВХОД: Изображения лица в формате тензора
    ВЫХОД: Вектор лица 
    """

    new_vector=model(img)
    return new_vector.numpy()

def open_numpy_as_tensor(numpy_img):
    """
    ВХОД: Изображение в формате numpy
    ВЫХОД: Изображение в формате тензора
    """
    numpy_img=cv2.resize(numpy_img,(112,112))
    numpy_img=numpy_img.astype('float32')/255.0
    numpy_img=numpy_img.transpose(2,0,1)
    img=torch.from_numpy(numpy_img)
    return img.unsqueeze(0)






