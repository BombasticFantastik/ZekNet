import torch
from torch.nn import Module
import faiss
import onnxruntime as ort
import yaml
from torchvision import transforms
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

def open_img_as_tensor(img_path):
    """
    ВХОД: Путь до изображения
    ВЫХОД: Изображение в формате тензора
    """

    trans=transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor()
    ])

    img=trans(PIL.Image.open(img_path))
    return img.unsqueeze(0)

def open_numpy_as_tensor(numpy_img):
    """
    ВХОД: Изображение в формате numpy
    ВЫХОД: Изображение в формате тензора
    """
    numpy_img=torch.from_numpy(numpy_img)
    trans=transforms.Compose([
        transforms.Resize((112,112)),
        #transforms.ToTensor()
    ])

    img=trans(numpy_img)
    return img.unsqueeze(0)


if __name__ == "__main__":

    #тест 1 - запуск модели
    embedder=BuffaloModel('w600k_r50.onnx')

    #тест 2 - получение вектора из фото
    test_img=open_img_as_tensor(img_path=files_option['demonstration_img'])
    vector=get_vector_from_face(img=test_img,model=embedder)
    print(vector.shape)

    #тест 3 - сравнение нового фото со всей базой 
    embendings=None# заменить когда будет готово получение вектора из базы
    if embendings!=None:
        compare_new_face(img=test_img,vectors=embendings,model=embedder)
    else:
        pass

    print('УСПЕШНО!')



