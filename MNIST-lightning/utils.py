import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
import random


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


# def reduce_img(x, size, pos):
#     original_size = 28
#     if (pos[0] + size)>original_size:
#       pos[0] = pos[0] - (pos[0]+size-original_size)
#     if (pos[1] + size)>original_size:
#       pos[1] = pos[1] - (pos[1]+size-original_size)
#     canvas = np.zeros((original_size, original_size), dtype=np.float32)
#     small_img = cv2.resize(np.array(x).reshape(original_size,original_size, 1), (size,size))
#     canvas[ pos[1]:pos[1]+small_img.shape[0], pos[0]:pos[0]+small_img.shape[1]] = small_img
#     return canvas

def reduce_img(x, size, pos):
    original_size = 28
    if (pos[0] + size)>original_size:
      pos[0] = pos[0] - (pos[0]+size-original_size)
    if (pos[1] + size)>original_size:
      pos[1] = pos[1] - (pos[1]+size-original_size)
    canvas = np.zeros((original_size, original_size), dtype=np.float32)
    small_img = cv2.resize(x.numpy().reshape(original_size,original_size, 1), (size,size))
    canvas[ pos[1]:pos[1]+small_img.shape[0], pos[0]:pos[0]+small_img.shape[1]] = small_img
    return canvas

# class zoom_image(object):
#     def __init__(self,zoom_images=5,monitor = False):
#         self.zoom_images = zoom_images
#         self.monitor = monitor
#         self.ss_list = []
    
#     def __call__(self,img):
#         rimgs = np.zeros((28, 28 * self.zoom_images))
#         img = np.asarray(img)
#         img_dtype = img.dtype
#         zoomed_imgs = []
#         image_sizes = (np.random.dirichlet([25,10,10,10,10])*28).astype(int)
#         ss = 0
#         x = int(np.random.rand() * 28)
#         y = int(np.random.rand() * 28)
#         rand_img = np.random.randint(0,self.zoom_images,1)
#         for i, image_size in enumerate(image_sizes):
#             ss +=image_size
#             nimg = reduce_img(img, ss, [x,y])
#             nimg = np.clip(a=nimg, a_min=0, a_max=1)
#             rimgs[:, i*28:(i+1)*28] = nimg
#             img_variable = nimg.reshape((28,28))
#             zoomed_imgs.append(img_variable)
#             self.ss_list.append(ss)
#         zoomed_imgs = np.array(zoomed_imgs)
#         if self.monitor == True:
#             return zoomed_imgs,self.ss_list,rimgs
#         elif self.monitor == False:
#             return zoomed_imgs[rand_img].reshape((28,28))
    
#     def  __repr__(self):
#         return self.__class__.__name__+'()'

class zoom_image(object):
    def __init__(self,zoom_images=5,monitor = False):
        self.zoom_images = zoom_images
        self.monitor = monitor
        self.x = int(np.random.rand() * 28)
        self.y = int(np.random.rand() * 28)
        self.image_sizes = (np.random.dirichlet([25,10,10,10,10])*28).astype(int)
        self.img_size = random.choice(self.image_sizes)
        self.reduced_imgs = []
        self.ss =0
        
    def __call__(self,img):
        if self.monitor == True:
            for i,image_size in enumerate(self.image_sizes):
                self.ss += image_size 
                new_img = reduce_img(img, self.ss, [self.x,self.y])
                self.reduced_imgs.append(new_img)
            return self.reduced_imgs
        elif self.monitor == False:
            new_img = reduce_img(img, self.img_size, [self.x,self.y])
            new_img = np.clip(a=new_img, a_min=0, a_max=1)
            return new_img
    
    def  __repr__(self):
        return self.__class__.__name__+'()'
    

if __name__ == "__main__":
    zoom_img = zoom_image(zoom_images=5,monitor = True)
    img = np.random.rand(28,28)
    zoomed  = zoom_img(img)
    print(zoomed[0])
        
    
