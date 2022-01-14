import torch
import numpy as np
import cv2
import random


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


class zoom_image(object):
    
    def __init__(self,num_zoom_imgs =5,monitor = False):
        self.num_zoom_imgs = num_zoom_imgs
        self.monitor=monitor
        self.zoomed_imgs=[]
        self.x = [int(np.random.rand() * 224) for i in range(num_zoom_imgs)]
        self.y = [int(np.random.rand() * 224) for i in range(num_zoom_imgs)]
        self.dim_lst = list(zip(self.x,self.y))
        
    def __call__(self,img):
        if self.monitor == True:
            for dim in self.dim_lst:
                img_new = self.center_crop(img, dim)
                self.zoomed_imgs.append(img_new)
            return self.zoomed_imgs
        elif self.monitor == False:
            dim = random.choice(self.dim_lst)
            return self.center_crop(img, dim)
    
    def center_crop(self,img, dim):
        width, height = img.shape[1], img.shape[0]
        crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
        mid_x, mid_y = int(width/2), int(height/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        return crop_img
        

    def  __repr__(self):
            return self.__class__.__name__+'()'
    