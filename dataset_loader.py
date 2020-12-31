import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random



def get_dataset(transformation, img_path, outline_path=None, seeds_amount=100, show_examples_amount=0):
    Images = []
    Segmented = []
    
    example_shown = 0
    for filename in os.listdir(img_path):
        img=plt.imread(img_path+filename) 
        
        if outline_path:
            outline_filename = filename[0:6] + "cells.png"
            outline = plt.imread(outline_path+outline_filename)
            outline = outline[:,:,None]
        else:
            outline = img[:,:,1]
        
        seeds = np.random.randint(0, seeds_amount * 15, seeds_amount)
        for seed in seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            img_trnsfrmd=transformation(img).numpy()
            img_trnsfrmd=img_trnsfrmd.transpose(1,2,0)

            random.seed(seed)
            torch.manual_seed(seed)
            outline_trnsfrmd=transformation(outline).numpy()
            outline_trnsfrmd=outline_trnsfrmd.squeeze()

            Images.append(img_trnsfrmd)
            Segmented.append(outline_trnsfrmd)

            if (example_shown < show_examples_amount) and (np.random.binomial(1, 0.0007) == 1):
                ### show some example
                f, ax = plt.subplots(2,2, figsize=(6,6))
                
                ax[0][0].set_title('original image')
                ax[0][0].imshow(img)
                
                ax[0][1].set_title('transformed original image')
                ax[0][1].imshow(img_trnsfrmd)
                
                ax[1][0].set_title('outline image')
                ax[1][0].imshow(outline, cmap='gray')
                
                ax[1][1].set_title('transformed outline image')
                ax[1][1].imshow(outline_trnsfrmd, cmap='gray')
                plt.show()
                example_shown +=1
    
    Images_npy_array = np.array(Images).transpose(0,3,1,2)  # transpose for the torch image format: C,W,H
    Images_tensor = torch.from_numpy(Images_npy_array).type(torch.DoubleTensor)
    
    Segmented_npy_array = np.array(Segmented)[:,None]
    Segmented_tensor = torch.from_numpy(Segmented_npy_array).type(torch.DoubleTensor)
    
    Dataset = torch.cat((Images_tensor, Segmented_tensor), dim=1)
    return Dataset