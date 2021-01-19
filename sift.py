import torch
import matplotlib.pyplot as plt
import numpy as np


def sift(model, train_dataset, test_dataset, cuda=False):
    i_train = np.random.randint(0, train_dataset.shape[0], 1)[0]
    i_test = np.random.randint(0, test_dataset.shape[0], 1)[0]
    
    img_train = train_dataset[i_train, :-1].double()
    seg_train = train_dataset[i_train, -1].double()
    
    img_test = test_dataset[i_test, :-1].double()
    seg_test = test_dataset[i_test, -1].double()
    
    imgs = torch.stack((img_train, img_test))
    if cuda:
        imgs = imgs.cuda()
    
    reco = model(imgs)
    
    #print(reco.shape)
    
    f, ax = plt.subplots(2,3, figsize=(15,7))
    ax[0][0].set_title('original train')
    ax[0][0].imshow(img_train.cpu().numpy().transpose(1,2,0).clip(0,1))
    
    ax[0][1].set_title('outline train ground truth')
    ax[0][1].imshow(seg_train.cpu().numpy().clip(0,1), 'gray')
    
    ax[0][2].set_title('segmented train NN predicted')
    ax[0][2].imshow(reco[0][0].cpu().detach().numpy().clip(0,1), 'gray')
    
    ax[1][0].set_title('original test')
    ax[1][0].imshow(img_test.cpu().numpy().transpose(1,2,0).clip(0,1))
    
    ax[1][1].set_title('outline test ground truth')
    ax[1][1].imshow(seg_test.cpu().numpy().clip(0,1), 'gray')
    
    ax[1][2].set_title('segmented test NN predicted')
    ax[1][2].imshow(reco[1][0].cpu().detach().numpy().clip(0,1), 'gray')
    
    plt.show()