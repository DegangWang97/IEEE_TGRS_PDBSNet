import torch
import numpy as np
import scipy.io as sio

def PDBSNetData(opt):
    
    # train dataloader
    data_dir = './data/'
    image_file = data_dir + opt.dataset + '.mat'
    
    input_data = sio.loadmat(image_file)
    image = input_data['data']
    image = image.astype(np.float32)

    image = ((image - image.min()) / (image.max() - image.min()))
    band = image.shape[2]

    train_data = np.expand_dims(image, axis=0)
    loader_train = torch.from_numpy(train_data.transpose(0,3,1,2)).type(torch.FloatTensor)
        
    print("The training dataloader construction process is done")
    print('-' * 50)
    return loader_train, band