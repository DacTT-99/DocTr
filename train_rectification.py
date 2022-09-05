import os
import time
import warnings

import torch
from torch.optim import lr_scheduler
from torch.utils import data

from data_loader import Doc3D_Rectify
from GeoTr import GeoTr
from loss import Rectification_loss

warnings.filterwarnings('ignore')

def train(train_img_path, ckpt_path, batch_size, lr, num_workers, epoch_iter, interval):
    dataset = Doc3D_Rectify(dataset_path=train_img_path)
    file_num = len(dataset)
    train_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    Rectify_model = GeoTr(num_attn_layers=6)
    criterion = Rectification_loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Rectify_model.to(device)
    optimizer = torch.optim.Adam(Rectify_model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)

    for epoch in range(epoch_iter):
        Rectify_model.train()
        scheduler.step()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, bm ) in enumerate(train_loader):
            start_time = time.time()
            img, bm = img.to(device), bm.to(device)
            output = Rectify_model(img)
            loss = criterion(output, bm)
            
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
            epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
        
        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
        print(time.asctime(time.localtime(time.time())))
        print('='*50)
        if (epoch + 1) % interval == 0:
            state_dict = Rectify_model.state_dict()
            torch.save(state_dict, os.path.join(ckpt_path, 'rectify_model_epoch_{}.pth'.format(epoch+1)))

if __name__ == '__main__':
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    train_img_path = '/home/list_99/data/doc3D'
    ckpt_path      = './rectify_ckpt'
    batch_size     = 16
    lr             = 1e-3
    num_workers    = 4
    epoch_iter     = 600
    save_interval  = 5
    train(train_img_path, ckpt_path, batch_size, lr, num_workers, epoch_iter, save_interval)	
