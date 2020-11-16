
# Test bair_push, so an example of how to run it on the GPU

import bair_push
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


dataset =bair_push.PushDataset(split='train',dataset_dir='/work1/s144077/bair_robot_data/processed_data/',seq_len=10)

batch_size=16
train_loader=DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last = True)

frames,index=next(iter(train_loader))

print('% shape data' % str(frames.shape))
