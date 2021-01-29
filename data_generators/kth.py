import random
import os
import numpy as np
import socket
import torch
import imageio
import torchfile


class KTH(object):

    def __init__(self, train, data_root, seq_len = 20, image_size=64):
        self.data_root = '%s/processed' % data_root
        self.seq_len = seq_len
        self.image_size = image_size 
        self.classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

        self.dirs = os.listdir(self.data_root)
        if train:
            self.train = True
            data_type = 'train'
            self.persons = list(range(1, 21))
        else:
            self.train = False
            self.persons = list(range(21, 26))
            data_type = 'test'

        self.data= {}
        for c in self.classes:
            self.data[c] = torchfile.load('%s/%s/%s_meta%dx%d.t7' % (self.data_root, c, data_type, image_size, image_size))
        
        self.seed_set = False

    def get_sequence(self):
        t = self.seq_len
        files_name = bytes('files', "utf-8")
        while True: # skip seqeunces that are too short
            c_idx = np.random.randint(len(self.classes))
            c = self.classes[c_idx]
            vid_idx = np.random.randint(len(self.data[c]))
            vid = self.data[c][vid_idx]
            #for key, value in vid.items() :
            #    print(key)
            seq_idx = np.random.randint(len(vid[files_name]))
            if len(vid[files_name][seq_idx]) - t >= 0:
                break
        vid_name = bytes("vid", "utf-8")
        dname = '%s/%s/%s' % (self.data_root, c, vid[vid_name].decode())
        st = random.randint(0, len(vid[files_name][seq_idx])-t)
           

        seq = [] 
        for i in range(st, st+t):
            fname = '%s/%s' % (dname, vid[files_name][seq_idx][i].decode())
            im = imageio.imread(fname)/255.
            seq.append(im[:, :, 0].reshape(self.image_size, self.image_size, 1))
        return np.array(seq)

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
            #torch.manual_seed(index)
        return torch.from_numpy(self.get_sequence()).permute(0,3,1,2).to(dtype=torch.float)

    def __len__(self):
        return len(self.dirs)*36*5 # arbitrary
