import socket
import numpy as np
from torchvision import datasets, transforms

class MovingMNIST(object):
    
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=64,digit_size=32, deterministic=True, three_channels = True,step_length=1):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits  
        self.image_size = image_size 
        self.step_length = step_length
        self.digit_size = digit_size
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1 
        self.three_channels = three_channels
        
        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data) 

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size, 
                      image_size, 
                      self.channels),
                    dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]
            digit=digit.numpy()
            #digit = resize(digit, (1,self.digit_size, self.digit_size))
            ds=digit.shape[1]
            sx = np.random.randint(image_size-ds)
            sy = np.random.randint(image_size-ds)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0 
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size-ds:
                    sy = image_size-ds-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)
                    
                if sx < 0:
                    sx = 0 
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size-ds:
                    sx = image_size-ds-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)
                   
                x[t, sy:sy+ds, sx:sx+ds, 0] += digit.squeeze()
                sy += dy*self.step_length
                sx += dx*self.step_length

        x[x>1] = 1. # When the digits are overlapping.
        n_channels = 1
        x=x.reshape(self.seq_len, n_channels, self.image_size, self.image_size)
        if self.three_channels:
            x=np.repeat(x, 3, axis=1)
        return x # This should be changed to something else if seqence length of more than 1
