import os
import numpy as np
from scipy import io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../")
from sin_conv.modules.sin_conv import SinConv

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

class FanFBP(nn.Module):
    def __init__(self, out_height=512, out_width=512):
        super().__init__()
        self._init_config()
        self._init_filter_param()
        self._init_param_to_device()
        self.out_height, self.out_width = out_height, out_width
        
        self.fbp = SinConv(out_height, out_width)
#         self._pre_calculate()
        
    def _init_config(self):
        self.D = 54.1
        self.N = 1024 #探测器个数
        self.M = 720  #角度数
        self.nT = np.arange(-51.15, 51.25, 0.1) * self.D / (self.D + 40.8)
        self.theta = np.deg2rad(np.arange(0.0, 360, 0.5))
        self.T = self.nT[1] - self.nT[0]
        self._load_extra_param()
        self.cosine_weight = self.D / np.sqrt(self.D**2 + self.nT**2) * (self.theta[1] - self.theta[0])
    
    def _load_extra_param(self):
        MAT_PATH = '/home/pengchengtao/Desktop/Sinogram_wsss/Task2'
        self.r = sio.loadmat(os.path.join(MAT_PATH, 'r.mat'))['r']
        self.phi = sio.loadmat(os.path.join(MAT_PATH, 'phi.mat'))['phi']
    
    def _init_filter_param(self):
        h_RL = np.zeros((2*self.N-1, 1))
        h_RL[0:2*self.N-1:2, 0] = -0.5/self.T**2/(np.arange(-self.N+1, self.N, 2))**2/np.pi**2
        h_RL[self.N-1] = 1/8/self.T**2
        h_RL *= self.T
        self.filters = h_RL.transpose(1, 0)[np.newaxis, np.newaxis, ...]
    
    def _init_param_to_device(self):
        self.filters = torch.from_numpy(self.filters).float().to(device)
        self.cosine_weight = torch.from_numpy(self.cosine_weight).float().to(device)
    
#     def _pre_calculate(self):
#         self.U_2 = np.zeros((self.M, self.out_height, self.out_width))
#         self.detector_pos = np.zeros((self.M, self.out_height, self.out_width))
#         for t in range(self.M):
#             beta = self.theta[t] - np.pi
#             th = np.pi/2 + beta + self.phi
#             s = self.D * self.r * np.sin(th) / (self.D + self.r * np.cos(th))
#             self.detector_pos[t, ...] = (((s - self.nT[0])/ self.T + 0.5) / float(self.N) - 0.5) * 2
#             self.U_2[t, ...] = (1 + self.r * np.sin(beta - self.phi) / self.D) ** 2
            
#     def fbp(self, x):
#         n, c, h, w = x.shape
        
#         outputs = torch.zeros(n, c, self.out_height, self.out_width).to(device)
#         _tmp_grid = np.zeros((n, self.out_height, self.out_width, 2)).astype(np.float32)

#         for t in range(self.M):

#             _tmp_grid[:, ..., 0] = self.detector_pos[t]
#             _tmp_grid[:, ..., 1] = (t / float(self.M) - 0.5) * 2
        
#             outputs = outputs + F.grid_sample(x, torch.from_numpy(_tmp_grid).to(device)) / torch.from_numpy(self.U_2[t]).float().to(device)
        
#         return outputs
    
    def forward(self, x):
                
        x = self.cosine_weight * x
        pj = F.conv2d(x, self.filters, padding=(0, self.N-1))
        
        out = self.fbp(pj)
        
        return out