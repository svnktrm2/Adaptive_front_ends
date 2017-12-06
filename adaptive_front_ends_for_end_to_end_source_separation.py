# Disclaimer
# ———————————————————
# University of Illinois
# Open Source License

# Copyright © <Year>, <Organization Name>. All rights reserved.

# Developed by:
# Shrikant Venkataramani, Paris Smaragdis
# University of Illinois at Urbana-Champaign, Adobe Research
# This work was supported by NSF grant 1453104.
# Paper: End-to-end source separation using adaptive front ends
# Paper link: https://arxiv.org/pdf/1705.02514.pdf

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution.
# Neither the names of <Name of Development Group, Name of Institution>, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
# ———————————————————



# Prerequisites
from __future__ import print_function

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import DataLoaderIter
import visdom

# General utilities
import time
import glob2
import librosa
from tqdm import trange
import matplotlib.pyplot as plt
from IPython.display import Audio
import pdb

import numpy.random as random
from numpy import std, vstack, hstack, argsort, argmax, array, hanning, real, imag, floor, eye, savez, dot, log10
from numpy import expand_dims
from numpy.fft import rfft, fft
from numpy.linalg import pinv
import numpy.linalg as linalg
import numpy as np

#Setting to my gpus#
import os
import pickle as pkl

eps = 1e-7

# BSS_eval metrics
def bss_eval( sep, i, sources):
    # Current target
    from numpy import dot, linalg, log10
    min_len = min([len(sep), len(sources[i])])
    sources = sources[:,:min_len]
    sep = sep[:min_len]
    target = sources[i]

    # Target contribution
    s_target = target * dot( target, sep.T) / dot( target, target.T)

    # Interference contribution
    pse = dot( dot( sources, sep.T), \
    linalg.inv( dot( sources, sources.T))).T.dot( sources)
    e_interf = pse - s_target

    # Artifact contribution
    e_artif= sep - pse;

    # Interference + artifacts contribution
    e_total = e_interf + e_artif;

    # Computation of the log energy ratios
    sdr = 10*log10( sum( s_target**2) / sum( e_total**2));
    sir = 10*log10( sum( s_target**2) / sum( e_interf**2));
    sar = 10*log10( sum( (s_target + e_interf)**2) / sum( e_artif**2));

    # Done!
    return (sdr, sir, sar)


# Loss function
def myloss( out, targ, par1 = None, par2 = None, mse=False):
    if mse:
        # MSE
        l = torch.mean( torch.pow( out - targ[ : , : out.size(1) ], 2.))
    else:
        # SDR
        l = -torch.mean( out * targ[ : , : out.size(1) ])**2 / (torch.mean( out**2) + eps)

    # Add some regularization
    if par1 is not None:
        l = l + 0.2 * torch.mean( torch.abs( par1))

    if par2 is not None:
        l = l + 0.2 * torch.mean( torch.abs( par2))

    return l


class aet( nn.Module):
    # Init
    def __init__( self, sz=1024, dnn_size=[512], hp=16,
        dropout=0.2, avg_len=5, ortho=False, transform='dft'):
        super( aet, self).__init__()

        # Initialization
        self.sz = sz
        self.hp = hp
        self.ortho = ortho
        self.transform = transform

        if transform == 'dft':
            self.trainable_frontend = False
            self.trainable_hidden = False

            # Hann window
            wn = hanning( ft_size+1)[:-1]
            f = fft( eye( sz))
            # Put cos terms on top, sin terms in bottom
            f = vstack( (real( f[:int(sz/2),:]),imag(f[:int(sz/2),:])))

        if transform == 'dct':
            self.trainable_frontend = False
            self.dctrainable_hidden = True
            # Hann window
            wn = hanning( sz+1)[:-1]
            # DCT matrix
            f = fftpack.dct(eye(sz))

        if transform == 'aet':
            self.trainable_frontend = True
            self.trainable_hidden = True
            # Hann window
            wn = hanning( sz+1)[:-1]
            # Random initialization
            f = 0.01*numpy.random.randn(sz,sz)

        # Define front-end conv layer variables
        # Initialize as window * f
        self.forward_layer = nn.Parameter( torch.FloatTensor( wn * f[:,None,:]), requires_grad=trainable_frontend)

        # Is it orthogonal?
        if self.ortho == True:
            self.inverse_layer = self.forward_layer
        else:
            self.inverse_layer = nn.Parameter( torch.FloatTensor( f[:,None,:]), requires_grad=trainable_frontend)

        # Combining the magnitudes for DCT
        identity = eye(sz)
        c1 = vstack([identity, -identity])
        identity = eye(sz // 2)
        c2_1 = hstack([identity, identity, identity, identity])
        flip_identity = flip(identity, 1)
        c2_2 = hstack([flip_identity, flip_identity, flip_identity, flip_identity])
        c2 = vstack([c2_1, c2_2])

        # if trainable_hidden:
        # For now, do only for DCT front-end
        if transform == 'dct':
             c1 = np.random.randn(2*ft_size, ft_size) * (1 / ft_size)
             c2 = np.random.randn(ft_size, 2*ft_size) * (2 / ft_size)
             self.c1 = torch.nn.Parameter( torch.FloatTensor(c1), requires_grad = trainable_hidden)
             self.c2 = torch.nn.Parameter( torch.FloatTensor(c2), requires_grad = trainable_hidden)


        # Smoothing variables (only for DCT and AET)
        # Pad with avg_len - 1 zeros at the end
        if trainable_hidden:
            self.smoothing_layer = nn.Conv1d( in_channels = sz,
                out_channels = sz,
                kernel_size = avg_len,
                groups = sz, # Separate smoothing per row
                padding = avg_len-1)

            # Batch normalization
            self.bn = nn.BatchNorm1d( sz)

        # Source Separation network 3 DNNs as Minje
        # Specify network in a loop
        n = [sz] + dnn_size + [sz]
        self.sep = nn.ModuleList([])
        for i in range( len( n)-1):
            ll = nn.Linear( n[i], n[i+1])
            self.sep.append( ll)

        # dropout
        self.dropout_layer = nn.Dropout(dropout)


    # Override forward function
    def forward( self, x):
        batchsize = x.size(0)
        T = x.size(1)

        # Reshape for conv1D to size batches X inchannels X T
        x_1 = x.view(batchsize, 1, T)

        # Front-end conv layer
        x_2 = F.conv1d( x_1, self.forward_layer, bias = None, stride = self.hp, padding = self.sz-1)

        # Magnitudes
        x_2 = torch.abs(x_2)

        # Combining for DCTs
        if self.transform == 'dct':
            # Permute for linear layers
            x_2_perm = x_2.permute(0, 2, 1).contiguous()
            # Rectify and combine to get magnitude spectra
            txr = F.softplus(F.linear(x_2_perm, self.c1))
            x_3 = F.linear(txr, self.c2)

        if self.transform == 'aet'
            x_2 = self.bn( x_2)


        # Smoothing layer
        # Ignore the last few frames
        x_3 = self.smoothing_layer( x_2)[:,:,:-int(self.avg_len/2)]
        # non-linearity
        # x_3 = F.relu(x_3)
        x_3 = F.softplus( x_3)

        # Magnitude and Phase
        mag = x_3
        phase = x_2/(x_3 + eps)

        # Reshape to (batchsize . time X dim)
        mag_r = mag.view( mag.size(0)*mag.size(1), mag.size(2))

        for l in self.sep:
            mag_separated = ( F.softplus(l( mag_r)))

        # Reshape for conv layer inverse
        x_4 = mag_separated.view(mag.size(0), mag.size(1), mag.size(2))

        # Multiply with mixture Phase
        x_5 = x_4 * phase
        x_5 = x_5.permute(0, 2, 1).contiguous()

        # Upsample and inversion simultaneously done by transposed convolution
        out = F.conv_transpose1d( x_5, self.inverse_layer, stride=self.hp, padding=self.sz-1)

        return out.view( batchsize, -1), x_2, mag, phase


    def main():
        net =  aet( sz=1024, dnn_size=[512], hp=16,
            dropout=0.2, avg_len=5, ortho=False, transform='dft')

        # Start GPUs
        net = torch.nn.DataParallel( net, device_ids=[0, 1])
        net = net.cuda()

        random.seed(25)
        print('Select data')
        M = Mix_Dataset('/usr/local/timit/timit-wav/train/dr1/f*',
                        '/usr/local/timit/timit-wav/train/dr1/m*',
                       val=8)
        MI = DataLoaderIter( DataLoader( M, batch_size=args.batchsize, num_workers=0, pin_memory=True))

        # Setup optimizer
        opt = torch.optim.RMSprop( filter( lambda p: p.requires_grad, net.parameters()), lr=0.01)

        # Get validation data
        xv,yv = M.getvals()

        # Initialize these for training
        e = []
        try:
            it = 0
            while it <= iterations:

                # Get data and move to GPU
                x,y = next(MI)
                inps = Variable( x).type_as( next( net.parameters()))
                target = Variable( y).type_as( next( net.parameters()))

                # Get loss
                net.train()
                out,h1,mag,phase = net( inps )
                loss = myloss( z, target, par1=h1, par2=mag, mse=False)

                # Update
                opt.zero_grad()
                loss.backward()
                opt.step()

                # Report
                e.append( abs( loss.data[0]) )
                it += batchsize
                be = [list( evaluate( net, xv, yv))]

                if it%10 == 0:
                    print(it, abs( loss.data[0]), be)
            net.eval()
            except KeyboardInterrupt:
        net.eval()
        pass

    return net
