import math
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod

from models.NetworkTypes import NetworkTypes, NetworkRoles
from models.encoders import VaeComponent
from models.nn_modules import DeconvNet, expand_layer, downsize_layer
from models.nn_modules import PositionalEncoding, AttentionResidualBlock, \
    SamePadConvTranspose3d
from utils import Constants


class VaeDecoder(VaeComponent):
    def __init__(self, latent_dim, data_dim, latent_private, growtype, net_type: NetworkTypes):
        """
        Base for all decoders

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: tuple
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        :net_type: network type used as encoder
        :type net_type: NetworkTypes
        """
        super().__init__(latent_dim, data_dim, latent_private, growtype, net_type, NetworkRoles.DECODER)
        self.first = None

    def update_latent_dim(self, new_dim, delete_neurons=[]):
        """
        Changes latent dimensionality (shape of the encoder output)
        :param new_dim:
        :type new_dim:
        :return:
        :rtype:
        """
        old_dim = self.latent_dim
        self.latent_dim = new_dim
        if self.latent_private is not None:
            self.out_dim = self.latent_dim + self.latent_private
        else:
            self.out_dim = self.latent_dim
        if self.first is not None:
            if old_dim < self.latent_dim:
                self.first = expand_layer(self.first, nn.Linear, (new_dim, self.first.out_features))
            else:
                self.first = downsize_layer(self.first, nn.Linear, (new_dim, self.first.out_features), delete_neurons)


class Dec_CNN(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private, growtype):
        """
        CNN decoder for RGB images of size 64x64x3

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: private latent space size (optional)
        :type latent_private: int
        """
        super(Dec_CNN, self).__init__(latent_dim, data_dim, latent_private, growtype, net_type=NetworkTypes.CNN)

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        self.n_chan = 3

        # Fully connected lay
        self.first = nn.Linear(self.out_dim, hidden_dim)
        self.lin2 = torch.nn.DataParallel(nn.Linear(hidden_dim, hidden_dim))
        self.lin3 = torch.nn.DataParallel(nn.Linear(hidden_dim, np.product(self.reshape)))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        self.convT_64 = torch.nn.DataParallel(nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs))

        self.convT1 = torch.nn.DataParallel(nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs))
        self.convT2 = torch.nn.DataParallel(nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs))
        self.convT3 = torch.nn.DataParallel(nn.ConvTranspose2d(hid_channels, self.n_chan, kernel_size, **cnn_kwargs))

    def forward(self, z):
        """
        Forward pass

        :param z: sampled latent vectors z
        :type z: torch.tensor
        :return: output reconstructions, log variance
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        z = z["latents"]
        if len(z.shape) == 2:
            batch_size = z.size(0)
        else:
            batch_size = z.size(1)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.first(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size * x.shape[0], *self.reshape)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        x = (self.convT3(x))
        d = torch.sigmoid(x.view(*z.size()[:-1], *self.data_dim))  # reshape data
        d = d.clamp(Constants.eta, 1 - Constants.eta)
        return d.squeeze().reshape(-1, *self.data_dim), torch.tensor(0.75).to(z.device)

def extra_hidden_layer(hidden_dim):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))

class Dec_MNIST(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private, growtype):
        """
        Image decoder for the MNIST images

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Dec_MNIST, self).__init__(latent_dim, data_dim, latent_private, growtype, net_type=NetworkTypes.FNN)
        self.data_dim = data_dim
        self.net_type = "CNN"
        self.hidden_dim = 400
        self.first = nn.Linear(self.out_dim, self.hidden_dim)
        modules = []
        modules.append(nn.ReLU(True))
        modules.extend(
            [nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True)) for _ in range(2 - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(self.hidden_dim, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        """
        Forward pass

        :param z: sampled latent vectors z
        :type z: torch.tensor
        :return: output reconstructions, log variance
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        z = z["latents"]
        z = self.first(z)
        x_hat = self.dec(z)
        x_hat = self.fc3(x_hat)
        x_hat = self.sigmoid(x_hat)
        d = x_hat.view(*z.size()[:-1], *self.data_dim).squeeze(0)
        d = d.permute(0, 3, 1, 2) if len(d.shape) == 4 else d.permute(0, 1, 4, 2, 3)
        return d.squeeze(0), torch.tensor(0.75).to(z.device)


class Dec_FNN(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private, growtype):
        """
        Fully connected layer decoder for any type of data

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Dec_FNN, self).__init__(latent_dim, data_dim, latent_private, growtype, net_type=NetworkTypes.FNN)
        self.net_type = "FNN"
        self.hidden_dim = 128
        self.data_dim = data_dim
        self.first = torch.nn.DataParallel(nn.Linear(self.out_dim, self.hidden_dim))
        #self.lin2 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, self.hidden_dim))
        #self.lin3 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc3 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, np.prod(data_dim)))

    def forward(self, z):
        """
        Forward pass

        :param z: sampled latent vectors z
        :type z: torch.tensor
        :return: output reconstructions, log variance
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        z = z["latents"]
        p = torch.relu(self.first(z))
        #p = torch.relu(self.lin2(p))
        #p = torch.relu(self.lin3(p))
        d = (self.fc3(p))  # reshape data
        d = d.reshape(-1, *self.data_dim)
        return d, torch.tensor(0.75).to(z.device)  # mean, length scale

class Dec_VideoGPT(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private, growtype, n_res_layers=4):
        """
        Decoder for image sequences taken from https://github.com/wilson1yan/VideoGPT

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data (e.g. [10, 64, 64, 3] for 64x64x3 image sequences with max length 10 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        :param n_res_layers: number of ResNet layers
        :type n_res_layers: int
        """
        super(Dec_VideoGPT, self).__init__(latent_dim, data_dim, latent_private, growtype, net_type=NetworkTypes.DCNN)
        self.net_type = "3DCNN"
        self.upsample = (1, 4, 4)
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(self.out_dim)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(self.out_dim),
            nn.ReLU())
        n_times_upsample = np.array([int(math.log2(d)) for d in self.upsample])
        max_us = n_times_upsample.max()
        self.first = nn.Linear(self.out_dim, latent_dim)
        self.convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = 3 if i == max_us - 1 else self.out_dim
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            convt = SamePadConvTranspose3d(self.out_dim, out_channels, 4, stride=us)
            self.convts.append(convt)
            n_times_upsample -= 1
        self.upsample = torch.nn.DataParallel(nn.Linear(self.out_dim, self.out_dim * 16 * 16 * self.data_dim[0]))

    def forward(self, z):
        """
        Forward pass

        :param x: sampled latent vectors z
        :type x: torch.tensor
        :return: output reconstructions, log variance
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = z["latents"]
        x = self.first(x)
        x_upsampled = self.upsample(x)
        h = self.res_stack(x_upsampled.view(-1, x.shape[2], self.data_dim[0], 16, 16))
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts) - 1:
                h = F.relu(h)
        h = h.permute(0, 2, 3, 4, 1)
        h = torch.sigmoid(h)
        return h, torch.tensor(0.75).to(x.device)
