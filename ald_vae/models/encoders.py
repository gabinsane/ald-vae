import abc
import math
import numpy as np
import torch
from torch.nn import Conv2d, BatchNorm3d, Sequential, TransformerEncoderLayer, Embedding, ReLU, TransformerEncoder, \
    ModuleList, Module, Linear, SiLU
import torch.nn.functional as F
from numpy import prod
from models.NetworkTypes import NetworkTypes, NetworkRoles
from models.nn_modules import PositionalEncoding, ConvNet, SamePadConv3d, AttentionResidualBlock, expand_layer, downsize_layer
from utils import Constants


class VaeComponent(Module):
    def __init__(self, latent_dim: int, data_dim: tuple, latent_private=None, growtype=None, net_type=NetworkTypes.UNSPECIFIED, net_role=NetworkRoles.UNSPECIFIED):
        """
        Base for all

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: tuple
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        :param net_type: network type used as encoder
        :type net_type: NetworkTypes
        """
        super().__init__()
        self.net_role = net_role
        self.latent_dim = latent_dim
        self.latent_private = latent_private
        if self.latent_private is not None:
            self.out_dim = self.latent_dim + self.latent_private
        else:
            self.out_dim = self.latent_dim
        self.data_dim = data_dim
        self.net_type = net_type
        self.growtype = growtype
        self.mu_layer = None
        self.logvar_layer = None

    def update_latent_dim(self, new_dim, delete_neurons=[]):
        """
        Changes latent dimensionality (shape of the encoder output)
        :param new_dim:
        :type new_dim:
        :return:
        :rtype:
        """
        prev_dim = self.latent_dim
        self.latent_dim = new_dim
        if self.latent_private is not None:
            self.out_dim = self.latent_dim + self.latent_private
        else:
            self.out_dim = self.latent_dim
        if self.growtype == "neurons":
            if self.mu_layer is not None:
                if self.latent_dim > prev_dim:  # expand
                    self.mu_layer = expand_layer(self.mu_layer, Linear, (self.mu_layer.in_features, new_dim))
                    self.logvar_layer = expand_layer(self.logvar_layer, Linear, (self.logvar_layer.in_features, new_dim))
                else:  # downsize
                    self.mu_layer = downsize_layer(self.mu_layer, Linear, (self.mu_layer.in_features, new_dim), delete_neurons)
                    self.logvar_layer = downsize_layer(self.logvar_layer, Linear, (self.logvar_layer.in_features, new_dim), delete_neurons)
        elif self.growtype == "layers":
            if self.latent_dim > prev_dim:
                self.mu_logvar.append(Linear(self.mu_logvar[0].in_features, 2))
            else:
                self.mu_logvar = self.mu_logvar[:-1]

    def init_final_layers(self, in_feats):
        if self.growtype == "neurons":
            self.mu_layer = Linear(in_feats, self.latent_dim, bias=True)
            self.logvar_layer = Linear(in_feats, self.latent_dim, bias=True)
        elif self.growtype in "layers":
            self.mu_logvar = torch.nn.ModuleList()
            for i in range(self.latent_dim):
                self.mu_logvar.append(Linear(in_feats, 2))
        elif self.growtype == "layers_cnn":
            self.mu_logvar = torch.nn.ModuleList()
            assert len(self.net_list) == self.latent_dim, "the latent dim size must be equal to number of encoder layers"
            for i, net in enumerate(self.net_list):
                if hasattr(net, "out_features"):
                    self.mu_logvar.append(Linear(net.out_features, 2))
                else:
                    self.mu_logvar.append(Linear(np.product(self.o_shapes[i]), 2))

    def process_output(self, data, inter_outputs=None):
        if self.growtype in ["neurons", None, "none"]:
            out_mus = self.mu_layer(data)
            out_lvs = F.softmax(self.logvar_layer(data), dim=-1) + Constants.eta
        elif self.growtype == "layers":
            out_mus, out_lvs = [], []
            for idx, layer in enumerate(self.mu_logvar):
                o = layer(data)
                o1 = F.softmax(o[:,1], dim=-1) + Constants.eta
                out_mus.append(o[:,0])
                out_lvs.append(o1)
            out_mus = torch.stack(out_mus).transpose(1,0)
            out_lvs = torch.stack(out_lvs).transpose(1,0)
        elif self.growtype == "layers_cnn":
            out_mus, out_lvs = [], []
            for idx, layer in enumerate(self.mu_logvar):
                o = layer(inter_outputs[idx].reshape(inter_outputs[idx].shape[0], -1))
                o1 = F.softmax(o[:,1], dim=-1) + Constants.eta
                out_mus.append(o[:,0])
                out_lvs.append(o1)
            out_mus = torch.stack(out_mus).transpose(1,0)
            out_lvs = torch.stack(out_lvs).transpose(1,0)
        return out_mus, out_lvs

    @abc.abstractmethod
    def forward(self, x):
        """
            Forward pass

            :param x: data batch
            :type x: list, torch.tensor
            :return: tensor of means, tensor of log variances
            :rtype: tuple(torch.tensor, torch.tensor)
        """
        pass

class VaeEncoder(VaeComponent):
    def __init__(self, latent_dim, data_dim, latent_private, growtype, net_type: NetworkTypes):
        """
        Base for all encoders

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: tuple
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        :param net_type: network type used as encoder
        :type net_type: NetworkTypes
        """
        super().__init__(latent_dim, data_dim, latent_private, growtype, net_type, net_role=1)
        self.net_role = NetworkRoles.ENCODER


class Enc_CNN(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, growtype):
        """
        CNN encoder for RGB images of size 64x64x3

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim:
        :param latent_private: private latent space size (optional)
        :type latent_private: int
        """
        data_dim = (3, 64, 64)
        super(Enc_CNN, self).__init__(latent_dim, data_dim, latent_private, growtype, net_type=NetworkTypes.CNN)
        hid_channels = 32
        kernel_size = 4
        self.hidden_dim = 256
        self.silu = SiLU()
        self.o_shapes = [(32,32,32), (32,16,16),(32,8,8), (512), (256), (256)]
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = 3
        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        self.conv4 = Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.pooling = torch.nn.AvgPool2d(kernel_size)
        # Fully connected layers
        self.lin1 = Linear(np.product(self.reshape), self.hidden_dim)
        self.lin2 = Linear(self.hidden_dim, self.hidden_dim)
        self.net_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.lin1, self.lin2]

        # Fully connected layers for mean and variance
        self.mu_layer = Linear(self.hidden_dim, self.out_dim)
        self.logvar_layer = Linear(self.hidden_dim, self.out_dim)
        self.init_final_layers(self.hidden_dim)

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        if isinstance(x, dict):
            x = x["data"]
        batch_size = x.size(0) if len(x.shape) == 4 else x.size(1)
        # Convolutional layers with ReLu activations
        o1 = self.silu(self.conv1(x.float()))
        o2 = self.silu(self.conv2(o1))
        o3 = self.silu(self.conv3(o2))
        o4 = self.silu(self.conv4(o3))

        # Fully connected layers with ReLu activations
        o4 = o4.view((batch_size, -1))
        o5 = self.silu(self.lin1(o4))
        o6 = (self.lin2(o5))
        inter_outputs = [o1, o2, o3, o4, o5, o6]
        return self.process_output(o6, inter_outputs)


class Enc_MNIST(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, growtype):
        """
        Image encoder for the MNIST images

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Enc_MNIST, self).__init__(latent_dim, data_dim, latent_private, growtype, net_type=NetworkTypes.FNN)
        self.net_type = "CNN"
        self.hidden_dim = 400
        modules = [Sequential(Linear(784, self.hidden_dim), ReLU(True))]
        modules.extend([Sequential(Linear(self.hidden_dim, self.hidden_dim), ReLU(True))
                        for _ in range(1)])
        self.enc = Sequential(*modules)
        self.relu = ReLU()
        self.init_final_layers(self.hidden_dim)

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = x["data"]
        h = x.view(*x.size()[:-3], -1)
        h = self.enc(h.float())
        h = h.view(h.size(0), -1)
        return self.process_output(h)

def extra_hidden_layer(hidden_dim):
    return Sequential(Linear(hidden_dim, hidden_dim), ReLU(True))


class Enc_FNN(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, growtype):
        """
        Fully connected layer encoder for any type of data

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Enc_FNN, self).__init__(latent_dim, data_dim, latent_private, growtype, net_type=NetworkTypes.FNN)
        self.net_type = "FNN"
        self.hidden_dim = 128
        self.lin1 = torch.nn.DataParallel(Linear(np.prod(data_dim), self.hidden_dim))
        #self.lin2 = torch.nn.DataParallel(Linear(self.hidden_dim, self.hidden_dim))
        #self.lin3 = torch.nn.DataParallel(Linear(self.hidden_dim, self.hidden_dim))

        self.mu_layer = torch.nn.DataParallel(Linear(self.hidden_dim, self.out_dim))
        self.logvar_layer = torch.nn.DataParallel(Linear(self.hidden_dim, self.out_dim))
        self.init_final_layers(self.hidden_dim)

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = x["data"]
        x = (x).float()
        e = torch.relu(self.lin1(x.view(x.shape[0], -1)))
        return self.process_output(e)


class Enc_VideoGPT(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, growtype, n_res_layers=4, downsample=(2, 4, 4)):
        """
        Encoder for image sequences taken from https://github.com/wilson1yan/VideoGPT

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data (e.g. [10, 64, 64, 3] for 64x64x3 image sequences with max length 10 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        :param n_res_layers: number of ResNet layers
        :type n_res_layers: int
        """
        super(Enc_VideoGPT, self).__init__(latent_dim, data_dim, latent_private, growtype, net_type=NetworkTypes.DCNN)
        self.net_type = "3DCNN"
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.convs = ModuleList()
        max_ds = n_times_downsample.max()
        for i in range(max_ds):
            in_channels = 3 if i == 0 else self.out_dim
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(in_channels, self.out_dim, 4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
        self.conv_last = SamePadConv3d(in_channels, self.out_dim, kernel_size=3)
        self.res_stack = Sequential(
            *[AttentionResidualBlock(self.out_dim)
              for _ in range(n_res_layers)],
            BatchNorm3d(self.out_dim),
            ReLU())
        self.hidden_dim = self.out_dim * 16 * 16 * 4
        self.mu_layer = torch.nn.DataParallel(Linear(self.hidden_dim, self.out_dim))
        self.logvar_layer = torch.nn.DataParallel(Linear(self.hidden_dim, self.out_dim))
        self.init_final_layers(self.hidden_dim)

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = x["data"]
        h = x.permute(0, 4, 1, 2, 3)
        for conv in self.convs:
            h = F.relu(conv(h.float()))
        h = self.conv_last(h)
        h = self.res_stack(h)
        return self.process_output(h.reshape(x.shape[0], -1))

