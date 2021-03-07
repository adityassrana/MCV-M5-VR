from torch import nn
import torch.nn.functional as F

def noop (x=None, *args, **kwargs):
    "Do nothing"
    return x

def conv_layer(ni, nf, ks=3, stride=2, padding=1 **kwargs):
    """
    Block function to play around with convolutions, activations and batchnorms.
    Each part can be changed easily here and then used in the main function get_model()
    Downsampling is done using stride 2

    Args:
        ni: number of input channels to conv layer
        nf: number of output channels to conv layer

    Returns:
        A conv block

    """
    _conv = nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding,bias=False,**kwargs)
    nn.init.kaiming_normal_(_conv.weight)
    return nn.Sequential(_conv, nn.BatchNorm2d(nf), nn.ReLU())

def block(ni,nf,stride):
    return nn.Sequential(
        conv_layer(ni, nf, stride=stride),
        conv_layer(nf, nf, 1))

class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=2):
        super().__init__()
        self.convs = block(ni,nf,stride)
        self.idconv = noop if ni==nf else nn.Conv2d(ni, nf, 1)
        self.pool = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))