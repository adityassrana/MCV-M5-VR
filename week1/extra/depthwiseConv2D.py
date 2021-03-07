from torch import nn

class depthConv2D(nn.Module):
    def __init__(self):
        super(depthConv2D, self).__init__()
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 0, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))
        
        self.conv = conv_dw(3, 32, 1)
    
    def forward(self,x):
        return self.conv1(x)