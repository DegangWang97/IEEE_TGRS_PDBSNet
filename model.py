import torch
import torch.nn as nn

class PDBSNet(nn.Module):
    def __init__(self, nch_in=189, nch_out=189, nch_ker=64, nblk=9):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(nch_in, nch_ker, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)

        self.branch1 = DC_branchl(2, nch_ker, nblk)
        self.branch2 = DC_branchl(3, nch_ker, nblk)

        ly = []
        ly += [ nn.Conv2d(nch_ker*2,  nch_ker,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(nch_ker,    nch_ker//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(nch_ker//2, nch_out,    kernel_size=1) ]
        self.tail = nn.Sequential(*ly)

    def forward(self, x):
        x = self.head(x)

        br1 = self.branch1(x)
        br2 = self.branch2(x)

        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)

class DC_branchl(nn.Module):
    def __init__(self, stride, nch_in, nblk):
        super().__init__()

        ly = []
        ly += [ CentralMaskedConv2d(nch_in, nch_in, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(nch_in, nch_in, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(nch_in, nch_in, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ DCl(stride, nch_in) for _ in range(nblk) ]

        ly += [ nn.Conv2d(nch_in, nch_in, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)

class DCl(nn.Module):
    def __init__(self, stride, nch_in):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(nch_in, nch_in, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(nch_in, nch_in, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)

class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH//2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
