import torch
from torch import nn
import numpy as np

class Abs_conv(torch.nn.Module):
    def __init__(self, N_features=1, d_out=1, dtype=torch.float, device="cpu" ):

        super(Abs_conv, self).__init__()
        self.dtype = dtype
        self.device = device
        self.dim = N_features
        self.d_out = d_out
        self.centers = torch.nn.parameter.Parameter(  torch.rand( self.d_out, self.dim,
                                                        device=torch.device(self.device),
                                                        dtype=self.dtype) ) #, requires_grad=True

    def forward(self, x):
        n_batch = x.size(0)
        x = x.view(-1, self.dim)
        result = torch.zeros( self.d_out, x.size(0) ).to(self.device)

        for i, c in enumerate( self.centers ):
            result[i] = (x-c).abs().sum(axis=1)
        result.transpose_(1,0)
        return result.view(n_batch, -1, self.d_out)



class Seg_Asb_net(torch.nn.Module):
    def __init__(self, n_points=1024, do=.25, r_ch=64, device="cpu"):
        super(Seg_Asb_net, self).__init__()
        self.n_points = n_points
        self.do = do
        self.device = device
        dim = 1
        self.rad_c = int(r_ch)

        self.rc1 = Abs_conv( 3, self.rad_c, None, self.device )
        self.bnrc1 = nn.BatchNorm1d( self.n_points )

        self.mlp1 = nn.Linear( self.rad_c, 128 )
        self.mlp2 = nn.Linear( 128, 512) #1024 )
        self.bnl1 = nn.BatchNorm1d( 128 )
        self.bnl2 = nn.BatchNorm1d( 512 )#1024 )
        self.maxpool = nn.MaxPool2d( (self.n_points, 1) )
        
        self.mlp3 = nn.Linear( 512+self.rad_c, 512 )
        self.mlp4 = nn.Linear( 512, 128 )#256 )
        self.bnl3 = nn.BatchNorm1d( 512 )
        self.bnl4 = nn.BatchNorm1d( 128 )#256 )

        self.mlpf = nn.Linear( 128, 2 ) # ( 256, 2 )
        self.bnlf = nn.BatchNorm1d( 2 )

        
    def concat( self, x, g_feature ):
        g_feature = g_feature.repeat( 1, 1, self.n_points, 1)
        result = torch.cat([x, g_feature.view(x.size(0), self.n_points, -1) ], dim=2)
        return result

    def forward(self, entrada ):
        n_batch = entrada.size()[0]

        code = entrada.view( n_batch, 1, self.n_points, -1)
        r1 = self.bnrc1( self.rc1( code ) ).view( n_batch,-1, self.rad_c )

        code = r1.view( -1, self.rad_c )

        code = nn.ReLU(inplace=True)( self.bnl1( self.mlp1( code )) )
        code = nn.ReLU(inplace=True)( self.bnl2( self.mlp2( code )) )
        code = nn.Dropout(p=self.do)( code )
        code = code.view( n_batch, 1, self.n_points, -1 )
        code = self.maxpool( code )


        code = self.concat( r1, code )

        code = nn.ReLU(inplace=True)(self.bnl3( self.mlp3( code.view( -1, self.rad_c+512 ) ) ))
        code = nn.ReLU(inplace=True)(self.bnl4( self.mlp4( code ) ))
        code = nn.Dropout(p=self.do)( code )

        code = nn.ReLU(inplace=True)(self.bnlf( self.mlpf( code ) ))
        code = code.view( n_batch, self.n_points, -1)
        return nn.Softmax(dim=2)( code )