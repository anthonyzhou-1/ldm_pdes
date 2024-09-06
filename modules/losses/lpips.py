"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""
import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple
from modules.losses.dpot import DPOTNet
from einops import repeat, rearrange
import torch.nn.functional as F 

class LPIPS_DPOT(nn.Module):
    # Learned perceptual metric
    def __init__(self, model_size='Ti', model_path=None, use_dropout=True):
        super().__init__()
        self.step_size = 10
        print("using LPIPS-DPOT")
        if model_size == 'Ti':
            self.net = self.get_tiny()
            self.n_features = 4
            self.dim = 512
        elif model_size == 'S':
            self.net = self.get_small()
            self.n_features = 6
            self.dim = 1024
        elif model_size == 'M':
            self.net = self.get_medium()
            self.n_features = 12
            self.dim = 1024
        else:
            raise ValueError(f"Invalid model size: {model_size}")
        
        self.net.load_state_dict(torch.load(model_path)['model'])

        print(f"Model size: DPOT-{model_size}, loaded from {model_path}")

        self.linear_layers = nn.ModuleList()

        for i in range(self.n_features):
            self.linear_layers.append(NetLinLayer(self.dim, use_dropout=use_dropout))

        for param in self.net.parameters():
            param.requires_grad = False # freeze backbone



    def preprocess(self, x):  
        # x in shape [b, c, nt, nx, ny]
        # one consideration is that the channel dimension is a different shape in DPOT. 
        # Our dataloader puts (vx, vy, u) in the channel dimension, but DPOT expects (u, vx, vy)

        # rearrange channel dimension
        nc = x.shape[1]
        u = x[:, 2:3, :, :, :]
        v = x[:, 0:2, :, :, :]
        mask = x[:, :-1, :, :, :]
        x = torch.cat([u, v, mask], dim=1)

        assert x.shape[1] == nc # check if the channel dimension is the same

        if x.shape[-1] != 128:
            c = x.shape[1]
            x = rearrange(x, 'b c t x y -> b (c t) x y')
            x = F.interpolate(x, size=128, mode='bilinear', align_corners=False)
            x = rearrange(x, 'b (c t) x y -> b c t x y', c=c)

        x = rearrange(x, 'b c t x y -> b x y t c')
        if x.shape[1] != x.shape[2]:
            x = repeat(x, 'b x y t c -> b x (n y) t c', n=4)
        
        return x

    def forward(self, input, target):
        # input, target in shape [b, c, nt, nx, ny]
        nt = input.shape[2]
        b = input.shape[0]
        in0_input, in1_input = self.preprocess(input), self.preprocess(target)

        loss = torch.zeros(b, 1, nt, 1, 1, device=input.device)
        for i in range(nt-self.step_size):
            start = i
            end = i + self.step_size

            in0_input_step, in1_input_step = in0_input[:, :, :, start:end], in1_input[:, :, :, start:end]

            outs0, outs1 = self.net(in0_input_step, features=True), self.net(in1_input_step, features=True)
            feats0, feats1, diffs = {}, {}, {}

            for kk in range(self.n_features):
                feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
                diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

            res = [spatial_average(self.linear_layers[kk].model(diffs[kk]), keepdim=True) for kk in range(self.n_features)]
            val = res[0]
            for l in range(1, self.n_features):
                val += res[l]
            
            loss[:, :, start] = val # val in shape [b, 1, 1, 1]
        
        loss[:, :, start+1:] = repeat(loss[:, :, start], 'b 1 1 1 -> b 1 nt 1 1', nt = self.step_size) # fill in the rest of the steps with the last step
        return loss 
    

    def get_tiny(self):
        model = DPOTNet(img_size=128, 
                patch_size=8, 
                mixing_type='afno', 
                in_channels=4, 
                in_timesteps=10, 
                out_timesteps=1, 
                out_channels=4, 
                normalize=False, 
                embed_dim=512,
                modes=32, 
                depth=4, 
                n_blocks=4, 
                mlp_ratio=1, 
                out_layer_dim=32, 
                n_cls=12)
        return model
    
    def get_small(self):
        model = DPOTNet(img_size=128, 
                patch_size=8, 
                mixing_type='afno', 
                in_channels=4, 
                in_timesteps=10, 
                out_timesteps=1, 
                out_channels=4, 
                normalize=False, 
                embed_dim=1024,
                modes=32, 
                depth=6, 
                n_blocks=8, 
                mlp_ratio=1, 
                out_layer_dim=32, 
                n_cls=12)
        return model
    
    def get_medium(self):
        model = DPOTNet(img_size=128, 
                patch_size=8, 
                mixing_type='afno', 
                in_channels=4, 
                in_timesteps=10, 
                out_timesteps=1, 
                out_channels=4, 
                normalize=False, 
                embed_dim=1024,
                modes=32, 
                depth=12, 
                n_blocks=8, 
                mlp_ratio=4, 
                out_layer_dim=32, 
                n_cls=12)
        return model

class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)
