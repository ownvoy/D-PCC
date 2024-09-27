import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layer import DownsampleLayer



class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.downsample_rate = args.downsample_rate

        encoder_layers = []
        for i in range(args.layer_num):
            encoder_layers.append(DownsampleLayer(args, i))
        self.encoder_layers = nn.ModuleList(encoder_layers)


    def forward(self, xyzs, feats):
        # input: (b, c, n)

        gt_xyzs = []
        gt_dnums = []
        gt_mdis = []
        downsample_cnt = 0
        remainders = []
        # print(f"downsampling {downsample_cnt}: {feats.shape}")
        for layer_idx, encoder_layer in enumerate(self.encoder_layers):
            # print(f"xyzs shape:{xyzs.shape[2]}")
            # print(xyzs.shape[2] * self.downsample_rate[layer_idx])
            gt_xyzs.append(xyzs)
            if xyzs.shape[2] * self.downsample_rate[layer_idx]>= 1:
                quotient = xyzs.shape[2] * self.downsample_rate[layer_idx]
                quotient = int(quotient)
                remainder = xyzs.shape[2]- int(quotient / self.downsample_rate[layer_idx])
                remainders.append(remainder)
                
                xyzs, feats, downsample_num, mean_distance = encoder_layer(xyzs, feats)
                # print(f"dnum element:{downsample_num}")
                # print(f"dnum shape:{downsample_num.shape}")
                gt_dnums.append(downsample_num)
                gt_mdis.append(mean_distance)
                downsample_cnt +=1
                # print(f"downsampling {downsample_cnt}: {feats.shape}")
                # print(f"remainder {remainders[layer_idx]}")
            else:
                # print(f"왜 여기로 downsampling {downsample_cnt}: {feats.shape}")
                # gt_dnums.append(torch.tensor([[xyzs.shape[2]]]).to(torch.device('cuda')))
                gt_mdis.append(torch.tensor([[0]]).to(torch.device('cuda')))
                
            
        latent_xyzs = xyzs
        latent_feats = feats
        remainders  = remainders[::-1]
        gt_dnums = gt_dnums[::-1]

        return gt_xyzs, gt_dnums, gt_mdis, latent_xyzs, latent_feats, downsample_cnt, remainders
