import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layer import DownsampleLayer



class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

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
        for layer_idx, encoder_layer in enumerate(self.encoder_layers):
            gt_xyzs.append(xyzs)
            if xyzs.shape[2] // args.downsample_rate[layer_idx]> 0:
                xyzs, feats, downsample_num, mean_distance = encoder_layer(xyzs, feats)
                gt_dnums.append(downsample_num)
                gt_mdis.append(mean_distance)
                downsample_ cnt +=1
            else:
                gt_dnums.append(xyzs.shape[2])
                gt_mdis.append(0)
                
                
                

        latent_xyzs = xyzs
        latent_feats = feats

        return gt_xyzs, gt_dnums, gt_mdis, latent_xyzs, latent_feats, downsample_cnt
