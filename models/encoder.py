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
        for layer_idx, encoder_layer in enumerate(self.encoder_layers):
            # print(f"xyzs shape:{xyzs.shape[2]}")
            print(xyzs.shape[2] * self.downsample_rate[layer_idx])
            gt_xyzs.append(xyzs)
            if xyzs.shape[2] * self.downsample_rate[layer_idx]>= 1:
                xyzs, feats, downsample_num, mean_distance = encoder_layer(xyzs, feats)
                print(f"downsample_num:{downsample_num.shape}")
                # print(f"mean_distance:{mean_distance.shape}")
                gt_dnums.append(downsample_num)
                gt_mdis.append(mean_distance)
                downsample_cnt +=1
            else:
                print(torch.tensor([[xyzs.shape[2]]]).shape)
                gt_dnums.append(torch.tensor([[xyzs.shape[2]]]).to(torch.device('cuda')))
                gt_mdis.append(torch.tensor([[0]]).to(torch.device('cuda')))
                
                
                

        latent_xyzs = xyzs
        latent_feats = feats

        return gt_xyzs, gt_dnums, gt_mdis, latent_xyzs, latent_feats, downsample_cnt
