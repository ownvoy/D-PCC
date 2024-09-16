import numpy as np
import os
import torch
from einops import rearrange
import open3d as o3d
import random
import argparse
import struct

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count


def index_points(xyzs, idx):
    """
    Input:
        xyzs: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = xyzs.shape[1]

    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)

    # (b, c, (s k))
    res = torch.gather(xyzs, 2, idx[:, None].repeat(1, fdim, 1))

    if reshape:
        res = rearrange(res, "b c (s k) -> b c s k", s=sample_num)

    return res


def save_pcd_pred(
    dir, name, pred_xyzs, pred_feats, gt_xyzs, gt_feats, all_pred2gt_idx, args
):
    path = os.path.join(dir, name)
    num_points = pred_xyzs.shape[1]

    # Update pred_feats based on conditions and all_pred2gt_idx
    if not args.compress_normal:
        # Override normals with closest gt features
        for i in range(len(all_pred2gt_idx)):
            pred_feats[i, 0:3, :] = gt_feats[i, 0:3, all_pred2gt_idx[i]]
    if not args.compress_opacitiy:
        # Override opacity with closest gt features
        for i in range(len(all_pred2gt_idx)):
            pred_feats[i, 3, :] = gt_feats[i, 3, all_pred2gt_idx[i]]
    if not args.compress_scales:
        # Override scales with closest gt features
        for i in range(len(all_pred2gt_idx)):
            pred_feats[i, 4:7, :] = gt_feats[i, 4:7, all_pred2gt_idx[i]]
    if not args.compress_rots:
        # Override rotations with closest gt features
        for i in range(len(all_pred2gt_idx)):
            pred_feats[i, 7:11, :] = gt_feats[i, 7:11, all_pred2gt_idx[i]]

    # Prepare data to write to PLY file
    output_data = np.concatenate((pred_xyzs, pred_feats), axis=2).reshape(
        -1, pred_xyzs.shape[2] + pred_feats.shape[2]
    )

    # Write header
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(num_points))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property float opacity\n")
        f.write("property float scale_0\n")
        f.write("property float scale_1\n")
        f.write("property float scale_2\n")
        f.write("property float rot_0\n")
        f.write("property float rot_1\n")
        f.write("property float rot_2\n")
        f.write("property float rot_3\n")
        f.write("element face 0\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

    # Write data
    with open(path, "a") as f:
        np.savetxt(f, output_data, fmt="%0.4f")


def save_pcd(dir, name, xyzs, feats):
    # Inputs:
    # xyzs: an (n, 3) numpy array for x, y, z coordinates
    # feats: an (n, 46) numpy array for features including normals, other features, and transformations
    # Ensure that xyzs and feats arrays are properly aligned in terms of row count
    
    path = os.path.join(dir, name)
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {xyzs.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        *["property float f_dc_{}".format(i) for i in range(3)],
        *["property float f_rest_{}".format(i) for i in range(45)],
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header"
    ]
    with open(path, "ab") as f:
        xyzs_and_feats = np.concatenate((xyzs, feats), axis=1)
        np.savetxt(f, xyzs_and_feats, fmt="%s")
    # with open(path, 'wb') as f:
    #     f.write('\n'.join(header).encode('utf-8'))
    #     f.write(b'\n')

    #     # Prepare data for binary format
    #     combined_data = np.concatenate((xyzs, feats), axis=1)
    #     for row in combined_data:
    #         packed_data = struct.pack('<' + 'f' * combined_data.shape[1], *row)
    #         f.write(packed_data)


def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ["yes", "true", "t", "y"]:
        return True
    elif val.lower() in ["no", "false", "f", "n"]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
