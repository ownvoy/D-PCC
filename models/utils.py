import numpy as np
import os
import torch
from einops import rearrange
import open3d as o3d
import random
import argparse


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


# def save_pcd(dir, name, xyzs, feats):
#     # input: (n, 3)
#     path = os.path.join(dir, name)
#     f = open(path, "w")
#     f.write("ply\n")
#     f.write("format ascii 1.0\n")
#     f.write("element vertex " + str(xyzs.shape[0]) + "\n")
#     f.write("property float x\n")
#     f.write("property float y\n")
#     f.write("property float z\n")
#     # f.write("property float nx\n")
#     # f.write("property float ny\n")
#     # f.write("property float nz\n")
#     f.write("property float opacity\n")
#     f.write("property float scale_0\n")
#     f.write("property float scale_1\n")
#     f.write("property float scale_2\n")
#     f.write("property float rot_0\n")
#     f.write("property float rot_1\n")
#     f.write("property float rot_2\n")
#     f.write("property float rot_3\n")
#     f.write("element face 0\n")
#     f.write("property list uchar int vertex_indices\n")
#     f.write("end_header\n")
#     f.close()

#     with open(path, "ab") as f:
#         xyzs_and_feats = np.concatenate((xyzs, feats), axis=1)
#         np.savetxt(f, xyzs_and_feats, fmt="%s")
def save_pcd(dir, name, xyzs, normals=None):
    # input: (n, 3)
    path = os.path.join(dir, name)
    f = open(path, 'w')
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex " + str(xyzs.shape[0]) + "\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    if isinstance(normals, np.ndarray):
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
    f.write("element face 0\n")
    f.write("property list uchar int vertex_indices\n")
    f.write("end_header\n")
    f.close()

    with open(path, 'ab') as f:
        if isinstance(normals, np.ndarray):
            # (n, 6)
            xyzs_and_normals = np.concatenate((xyzs, normals), axis=1)
            np.savetxt(f, xyzs_and_normals, fmt='%s')
        else:
            np.savetxt(f, xyzs, fmt='%s')

def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ["yes", "true", "t", "y"]:
        return True
    elif val.lower() in ["no", "false", "f", "n"]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
