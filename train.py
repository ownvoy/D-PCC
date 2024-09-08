import torch
import os
import sys

sys.path.append(os.getcwd())
from models.autoencoder import AutoEncoder
import time
from tqdm import tqdm
import wandb
import argparse
from datetime import datetime
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.utils import AverageMeter, str2bool
from dataset.dataset import CompressDataset
from args.shapenet_args import parse_shapenet_args
from args.semantickitti_args import parse_semantickitti_args
from torch.optim.lr_scheduler import StepLR
import open3d as o3d
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist

chamfer_dist = chamfer_3DDist()


wandb.init(project="dpcc2", entity="ownvoy")


def train(args):
    start = time.time()

    if args.batch_size > 1:
        print("The performance will degrade if batch_size is larger than 1!")

    # load data
    train_dataset = CompressDataset(
        data_path=args.train_data_path,
        cube_size=args.train_cube_size,
        batch_size=args.batch_size,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, shuffle=True, batch_size=args.batch_size
    )

    # set up folders for checkpoints
    str_time = datetime.now().isoformat()
    print("Experiment Time:", str_time)
    checkpoint_dir = os.path.join(args.output_path, str_time, "ckpt")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # create the model
    model = AutoEncoder(args)
    model = model.cuda()
    print("Training Arguments:", args)
    print("Model Architecture:", model)

    # optimizer for autoencoder
    parameters = set(
        p for n, p in model.named_parameters() if not n.endswith(".quantiles")
    )
    optimizer = optim.Adam(parameters, lr=args.lr)
    # lr scheduler
    scheduler_steplr = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
    # optimizer for entropy bottleneck
    aux_parameters = set(
        p for n, p in model.named_parameters() if n.endswith(".quantiles")
    )
    aux_optimizer = optim.Adam(aux_parameters, lr=args.aux_lr)
    best_loss = 999999

    # best validation metric
    best_val_chamfer_loss = float("inf")

    # train
    for epoch in range(args.epochs):
        epoch_loss = AverageMeter()
        epoch_chamfer_loss = AverageMeter()
        epoch_density_loss = AverageMeter()
        epoch_pts_num_loss = AverageMeter()
        epoch_latent_xyzs_loss = AverageMeter()
        epoch_feats_loss = AverageMeter()
        epoch_bpp_loss = AverageMeter()
        epoch_aux_loss = AverageMeter()

        model.train()
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, input_dict in enumerate(train_loader_tqdm):
            # input: (b, n, c)
            input = input_dict["xyzs"].cuda()
            # input: (b, c, n)
            input = input.permute(0, 2, 1).contiguous()

            feats = input_dict["feats"].cuda().permute(0, 2, 1).contiguous()
            input = torch.cat((input, feats), dim=1)
            # print(input.shape)

            # model forward
            decompressed_xyzs, loss, loss_items, bpp = model(input)
            epoch_loss.update(loss.item())
            epoch_chamfer_loss.update(loss_items["chamfer_loss"])
            # epoch_density_loss.update(loss_items["density_loss"])
            # epoch_pts_num_loss.update(loss_items["pts_num_loss"])
            epoch_latent_xyzs_loss.update(loss_items["latent_xyzs_loss"])
            epoch_feats_loss.update(loss_items["feats_loss"])
            epoch_bpp_loss.update(loss_items["bpp_loss"])

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the parameters of entropy bottleneck
            aux_loss = model.feats_eblock.loss()
            if args.quantize_latent_xyzs == True:
                aux_loss += model.xyzs_eblock.loss()
            epoch_aux_loss.update(aux_loss.item())
            aux_optimizer.zero_grad()
            aux_loss.backward()
            aux_optimizer.step()
            train_loader_tqdm.set_postfix(
                loss=epoch_loss.get_avg(),
                chamfer_loss=epoch_chamfer_loss.get_avg(),
                aux_loss=epoch_aux_loss.get_avg(),
            )

            # print loss
            if (i + 1) % args.print_freq == 0:
                print(
                    "train epoch: %d/%d, iters: %d/%d, loss: %f, avg chamfer loss: %f, "
                    "avg latent xyzs loss: %f, avg feats loss: %f, "
                    "avg bpp loss: %f, avg aux loss: %f"
                    % (
                        epoch + 1,
                        args.epochs,
                        i + 1,
                        len(train_loader),
                        epoch_loss.get_avg(),
                        epoch_chamfer_loss.get_avg(),
                        # epoch_density_loss.get_avg(),
                        # epoch_pts_num_loss.get_avg(),
                        epoch_latent_xyzs_loss.get_avg(),
                        epoch_feats_loss.get_avg(),
                        epoch_bpp_loss.get_avg(),
                        epoch_aux_loss.get_avg(),
                    )
                )

        scheduler_steplr.step()

        # print loss
        interval = time.time() - start
        print(
            "train epoch: %d/%d, time: %d mins %.1f secs, loss: %f, avg chamfer loss: %f, "
            "avg latent xyzs loss: %f, avg feats loss: %f, "
            "avg bpp loss: %f, avg aux loss: %f"
            % (
                epoch + 1,
                args.epochs,
                interval / 60,
                interval % 60,
                epoch_loss.get_avg(),
                epoch_chamfer_loss.get_avg(),
                # epoch_density_loss.get_avg(),
                # epoch_pts_num_loss.get_avg(),
                epoch_latent_xyzs_loss.get_avg(),
                epoch_feats_loss.get_avg(),
                epoch_bpp_loss.get_avg(),
                epoch_aux_loss.get_avg(),
            )
        )
        wandb.log(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss.get_avg(),
                "avg chamfer loss": epoch_chamfer_loss.get_avg(),
                # "avg density loss": epoch_density_loss.get_avg(),
                # "avg pts num loss": epoch_pts_num_loss.get_avg(),
                "avg latent xyzs loss": epoch_latent_xyzs_loss.get_avg(),
                "avg feats loss": epoch_feats_loss.get_avg(),
                "avg bpp loss": epoch_bpp_loss.get_avg(),
                "avg aux loss": epoch_aux_loss.get_avg(),
                "time per epoch": interval,
            }
        )
        if best_loss >= epoch_loss.get_avg():
            best_loss = epoch_loss.get_avg()
            model_save_path = os.path.join(
                "./model_checkpoints", f"cube{args.train_cube_size}_epoch_{epoch+1}.pth"
            )
            torch.save(model.state_dict(), model_save_path)
            wandb.save(model_save_path)
            print(f"Model saved to {model_save_path}")


def reset_model_args(train_args, model_args):
    for arg in vars(train_args):
        setattr(model_args, arg, getattr(train_args, arg))


def parse_train_args():
    parser = argparse.ArgumentParser(description="Training Arguments")

    parser.add_argument(
        "--dataset", default="semantickitti", type=str, help="shapenet or semantickitti"
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="the performance will degrade if batch_size is larger than 1!",
    )
    parser.add_argument(
        "--downsample_rate",
        default=[1 / 6, 1 / 6, 1 / 6],
        nargs="+",
        type=float,
        help="downsample rate",
    )
    parser.add_argument(
        "--max_upsample_num",
        default=[6,6,6],
        nargs="+",
        type=int,
        help="max upsmaple number, reversely symmetric with downsample_rate",
    )
    parser.add_argument(
        "--bpp_lambda", default=1e-2, type=float, help="bpp loss coefficient"
    )
    # feats compression
    parser.add_argument(
        "--compress_feats",
        default=True,
        type=str2bool,
        help="whether compress featss",
    )
    # compress latent xyzs
    parser.add_argument(
        "--quantize_latent_xyzs",
        default=True,
        type=str2bool,
        help="whether compress latent xyzs",
    )
    parser.add_argument(
        "--latent_xyzs_conv_mode",
        default="mlp",
        type=str,
        help="latent xyzs conv mode, mlp or edge_conv",
    )
    # sub_point_conv mode
    parser.add_argument(
        "--sub_point_conv_mode",
        default="mlp",
        type=str,
        help="sub-point conv mode, mlp or edge_conv",
    )
    parser.add_argument(
        "--output_path", default="./output", type=str, help="output path"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train_args = parse_train_args()
    assert train_args.dataset in ["shapenet", "semantickitti"]

    if train_args.dataset == "shapenet":
        model_args = parse_shapenet_args()
    else:
        model_args = parse_semantickitti_args()

    reset_model_args(train_args, model_args)
    wandb.config.update(vars(model_args))
    wandb.config.update(vars(train_args))

    train(model_args)
