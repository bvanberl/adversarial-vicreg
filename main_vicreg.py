# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets

import augmentations as aug
from distributed import init_distributed_mode
from pgd import pgd

import resnet
import mlflow
import mlflow.pytorch


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="/data/", required=True,
                        help='Path to the dataset')
    parser.add_argument("--dataset-name", type=str, default="cifar10", required=False,
                        help='Dataset name')
    parser.add_argument("--img-dim", type=int, default=32, required=False,
                        help='Dataset name')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Model
    parser.add_argument("--arch", type=str, default="resnet18",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="512-512-512",
                        help='Size and number of layers of the MLP expander head')

    # Optim
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=256,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')

    # Running
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    # Adversarial training
    parser.add_argument("--adv-train", default=False, action="store_true",
                        help='Add adversarial regularizer to loss')
    parser.add_argument("--adv-coeff", type=float, default=1.0,
                        help='Adversarial regularization loss coefficient')
    parser.add_argument("--pgd-step-size", type=float, default=0.003,
                        help='PGD attack step size')
    parser.add_argument("--pgd-epsilon", type=float, default=0.031,
                        help='PGD attack budget')
    parser.add_argument("--pgd-perturb-steps", type=int, default=10,
                        help='PGD attack number of steps')
    parser.add_argument("--pgd-distance", type=str, default="l_inf",
                        help='Norm for PGD attack (either "l_inf" or "l_2"')

    # Augmentations
    parser.add_argument("--gaussian-sigma", type=float, default=2.0,
                        help='Amount of noise in Gaussian blur')
    parser.add_argument("--gaussian-prob", type=float, default=0.5,
                        help='Probability of applying Gaussian blur')
    parser.add_argument("--solarization-prob", type=float, default=0.1,
                        help='Probability of applying solarization')
    parser.add_argument("--grayscale-prob", type=float, default=0.2,
                        help='Probability of applying grayscale')
    parser.add_argument("--color-jitter-prob", type=float, default=0.8,
                        help='Probability of applying random crop')
    parser.add_argument("--min-crop-area", type=float, default=0.08,
                        help='Minimum crop area, as fraction of original area')
    parser.add_argument("--max-crop-area", type=float, default=1.0,
                        help='Maximum crop area, as fraction of original area')
    parser.add_argument("--flip-prob", type=float, default=0.5,
                        help='Probability of applying horizontal clip')
    return parser


def main(args):
    torch.backends.cudnn.benchmark = True
    #init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

    training_on_imagenet = args.dataset_name == 'imagenet'
    transforms = aug.TrainTransform(args.img_dim, gaussian_sigma=args.gaussian_sigma, gaussian_prob=args.gaussian_prob,
                                    solarization_prob=args.solarization_prob, grayscale_prob=args.grayscale_prob,
                                    color_jitter_prob=args.color_jitter_prob, min_crop_area=args.min_crop_area,
                                    max_crop_area=args.max_crop_area, flip_prob=args.flip_prob,
                                    imagenet_norm=training_on_imagenet)

    if os.path.exists(args.data_dir):
        download = False
    else:
        download = True
    if args.dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=download, transform=transforms)
    elif args.dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=args.data_dir, train=True, download=download, transforms=transforms)
    elif args.dataset_name == 'imagenet':
        dataset = datasets.ImageNet(root=args.data_dir, train=True, download=download, transform=transforms)
    else:
        raise Exception(f"Dataset string not supported: {args.dataset_name}")

    #sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=True,
    )

    model = VICReg(args).cuda(gpu)
    #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in tqdm.tqdm(range(start_epoch, args.epochs)):
        #sampler.set_epoch(epoch)
        for step, ((x, y), _) in enumerate(loader, start=epoch * len(loader)):
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)

            lr = adjust_learning_rate(args, optimizer, loader, step)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    time=int(current_time - start_time),
                    lr=lr,
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                for key, value in stats.items():
                    mlflow.log_metric(key, value, step=step)
                last_logging = current_time
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "model.pth")
    if args.rank == 0:
        torch.save(model.state_dict(), args.exp_dir / args.arch + ".pth")


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(args, self.embedding)
        self.adv_train = True if args.adv_coeff > 0 else False
        self.model_stacked = nn.Sequential(self.backbone, self.projector)

        print("Architecture details:")
        print(self.backbone)
        print(self.projector)

    def vicreg_loss(self, x, y):
        repr_loss = F.mse_loss(x, y)

        # x = torch.cat(FullGatherLayer.apply(x), dim=0)
        # y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
                self.args.sim_coeff * repr_loss
                + self.args.std_coeff * std_loss
                + self.args.cov_coeff * cov_loss
        )
        return loss


    def adversarial_regularizer(self, x, z_x, y, z_y):

        # Determine adversarial examples
        x_adv = pgd(self.model_stacked, x, self.vicreg_loss, step_size=self.args.pgd_step_size, epsilon=self.args.pgd_epsilon,
                    perturb_steps=self.args.pgd_perturb_steps, distance=self.args.pgd_distance)

        y_adv = pgd(self.model_stacked, y, self.vicreg_loss, step_size=self.args.pgd_step_size, epsilon=self.args.pgd_epsilon,
                    perturb_steps=self.args.pgd_perturb_steps, distance=self.args.pgd_distance)

        # Compute embeddings for adversarial examples
        z_x_adv = self.model_stacked(x_adv)
        z_y_adv = self.model_stacked(y_adv)

        # Compute VICReg loss for adversarial/natural pairs
        x_adv_loss = self.vicreg_loss(z_x, z_x_adv)
        y_adv_loss = self.vicreg_loss(z_y, z_y_adv)
        return (x_adv_loss + y_adv_loss) / 2.


    def forward(self, x, y):
        z_x = self.model_stacked(x)
        z_y = self.model_stacked(y)

        total_loss = self.vicreg_loss(z_x, z_y)

        if self.adv_train:
            adv_loss = self.adversarial_regularizer(x, z_x, y, z_y)
            total_loss += args.adv_coeff * adv_loss

        return total_loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    tracking_uri = "http://ec2-18-206-121-84.compute-1.amazonaws.com"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    expr_name = "main_vicreg.py"
    try:
        # create a new experiment (do not replace)
        s3_bucket = "s3://mlflow-research-runs" # replace this value
        experiment = mlflow.create_experiment (expr_name, s3_bucket)
    except Exception as e:
        # print (e)
        experiment = mlflow.get_experiment_by_name(expr_name)
    mlflow.set_experiment (expr_name)
        
    with mlflow.start_run() as run:  
    # Log our parameters into mlflow
      for key, value in vars(args).items():
          mlflow.log_param(key, value)
      main(args)