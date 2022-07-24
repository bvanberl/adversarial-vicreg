
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

import trades_resnet as resnet
import trades_wide_resnet as wide_resnet

from main_vicreg import VICReg
from main_vicreg import LARS
from main_vicreg import exclude_bias_and_norm

def get_arguments():
	parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

	# Data
	parser.add_argument("--data-dir", type=Path, default="../data/", required=False,
						help='Path to the dataset')
	parser.add_argument("--dataset-name", type=str, default="cifar10", required=False,
						help='Dataset name')
	parser.add_argument("--img-dim", type=int, default=32, required=False,
						help='Dataset name')

	# Checkpoints
	parser.add_argument("--ckpt-dir", type=Path, default="./pretrained/resnet18_adv.pth",
						help='Path to the checkpoint folder')
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
	parser.add_argument("--epochs", type=int, default=50,
						help='Number of epochs')
	parser.add_argument("--batch-size", type=int, default=128,
						help='Effective batch size (per worker batch size is [batch-size] / world-size)')
	parser.add_argument("--base-lr", type=float, default=0.1,
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
	parser.add_argument("--avg-inv", default=True, action="store_true",
						help='Evaluate average invariance')
	parser.add_argument("--adv-coeff", type=float, default=5.0,
						help='Adversarial regularization loss coefficient')
	parser.add_argument("--swap-adv-loss", default=False, action="store_true",
						help='Adversarial regularization loss coefficient')
	parser.add_argument("--pgd-step-size", type=float, default=0.007,
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
	parser.add_argument("--min-crop-area", type=float, default=0.75,
						help='Minimum crop area, as fraction of original area')
	parser.add_argument("--max-crop-area", type=float, default=1.0,
						help='Maximum crop area, as fraction of original area')
	parser.add_argument("--flip-prob", type=float, default=0.5,
						help='Probability of applying horizontal clip')
	return parser

def main(args):
	torch.backends.cudnn.benchmark = True
	gpu = torch.device(args.device)

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
		dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=download, transform=transforms)
	elif args.dataset_name == 'cifar100':
		dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=download, transforms=transforms)
	elif args.dataset_name == 'imagenet':
		dataset = datasets.ImageNet(root=args.data_dir, train=False, download=download, transform=transforms)
	else:
		raise Exception(f"Dataset string not supported: {args.dataset_name}")

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

	ckpt = torch.load(args.ckpt_dir, map_location="cpu")
	model.load_state_dict(ckpt)

	model.eval()

	total_invariance = 0

	with torch.no_grad():
		for step, ((x, y), _) in enumerate(loader, start=len(loader)):
			x = x.cuda(gpu, non_blocking=True)
			y = y.cuda(gpu, non_blocking=True)

			loss = model.forward(x, y)
			model.zero_grad()

			total_invariance += loss

	avg_invariance = total_invariance/(len(loader) * 2)

	print("Average Invariance: " + str(avg_invariance))

if __name__ == "__main__":
	parser = argparse.ArgumentParser('Evaluate Average Invariance', parents=[get_arguments()])
	args = parser.parse_args()

	main(args)