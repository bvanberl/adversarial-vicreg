from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib
import tqdm

from torch import nn, optim
from torchvision import datasets, transforms
import torch
from pgd import pgd

import resnet

gpu = torch.device("cuda")

def get_arguments():

	parser = argparse.ArgumentParser(
		description="Evaluate a pretrained model on ImageNet"
	)
	
	parser.add_argument("--data-dir", type=Path, default="./data/", required=False,
						help='Path to the dataset')
	parser.add_argument("--dataset-name", type=str, default="cifar10", required=False,
						help='Dataset name')
	parser.add_argument(
		"--exp-dir",
		default="./checkpoint/lincls_001/",
		type=Path,
		metavar="DIR",
		help="path to checkpoint directory",
	)

	#Optim
	parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )

	# Model
	parser.add_argument("--arch", type=str, default="resnet18")

	#Augmentation
	parser.add_argument("--min-crop-area", type=float, default=0.75,
						help='Minimum crop area, as fraction of original area')
	parser.add_argument("--max-crop-area", type=float, default=1.0,
						help='Maximum crop area, as fraction of original area')
	parser.add_argument("--flip-prob", type=float, default=0.5,
						help='Probability of applying horizontal clip')
	
	#pgd
	parser.add_argument("--pgd-step-size", type=float, default=0.003,
                        help='PGD attack step size')
    parser.add_argument("--pgd-epsilon", type=float, default=0.031,
                        help='PGD attack budget')
    parser.add_argument("--pgd-perturb-steps", type=int, default=10,
                        help='PGD attack number of steps')
    parser.add_argument("--pgd-distance", type=str, default="l_inf",
                        help='Norm for PGD attack (either "l_inf" or "l_2"')

	return parser

def main():

	parser = get_arguments()
	args = parser.parse_args()

	evaluate_robust(args)

def evaluate_robust(args):

	backbone, embedding = resnet.__dict__[args.arch](zero_init_residual=True)

	#adjust the number of classes depending on the selected dataset
	if args.dataset_name == 'cifar10':
		head = nn.Linear(embedding, 10)
	elif args.dataset_name == 'cifar100':
		head = nn.Linear(embedding, 100)
	elif args.dataset_name == 'imagenet':
		head = nn.Linear(embedding, 1000)

	head.weight.data.normal_(mean=0.0, std=0.01)
	head.bias.data.zero_()
	model = nn.Sequential(backbone, head)
	model.cuda(gpu)

	#load checkpoint
	ckpt = torch.load(args.exp_dir / "checkpoint.pth", map_location="cpu")
	model.load_state_dict(ckpt["model"])

	if os.path.exists(args.data_dir):
		download = False
	else:
		download = True

	if args.dataset_name == 'cifar10' or args.dataset_name == 'cifar100':
		image_dim = 32
		resize_dim = 35
	elif args.dataset_name == 'imagenet':
		image_dim = 224
		resize_dim = 256

	transform_val = transforms.Compose(
					  [
						  transforms.Resize(resize_dim),
						  transforms.CenterCrop(image_dim),
						  transforms.ToTensor(),
					  ]
				)

	if args.dataset_name == 'cifar10':
		dataset_val = datasets.CIFAR10(root=args.data_dir, train=False, download=download, transform=transform_val)
	elif args.dataset_name == 'cifar100':
		dataset_val = datasets.CIFAR100(root=args.data_dir, train=False, download=download, transforms=transform_val)
	elif args.dataset_name == 'imagenet':
		dataset_val = datasets.ImageNet(root=args.data_dir, train=False, download=download, transform=transform_val)
	else:
		raise Exception(f"Dataset string not supported: {args.dataset_name}")

	val_loader = torch.utils.data.DataLoader(
		dataset_val,
		batch_size=1,
		pin_memory=False,
		shuffle=False,
	)

	#evaluate
	model.eval()

	test_accu = 0

	#top1 = AverageMeter("Acc@1")
	with torch.no_grad():
		for images, target in val_loader:
			output = model(images.cuda(gpu, non_blocking=True))
			if torch.argmax(output.cuda(gpu, non_blocking=True)) == target.cuda(gpu, non_blocking=True):
				test_accu += 1
			#acc1 = accuracy(output, target.cuda(gpu, non_blocking=True), topk=(1,))
			#top1.update(acc1[0].item(), images.size(0))
			#print("test accuracy: " + str(acc1))

		print("test accuracy: " + str(test_accu/len(dataset_val)))

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self, name, fmt=":f"):
		self.name = name
		self.fmt = fmt
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
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
		return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res

if __name__ == "__main__":
	main()



