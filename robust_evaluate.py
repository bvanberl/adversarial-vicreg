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
from torch.autograd import Variable
import torch
from pgd import pgd

import resnet

gpu = torch.device("cuda")
device = gpu

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
		"--batch-size", default=200, type=int, metavar="N", help="mini-batch size"
	)

	# Model
	parser.add_argument("--arch", type=str, default="resnet18")
	parser.add_argument("--mlp", default="512-512-512",
						help='Size and number of layers of the MLP expander head')

	#Augmentation
	parser.add_argument("--min-crop-area", type=float, default=0.75,
						help='Minimum crop area, as fraction of original area')
	parser.add_argument("--max-crop-area", type=float, default=1.0,
						help='Maximum crop area, as fraction of original area')
	parser.add_argument("--flip-prob", type=float, default=0.5,
						help='Probability of applying horizontal clip')

	#pgd
	parser.add_argument('--epsilon', default=0.031,
						help='perturbation')
	parser.add_argument('--num-steps', default=20,
						help='perturb number of steps')
	parser.add_argument('--step-size', default=0.003,
						help='perturb step size')
	parser.add_argument('--random',
						default=True,
						help='random initialization for PGD')

	return parser

parser = get_arguments()
args = parser.parse_args()

def main():

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
						  #transforms.Resize(resize_dim),
						  #transforms.CenterCrop(image_dim),
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
		batch_size=args.batch_size,
		pin_memory=False,
		shuffle=False,
	)

	#evaluate
	model.eval()

	eval_adv_test_whitebox(model, gpu, val_loader)

def _pgd_whitebox(model,
				  X,
				  y,
				  epsilon=args.epsilon,
				  num_steps=args.num_steps,
				  step_size=args.step_size):
	out = model(X)
	err = (out.data.max(1)[1] != y.data).float().sum()
	X_pgd = Variable(X.data, requires_grad=True)
	if args.random:
		random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
		X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

	for _ in range(num_steps):
		opt = optim.SGD([X_pgd], lr=1e-3)
		opt.zero_grad()

		with torch.enable_grad():
			loss = nn.CrossEntropyLoss()(model(X_pgd), y)
		loss.backward()
		eta = step_size * X_pgd.grad.data.sign()
		X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
		eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
		X_pgd = Variable(X.data + eta, requires_grad=True)
		X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
	err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
	print('err pgd (white-box): ', err_pgd)
	return err, err_pgd

def eval_adv_test_whitebox(model, device, test_loader):
	"""
	evaluate model by white-box attack
	"""
	model.eval()
	robust_err_total = 0
	natural_err_total = 0

	for data, target in test_loader:
		data, target = data.to(device), target.to(device)
		# pgd attack
		X, y = Variable(data, requires_grad=True), Variable(target)
		err_natural, err_robust = _pgd_whitebox(model, X, y)
		robust_err_total += err_robust
		natural_err_total += err_natural
	print('natural_err_total: ', natural_err_total)
	print('robust_err_total: ', robust_err_total)

if __name__ == "__main__":
	main()



