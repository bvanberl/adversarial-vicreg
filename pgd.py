import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms as T
import resnet
import argparse
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)

def get_arguments():
	parser = argparse.ArgumentParser(description="PGD Attack with VICReg", add_help=False)

	# Model
	parser.add_argument("--arch", type=str, default="resnet50",
						help='Architecture of the backbone encoder network')
	parser.add_argument("--mlp", default="1024-1024-1024",
						help='Size and number of layers of the MLP expander head')

	# Optim
	parser.add_argument("--epochs", type=int, default=100,
						help='Number of epochs')
	parser.add_argument("--batch-size", type=int, default=2048,
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

	return parser

#for testing as an aribitary loss function
def mse(x_natural, x_adv, model):

	x_natural_embed = model.projector(model.backbone(x_natural))
	x_adv_embed = model.projector(model.backbone(x_adv))

	return F.mse_loss(x_adv_embed, x_natural_embed)

def pgd(model,
		x_natural,
		loss_func,
		step_size=0.003,
		epsilon=0.031,
		perturb_steps=10,
		distance='l_inf'):
	
	#set model to evaluation mode
	model.eval()
	batch_size = len(x_natural)

	if distance == 'l_inf':

		#initialize a random adversarial example
		x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

		for _ in range(perturb_steps):

			x_adv.requires_grad_()

			with torch.enable_grad():
				loss = loss_func(model(x_natural), model(x_adv))

			grad = torch.autograd.grad(loss, [x_adv])[0]
			x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
			x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
			x_adv = torch.clamp(x_adv, 0.0, 1.0)

	elif distance == 'l_2':

		delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
		delta = Variable(delta.data, requires_grad=True)

		#set up optimizer
		optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

		for _ in range(perturb_steps):

			x_adv = x_natural + delta

			#zero gradient of optimizer
			optimizer_delta.zero_grad()

			with torch.enable_grad():
				loss = (-1) * loss_func(model(x_natural), model(x_adv))

			loss.backward()

			# renorming gradient
			grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
			delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

			# avoid nan or inf if gradient is 0
			if (grad_norms == 0).any():
				delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
			optimizer_delta.step()

			# projection
			delta.data.add_(x_natural)
			delta.data.clamp_(0, 1).sub_(x_natural)
			delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)

		x_adv = x_natural + delta.data
		x_adv = torch.clamp(x_adv, 0.0, 1.0)

	#set back to training mode
	model.train()

	x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

	return x_adv

## COMMENTED DUE TO CIRCULAR IMPORTS

# def main(args):
#
# 	gpu = torch.device('cuda')
#
# 	model = VICReg(args).cuda(gpu)
# 	model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
#
# 	x_natural = np.load('./test_data/cifar10_X.npy')[1]
# 	x_natural = np.array(np.expand_dims(x_natural, axis=0), dtype=np.float32)
#
# 	#display image
# 	x_natural_plt = x_natural[0]
# 	plt.imshow(x_natural_plt)
# 	plt.show()
#
# 	x_natural = np.transpose(x_natural, (0, 3, 1, 2))
# 	x_natural = torch.from_numpy(x_natural).to(gpu)
#
# 	x_adv_l_inf = pgd(model, x_natural, mse)
# 	x_adv_l_two = pgd(model, x_natural, mse, distance='l_2')
#
# 	#display perturbed images
# 	x_adv_l_inf_plt = np.transpose(x_adv_l_inf.cpu(), (0, 2, 3, 1))[0]
# 	plt.imshow(x_adv_l_inf_plt)
# 	plt.show()
#
# 	x_adv_l_two_plt = np.transpose(x_adv_l_two.cpu(), (0, 2, 3, 1))[0]
# 	plt.imshow(x_adv_l_two_plt)
# 	plt.show()
#
# 	return
#
# if __name__ == "__main__":
#
# 	parser = argparse.ArgumentParser('PGD with VICReg', parents=[get_arguments()])
# 	args = parser.parse_args()
# 	main(args)
#
#
