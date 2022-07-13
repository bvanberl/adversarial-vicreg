# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


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
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import resnet
import mlflow
import mlflow.pytorch

gpu = torch.device("cuda")

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    # Data
    parser.add_argument("--data-dir", type=Path, default="./data/", required=False,
                        help='Path to the dataset')
    parser.add_argument("--dataset-name", type=str, default="cifar10", required=False,
                        help='Dataset name')
    #parser.add_argument(
        #"--train-percent",
        #default=100,
        #type=int,
        #choices=(100, 10, 1),
        #help="size of traing set in percent",
    #)

    # Checkpoint
    parser.add_argument("--pretrained", default="./backbone/resnet18.pth", type=Path, help="path to pretrained model")
    parser.add_argument(
        "--exp-dir",
        default="./checkpoint/lincls/",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )
    parser.add_argument(
        "--print-freq", default=100, type=int, metavar="N", help="print frequency"
    )

    # Model
    parser.add_argument("--arch", type=str, default="resnet18")

    # Optim
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size", default=128, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--lr-backbone",
        default=0.0,
        type=float,
        metavar="LR",
        help="backbone base learning rate",
    )
    parser.add_argument(
        "--lr-head",
        default=0.001,
        type=float,
        metavar="LR",
        help="classifier base learning rate",
    )
    parser.add_argument(
        "--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument(
        "--weights",
        default="freeze",
        type=str,
        choices=("finetune", "freeze"),
        help="finetune or freeze resnet weights",
    )

    # Running
    #parser.add_argument(
        #"--workers",
        #default=8,
        #type=int,
        #metavar="N",
        #help="number of data loader workers",
    #)

    #Augmentation
    parser.add_argument("--min-crop-area", type=float, default=0.75,
                        help='Minimum crop area, as fraction of original area')
    parser.add_argument("--max-crop-area", type=float, default=1.0,
                        help='Maximum crop area, as fraction of original area')
    parser.add_argument("--flip-prob", type=float, default=0.5,
                        help='Probability of applying horizontal clip')

    return parser

parser = get_arguments()
args = parser.parse_args()

def main():
    #if args.train_percent in {1, 10}:
        #args.train_files = urllib.request.urlopen(
            #f"https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.train_percent}percent.txt"
        #).readlines()
    #args.ngpus_per_node = torch.cuda.device_count()
    #if "SLURM_JOB_ID" in os.environ:
        #signal.signal(signal.SIGUSR1, handle_sigusr1)
        #signal.signal(signal.SIGTERM, handle_sigterm)
    # single-node distributed training
    args.rank = 0
    #args.dist_url = f"tcp://localhost:{random.randrange(49152, 65535)}"
    #args.world_size = args.ngpus_per_node
    #torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)
    tracking_uri = "http://ec2-18-206-121-84.compute-1.amazonaws.com"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    expr_name = "evaluate.py"
    try:
        # create a new experiment (do not replace)
        s3_bucket = "s3://mlflow-research-runs" # replace this value
        experiment = mlflow.create_experiment (expr_name, s3_bucket)
    except Exception as e:
        print (e)
        experiment = mlflow.get_experiment_by_name(expr_name)
    mlflow.set_experiment (expr_name)

    with mlflow.start_run() as run:  
    # Log our parameters into mlflow
        for key, value in vars(args).items():
            mlflow.log_param(key, value)
        main_worker(args)

#def main_woker(gpu, args):
def main_worker(args):
    #args.rank += gpu
    #torch.distributed.init_process_group(
        #backend="nccl",
        #init_method=args.dist_url,
        #world_size=args.world_size,
        #rank=args.rank,
    #)

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

    #torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    backbone, embedding = resnet.__dict__[args.arch](zero_init_residual=True)
    state_dict = torch.load(args.pretrained, map_location="cpu")
    backbone_state_dict = {s.replace("backbone.", ""): state_dict[s] for s in state_dict if "backbone." in s}
    backbone.load_state_dict(backbone_state_dict, strict=False)

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

    if args.weights == "freeze":
        backbone.requires_grad_(False)
        head.requires_grad_(True)
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    param_groups = [dict(params=head.parameters(), lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(dict(params=backbone.parameters(), lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    if (args.exp_dir / "checkpoint.pth").is_file():
        ckpt = torch.load(args.exp_dir / "checkpoint.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        best_acc = ckpt["best_acc"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        start_epoch = 0
        best_acc = argparse.Namespace(top1=0, top5=0)
    
    if os.path.exists(args.data_dir):
        download = False
    else:
        download = True
    
    #normalize = transforms.Normalize(
        #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #)

    if args.dataset_name == 'cifar10' or args.dataset_name == 'cifar100':
        image_dim = 32
        resize_dim = 35
    elif args.dataset_name == 'imagenet':
        image_dim = 224
        resize_dim = 256

    transform_train = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(image_dim, scale=(args.min_crop_area, args.max_crop_area)),
                        transforms.RandomHorizontalFlip(p=args.flip_prob),
                        transforms.ToTensor(),
                    ]
                )
    
    transform_val = transforms.Compose(
                      [
                          transforms.Resize(resize_dim),
                          transforms.CenterCrop(image_dim),
                          transforms.ToTensor(),
                      ]
                )

    if args.dataset_name == 'cifar10':
        dataset_train = datasets.CIFAR10(root=args.data_dir, train=True, download=download, transform=transform_train)
        dataset_val = datasets.CIFAR10(root=args.data_dir, train=False, download=download, transform=transform_val)
    elif args.dataset_name == 'cifar100':
        dataset_train = datasets.CIFAR100(root=args.data_dir, train=True, download=download, transforms=transform_train)
        dataset_val = datasets.CIFAR100(root=args.data_dir, train=False, download=download, transforms=transform_val)
    elif args.dataset_name == 'imagenet':
        dataset_train = datasets.ImageNet(root=args.data_dir, train=True, download=download, transform=transform_train)
        dataset_val = datasets.ImageNet(root=args.data_dir, train=False, download=download, transform=transform_val)
    else:
        raise Exception(f"Dataset string not supported: {args.dataset_name}")

    # Data loading code
    #traindir = args.data_dir / "train"
    #valdir = args.data_dir / "val"

    #train_dataset = datasets.ImageFolder(
        #traindir,
        #transforms.Compose(
            #[
                #transforms.RandomResizedCrop(args.img_dim),
                #transforms.RandomHorizontalFlip(),
                #transforms.ToTensor(),
                #normalize,
            #]
        #),
    #)
    #val_dataset = datasets.ImageFolder(
        #valdir,
        #transforms.Compose(
            #[
                #transforms.Resize(256),
                #transforms.CenterCrop(args.img_dim),
                #transforms.ToTensor(),
                #normalize,
            #]
        #),
    #)

    #if args.train_percent in {1, 10}:
        #train_dataset.samples = []
        #for fname in args.train_files:
            #fname = fname.decode().strip()
            #cls = fname.split("_")[0]
            #train_dataset.samples.append(
                #(traindir / cls / fname, train_dataset.class_to_idx[cls])
            #)

    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #kwargs = dict(
        #batch_size=args.batch_size // args.world_size,
        #num_workers=args.workers,
        #pin_memory=True,
    #)
    #train_loader = torch.utils.data.DataLoader(
        #train_dataset, sampler=train_sampler, **kwargs
    #)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=True,
    )
    #val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=True,
    )

    start_time = time.time()
    for epoch in tqdm.tqdm(range(start_epoch, args.epochs)):
        adjust_learning_rate(optimizer, epoch)
        # train
        if args.weights == "finetune":
            model.train()
        elif args.weights == "freeze":
            model.eval()
        else:
            assert False
        #train_sampler.set_epoch(epoch)
        for step, (images, target) in enumerate(
            train_loader, start=epoch * len(train_loader)
        ):
            #output = model(images.cuda(gpu, non_blocking=True))
            #loss = criterion(output, target.cuda(gpu, non_blocking=True))
            loss = adv_loss(model, images.cuda(gpu, non_blocking=True), target.cuda(gpu, non_blocking=True))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0:
                #torch.distributed.reduce(loss.div_(args.world_size), 0)
                if args.rank == 0:
                    pg = optimizer.param_groups
                    lr_head = pg[0]["lr"]
                    lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
                    stats = dict(
                        epoch=epoch,
                        step=step,
                        lr_backbone=lr_backbone,
                        lr_head=lr_head,
                        loss=loss.item(),
                        time=int(time.time() - start_time),
                    )
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
                    for key, value in stats.items():
                        mlflow.log_metric(key, value, step=step)

        # evaluate
        model.eval()
        if args.rank == 0:
            top1 = AverageMeter("Acc@1")
            top5 = AverageMeter("Acc@5")
            with torch.no_grad():
                for images, target in val_loader:
                    output = model(images.cuda(gpu, non_blocking=True))
                    acc1, acc5 = accuracy(
                        output, target.cuda(gpu, non_blocking=True), topk=(1, 5)
                    )
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
            best_acc.top1 = max(best_acc.top1, top1.avg)
            best_acc.top5 = max(best_acc.top5, top5.avg)
            stats = dict(
                epoch=epoch,
                acc1=top1.avg,
                acc5=top5.avg,
                best_acc1=best_acc.top1,
                best_acc5=best_acc.top5,
            )
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)
            for key, value in stats.items():
                mlflow.log_metric(key, value, step=step)

        scheduler.step()
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                best_acc=best_acc,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
            )
            torch.save(state, args.exp_dir / "checkpoint.pth")


def adv_loss(model,
                x_natural,
                y,
                step_size=0.007,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
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
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr_backbone = args.lr_backbone
    lr_head = args.lr_head

    if epoch >= 25:
        lr_backbone = args.lr_backbone * 0.1
        lr_head = args.lr_head * 0.1
    if epoch >= 50:
        lr_backbone = args.lr_backbone * 0.01
        lr_head = args.lr_head * 0.01
    if epoch >= 75:
        lr_backbone = args.lr_backbone * 0.001
        lr_head = args.lr_head * 0.001
        
    optimizer.param_groups[0]['lr'] = lr_head

    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr_backbone

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


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