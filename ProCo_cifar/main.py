import builtins
import time
import shutil
import math

import torch
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.datasets as datasets

from loss.logitadjust import LogitAdjust
from loss.proco import ProCoLoss
from tensorboardX import SummaryWriter
from models import resnet_cifar
from dataset.cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from autoaug import CIFAR10Policy, Cutout
from utils import shot_acc, bool_flag

import warnings
import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='', choices=['cifar100', 'cifar10'])
parser.add_argument('--data', default='', metavar='DIR')
parser.add_argument('--arch', default='resnet32',)
parser.add_argument('--workers', default=2, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--temp', default=0.1, type=float, help='scalar temperature for contrastive learning')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64*4), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--imb_factor', type=float,
                    metavar='IF', help='imbalanced factor', dest='imb_factor')
parser.add_argument('--lr', '--learning-rate', default=0.3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[160, 180], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 2e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=40, type=int,
                    metavar='N', help='print frequency (default: 40)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension of mlp head')
parser.add_argument('--warmup_epochs', default=5, type=int,
                    help='warmup epochs')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--cos', default=False, type=bool,
                    help='lr decays by cosine scheduler. ')
parser.add_argument('--use_norm', default=False, type=bool_flag,
                    help='cosine classifier.')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--reload', default=False, type=bool, help='load supervised model')
parser.add_argument('--mark', type=str)

parser.add_argument('--test', nargs='?', const=None, default=None, type=str, help='test model')


def main():
    args = parser.parse_args()

    args.store_name = args.mark

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    args.gpu = int(os.environ["LOCAL_RANK"])

    args.distributed = True
    args.multiprocessing_distributed = True

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu


    # suppress printing if not master
    if args.rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    print(args.epochs)
    if args.epochs == 200:
        args.schedule = [160, 180]
        args.warmup_epochs = 5
    elif args.epochs == 400:
        args.schedule = [360, 380]
        args.warmup_epochs = 10
    else:
        args.schedule = [args.epochs * 0.8, args.epochs * 0.9]
        args.warmup_epochs = 5 * args.epochs // 200

    print(args)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))



    dist.init_process_group(backend='nccl', init_method='env://',
            world_size=args.world_size, rank=args.rank)


    # create model
    print("=> creating model '{}'".format(args.arch))

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        raise ValueError('Unknown dataset')

    if args.arch == 'resnet32':
        model = resnet_cifar.Model(name='resnet32', num_classes=num_classes, feat_dim=args.feat_dim, use_norm=args.use_norm)
    else:
        raise NotImplementedError('This model is not supported')

    torch.cuda.set_device(args.gpu)

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])


    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, 
                                weight_decay=args.weight_decay)


    # prepare data
    augmentation_regular = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),    # add AutoAug
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]

    augmentation_sim_cifar = [
        transforms.RandomResizedCrop(size=32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([ transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation_sim_cifar), transforms.Compose(augmentation_sim_cifar)]

    val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root=args.data, imb_type='exp', imb_factor=args.imb_factor, rand_number=0, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(
                root=args.data, 
                train=False, 
                download=True, 
                transform=val_transform)
    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root=args.data, imb_type='exp', imb_factor=args.imb_factor, rand_number=0, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(
                root=args.data, 
                train=False, 
                download=True, 
                transform=val_transform)
    else:
        raise ValueError('Unknown dataset')


    #print(transform_train)
    print(f'===> Training data length {len(train_dataset)}')
    print(f'===> Val dataset size: {len(val_dataset)}')


    cls_num_list = train_dataset.get_cls_num_list()
    args.cls_num = len(cls_num_list)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)


    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)


    # define loss function (criterion)
    criterion_ce = LogitAdjust(cls_num_list).cuda(args.gpu)
    criterion_scl = ProCoLoss(contrast_dim=args.feat_dim, temperature=args.temp, num_classes=args.cls_num).cuda(args.gpu)

    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    best_acc1 = 0.0
    best_many, best_med, best_few = 0.0, 0.0, 0.0

    #filename = os.path.join(args.root_log, args.store_name, 'ckpt.pth.tar')
    #if os.path.exists(filename):
    #    args.resume = filename
    
    if args.test:
        if os.path.isfile(args.test):
            print("=> loading checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_many, best_med, best_few = checkpoint['best_many'], checkpoint['best_med'], checkpoint['best_few']

            criterion_scl = checkpoint['criterion_scl'].cuda(args.gpu)
            criterion_scl.reload_memory()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {}, best_acc1 {})"
                  .format(args.test, checkpoint['epoch'], best_acc1))
            args.reload = True
        else:
            print("=> no checkpoint found at '{}'".format(args.test))
            raise FileNotFoundError
    # optionally resume from a checkpoint
    elif args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_many, best_med, best_few = checkpoint['best_many'], checkpoint['best_med'], checkpoint['best_few']
            criterion_scl = checkpoint['criterion_scl'].cuda(args.gpu)
            criterion_scl.reload_memory()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {}, best_acc1 {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))

            if args.start_epoch == args.epochs:
                print("=> already trained for {} epochs, exiting".format(args.epochs))
                args.reload = True
                filename = os.path.join(args.root_log, args.store_name, 'ckpt.best.pth.tar')
                checkpoint = torch.load(filename, map_location='cpu')
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                best_many, best_med, best_few = checkpoint['best_many'], checkpoint['best_med'], checkpoint['best_few']
                criterion_scl = checkpoint['criterion_scl'].cuda(args.gpu)
                criterion_scl.reload_memory()

                print(f"=> reload best checkpoint '{filename}' (epoch {checkpoint['epoch']}, best_acc1 {best_acc1}, best_many {best_many}, best_med {best_med}, best_few {best_few})")
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])




        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    cudnn.benchmark = True

    if args.reload:

        print(f'===> Val dataset size: {len(val_dataset)}')

        acc1, many, med, few = validate(train_loader, val_loader, model, criterion_ce, 1, args)
        print('Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(acc1, many, med, few))

        return


    for epoch in range(args.start_epoch, args.epochs):
        print(f"Epoch {epoch}")

        # set epoch
        adjust_lr(optimizer, epoch, args)
        train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion_ce, criterion_scl, optimizer, epoch, args, tf_writer)

        # evaluate on validation set
        acc1, many, med, few = validate(train_loader, val_loader, model, criterion_ce, epoch, args, tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_many = many
            best_med = med
            best_few = few
        print('Best Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(best_acc1, best_many, best_med, best_few))

        if args.rank == 0:
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_many': best_many,
                'best_med': best_med,
                'best_few': best_few,
                'optimizer': optimizer.state_dict(),
                'criterion_scl': criterion_scl,
            }, is_best)


    print(f'===> Val dataset size: {len(val_dataset)}')

    filename = os.path.join(args.root_log, args.store_name, 'ckpt.best.pth.tar')
    checkpoint = torch.load(filename, map_location='cpu')
    args.start_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']

    print(f"=> reload best checkpoint '{filename}' (epoch {checkpoint['epoch']}, best_acc1 {best_acc1}, best_many {best_many}, best_med {best_med}, best_few {best_few})")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])



    acc1, many, med, few = validate(train_loader, val_loader, model, criterion_ce, 1, args)
    print('Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(acc1, many, med, few))


def train(train_loader, model, criterion_ce, criterion_scl, optimizer, epoch, args, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    scl_loss_all = AverageMeter('SCL_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    model.train()
    end = time.time()


    if hasattr(criterion_scl, "_hook_before_epoch"):
        criterion_scl._hook_before_epoch(epoch, args.epochs)

    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs = torch.cat([inputs[0], inputs[1], inputs[2]], dim=0)
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        batch_size = targets.shape[0]

        feat_mlp, ce_logits, _ = model(inputs)
        _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
        ce_logits, _, __ = torch.split(ce_logits, [batch_size, batch_size, batch_size], dim=0)

        contrast_logits1 = criterion_scl(f2, targets, args=args)
        contrast_logits2 = criterion_scl(f3, targets, args=args)
        contrast_logits = (contrast_logits1 + contrast_logits2)/2


        scl_loss = (criterion_ce(contrast_logits1, targets) + criterion_ce(contrast_logits2, targets))/2 
        ce_loss = criterion_ce(ce_logits, targets)

        logits = ce_logits + args.alpha * contrast_logits
        loss = ce_loss + args.alpha * scl_loss


        ce_loss_all.update(ce_loss.item(), batch_size)
        scl_loss_all.update(scl_loss.item(), batch_size)

        acc1 = accuracy(logits, targets, topk=(1,))
        top1.update(acc1[0].item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}] \t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                      'SCL_Loss {scl_loss.val:.4f} ({scl_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                ce_loss=ce_loss_all, scl_loss=scl_loss_all, top1=top1, ))  # TODO
            print(output)


 
    tf_writer.add_scalar('CE loss/train', ce_loss_all.avg, epoch)
    tf_writer.add_scalar('SCL loss/train', scl_loss_all.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)


def validate(train_loader, val_loader, model, criterion_ce, epoch, args, tf_writer=None, flag='val'):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    total_logits = torch.empty((0, args.cls_num)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            inputs, targets = data
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            _, ce_logits, _ = model(inputs)
            logits = ce_logits

            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, targets))

            batch_time.update(time.time() - end)


        total_logits_list = [torch.zeros_like(total_logits) for _ in range(args.world_size)]
        total_labels_list = [torch.zeros_like(total_labels) for _ in range(args.world_size)]

        dist.all_gather(total_logits_list, total_logits)
        dist.all_gather(total_labels_list, total_labels)

        total_logits = torch.cat(total_logits_list, dim=0)
        total_labels = torch.cat(total_labels_list, dim=0)

        ce_loss = criterion_ce(total_logits, total_labels)
        acc1 = accuracy(total_logits, total_labels, topk=(1,))

        ce_loss_all.update(ce_loss.item(), 1)
        top1.update(acc1[0].item(), 1)

        if tf_writer is not None:
            tf_writer.add_scalar('CE loss/val', ce_loss_all.avg, epoch)
            tf_writer.add_scalar('acc/val_top1', top1.avg, epoch)


        probs, preds = F.softmax(total_logits, dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(preds, total_labels, train_loader, acc_per_cls=False)
        return top1.avg, many_acc_top1*100, median_acc_top1*100, low_acc_top1*100


def save_checkpoint(args, state, is_best):
    filename = os.path.join(args.root_log, args.store_name, 'ckpt.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class TwoCropTransform:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x), self.transform2(x)]


def adjust_lr(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1) / (args.epochs - args.warmup_epochs + 1)))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
