import torch

import builtins
import time
import shutil
from torchvision.transforms import transforms
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from loss.logitadjust import LogitAdjust
from loss.proco import ProCoLoss
import math
from tensorboardX import SummaryWriter
from dataset.inat import INaturalist
from dataset.imagenet import ImageNetLT
from models import resnext
import warnings
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import random
from randaugment import rand_augment_transform
from utils import shot_acc, bool_flag
import argparse
import os
import logging

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='imagenet', choices=['inat', 'imagenet'])
parser.add_argument('--data', default='', metavar='DIR')
parser.add_argument('--arch', default='resnext50', choices=['resnet50', 'resnext50'])
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--temp', default=0.07, type=float, help='scalar temperature for contrastive learning')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32*8), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[160, 180], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=200, type=int,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--randaug', default=True, type=bool, help='use RandAugmentation for classification branch')
parser.add_argument('--cl_views', default='sim-sim', type=str, choices=['sim-sim', 'sim-rand', 'rand-rand'],
                    help='Augmentation strategy for contrastive learning views')
parser.add_argument('--feat_dim', default=1024, type=int, help='feature dimension of mlp head')
parser.add_argument('--warmup_epochs', default=0, type=int,
                    help='warmup epochs')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--cos', default=True, type=bool,
                    help='lr decays by cosine scheduler. ')
parser.add_argument('--use_norm', default=True, type=bool_flag,
                    help='cosine classifier.')
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--reload', default=False, type=bool, help='load supervised model')
parser.add_argument('--mark', type=str)

parser.add_argument('--debug', default=False, type=bool_flag)
parser.add_argument('--test', default=None, type=str)


def main():
    args = parser.parse_args()

    args.store_name = args.mark


    log_name = os.path.join(args.root_log, args.store_name, 'log')
    os.makedirs(os.path.join(args.root_log, args.store_name), exist_ok=True)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_name)
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)



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



    if args.rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

        #suppress logger.info
        def info_pass(*args, **kwargs):
            pass
        logger.info = info_pass




    logger.info(args)

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))





    dist.init_process_group(backend='nccl', init_method='env://',
            world_size=args.world_size, rank=args.rank)




    # create model
    logger.info("=> creating model '{}'".format(args.arch))

    if args.dataset == 'inat':
        num_classes = 8142
    elif args.dataset == 'imagenet':
        num_classes = 1000
    else:
        raise ValueError('Unknown dataset')


    if args.arch == 'resnet50':
        model = resnext.Model(name='resnet50', num_classes=num_classes, feat_dim=args.feat_dim,
                                 use_norm=args.use_norm)
    elif args.arch == 'resnext50':
        model = resnext.Model(name='resnext50', num_classes=num_classes, feat_dim=args.feat_dim,
                                 use_norm=args.use_norm)
    else:
        raise NotImplementedError('This model is not supported')


    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])


    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    # prepare data

    txt_train = f'dataset/ImageNet_LT/ImageNet_LT_train.txt' if args.dataset == 'imagenet' \
        else f'dataset/iNaturalist18/iNaturalist18_train.txt'
    txt_val = f'dataset/ImageNet_LT/ImageNet_LT_val.txt' if args.dataset == 'imagenet' \
        else f'dataset/iNaturalist18/iNaturalist18_val.txt'

    normalize = transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192)) if args.dataset == 'inat' \
        else transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_randnclsstack = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ]
    if args.cl_views == 'sim-sim':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim),
                           transforms.Compose(augmentation_sim), ]
    elif args.cl_views == 'sim-rand':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack),
                           transforms.Compose(augmentation_sim), ]
    elif args.cl_views == 'rand-rand':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack),
                           transforms.Compose(augmentation_randnclsstack), ]
    else:
        raise NotImplementedError("This augmentations strategy is not available for contrastive learning branch!")
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    val_dataset = INaturalist(
        root=args.data,
        txt=txt_val,
        transform=val_transform, train=False,
    ) if args.dataset == 'inat' else ImageNetLT(
        root=args.data,
        txt=txt_val,
        transform=val_transform, train=False)

    train_dataset = INaturalist(
        root=args.data,
        txt=txt_train,
        transform=transform_train
    ) if args.dataset == 'inat' else ImageNetLT(
        root=args.data,
        txt=txt_train,
        transform=transform_train)

    logger.info(f'===> Training data length {len(train_dataset)}')
    logger.info(f'===> Validation data length {len(val_dataset)}')

    cls_num_list = train_dataset.cls_num_list
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

    if args.test:
        if os.path.isfile(args.test):
            logger.info("=> loading checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_many, best_med, best_few = checkpoint['best_many'], checkpoint['best_med'], checkpoint['best_few']


            criterion_scl = checkpoint['criterion_scl'].cuda(args.gpu)
            criterion_scl.reload_memory()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            logger.info("=> loaded checkpoint '{}' (epoch {}, best_acc1 {})"
                  .format(args.test, checkpoint['epoch'], best_acc1))
            args.reload = True






        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
            raise FileNotFoundError
    else:
        filename = os.path.join(args.root_log, args.store_name, 'ckpt.pth.tar')
        if os.path.exists(filename):
            args.resume = filename

        # optionally resume from a checkpoint
        # default is to resume from latest
        if args.resume:
            if os.path.isfile(args.resume):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location='cpu')
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                best_many, best_med, best_few = checkpoint['best_many'], checkpoint['best_med'], checkpoint['best_few']
                criterion_scl = checkpoint['criterion_scl'].cuda(args.gpu)
                criterion_scl.reload_memory()
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])

                logger.info("=> loaded checkpoint '{}' (epoch {}, best_acc1 {})"
                      .format(args.resume, checkpoint['epoch'], best_acc1))

                if args.start_epoch == args.epochs:
                    logger.info("=> already trained for {} epochs, exiting".format(args.epochs))
                    args.reload = True
                    filename = os.path.join(args.root_log, args.store_name, 'ckpt.best.pth.tar')
                    checkpoint = torch.load(filename, map_location='cpu')
                    args.start_epoch = checkpoint['epoch']
                    best_acc1 = checkpoint['best_acc1']
                    best_many, best_med, best_few = checkpoint['best_many'], checkpoint['best_med'], checkpoint['best_few']
                    criterion_scl = checkpoint['criterion_scl'].cuda(args.gpu)
                    criterion_scl.reload_memory()

                    logger.info(f"=> reload best checkpoint '{filename}' (epoch {checkpoint['epoch']}, best_acc1 {best_acc1}, best_many {best_many}, best_med {best_med}, best_few {best_few})")
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])





            else:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))


    cudnn.benchmark = True

    if args.reload:
        txt_test = f'dataset/ImageNet_LT/ImageNet_LT_test.txt' if args.dataset == 'imagenet' \
            else f'dataset/iNaturalist18/iNaturalist18_val.txt'
        test_dataset = INaturalist(
            root=args.data,
            txt=txt_test,
            transform=val_transform, train=False
        ) if args.dataset == 'inat' else ImageNetLT(
            root=args.data,
            txt=txt_test,
            transform=val_transform, train=False)

        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=test_sampler)



        acc1, many, med, few = validate(train_loader, test_loader, model, criterion_ce, 1, args)
        logger.info('Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(acc1, many, med, few))


        return




    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f"Epoch {epoch}")

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
        logger.info('Best Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(best_acc1, best_many, best_med, best_few))


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


    txt_test = f'dataset/ImageNet_LT/ImageNet_LT_test.txt' if args.dataset == 'imagenet' \
        else f'dataset/iNaturalist18/iNaturalist18_val.txt'
    test_dataset = INaturalist(
        root=args.data,
        txt=txt_test,
        transform=val_transform, train=False
    ) if args.dataset == 'inat' else ImageNetLT(
        root=args.data,
        txt=txt_test,
        transform=val_transform, train=False)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    filename = os.path.join(args.root_log, args.store_name, 'ckpt.best.pth.tar')
    checkpoint = torch.load(filename, map_location='cpu')
    args.start_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']

    logger.info(f"=> reload best checkpoint '{filename}' (epoch {checkpoint['epoch']}, best_acc1 {best_acc1}, best_many {best_many}, best_med {best_med}, best_few {best_few})")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])



    acc1, many, med, few = validate(train_loader, test_loader, model, criterion_ce, 1, args)
    logger.info('Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(acc1, many, med, few))



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

        loss.backward()

        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()


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
            logger.info(output)
        if args.debug:
            if i >= 50:
                break

 
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


        _, preds = F.softmax(total_logits, dim=1).max(dim=1)
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
