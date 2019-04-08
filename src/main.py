## This project borrows a lot from [fyu/drn](https://github.com/fyu/drn)
## notably in `main.py` and `data_transforms.py`

import argparse
import logging
import math
import os
from os.path import exists, join, dirname
import time
import shutil
from time import localtime
from pdb import set_trace as db

from PIL import Image
import numpy as np

import torch
import torch.utils.data
from torch import nn
import torch.backends.cudnn as cudnn

import data_transforms as transforms

from utils import ComLoss, AverageMeter, accuracy, read_config

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
formatter = logging.Formatter(fmt=FORMAT)
logger_s = logging.getLogger('screen')
logger_s.setLevel(logging.INFO)
logger_f = logging.getLogger('file')
logger_f.setLevel(logging.DEBUG)

from unet_parts import *

class QGCNN(nn.Module):
    def __init__(self):
        super(QGCNN, self).__init__()
        
        self.conv_in = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 64)
        self.conv_out = outconv(64, 3)

        self.activate = nn.Sigmoid()

        self._initialize_weights()
        
    def forward(self, x):
        self.residual = x
        x1 = self.conv_in(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.conv_out(x)
        x = self.activate(self.residual - x)
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            else:
                pass


class DataList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.read_lists()

    def __getitem__(self, index):

        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        if self.label_list is not None:
            data.append(Image.open(
                join(self.data_dir, self.label_list[index])))
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)

        # if self.phase == 'train':
        #     # More batches per epoch
        #     self.image_list *= 128
        #     self.label_list *= 128

        # It takes too long to go through all val imgs
        if self.phase == 'val':
            self.image_list = self.image_list[:32]
            self.label_list = self.label_list[:32]


def validate(val_loader, model, criterion, eval_score=None, print_freq=16):
    batch_time = AverageMeter()
    losses = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()

    # Switch to evaluate mode
    model.eval()
    criterion.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.float()
            input = input.cuda().unsqueeze(0)
            target = target.cuda().unsqueeze(0)

            # Compute output
            output = model(input)
            loss = criterion(output, target)

            # Record loss and measure accuracy
            losses.update(loss.data, input.size(0))

            if eval_score is not None:
                psnr.update(eval_score(output, target, 'PSNR'), input.size(0))
                ssim.update(eval_score(output, target, 'SSIM'), input.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            log = 'Test: [{0}/{1}]\t'\
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                    'PSNR {psnr.val:.3f} ({psnr.avg:.3f})\t'\
                    'SSIM {ssim.val:.3f} ({ssim.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    psnr=psnr, ssim=ssim)

            logger_f.info(log)
            if i % print_freq == 0:
                logger_s.info(log)

    logger_s.info(' * PSNR={psnr.avg:.3f}\tSSIM={ssim.avg:.3f}'.format(psnr=psnr, ssim=ssim))

    return ssim.avg


def test(eval_data_loader, model, output_dir='pred', save_vis=True, suffix=''):
    model.eval()
    psnr = AverageMeter()
    ssim = AverageMeter()
    with torch.no_grad():
        for iter, (image, label, name) in enumerate(eval_data_loader):
            final = model(image)

            psnr_dis = accuracy(image, label, 'PSNR')
            psnr_gen = accuracy(final, label, 'PSNR')
            ssim_dis = accuracy(image, label, 'SSIM')
            ssim_gen = accuracy(final, label, 'SSIM')

            psnr.update(psnr_gen, 1)
            ssim.update(ssim_gen, 1)

            pred = transforms.to_numpy(final)

            if save_vis:
                save_output_images(pred, name[0], output_dir, suffix)

            log = 'Eval: [{0}/{1}]\t'\
                    'psnr_dis: {psnr_dis:.6f}\t'\
                    'psnr_gen: {psnr_gen:.6f}({psnr_avg:.6f})\t'\
                    'ssim_dis: {ssim_dis:.6f}\t'\
                    'ssim_gen: {ssim_gen:.6f}({ssim_avg:.6f})'\
                    .format(iter, len(eval_data_loader), 
                    psnr_dis=psnr_dis, psnr_gen=psnr_gen, ssim_dis=ssim_dis, ssim_gen=ssim_gen, 
                    psnr_avg=psnr.avg, ssim_avg=ssim.avg)

            logger_f.info(log)
            logger_s.info(log)


def train(train_loader, model, criterion, optimizer, epoch,
          eval_score=None, print_freq=32):
    losses = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()
    feat_loss = AverageMeter()
    pixel_loss = AverageMeter()

    # Switch to train mode
    model.train()
    criterion.train()

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        # Compute output
        output = model(input)
        loss, ploss, floss = criterion(output, target)

        # Record losses
        losses.update(loss.data, input.size(0))
        pixel_loss.update(ploss.data, input.size(0))
        feat_loss.update(floss.data, input.size(0))

        # Measure accuracy
        if eval_score is not None:
            psnr.update(eval_score(output.data, target.data, 'PSNR'), input.size(0))
            ssim.update(eval_score(output.data, target.data, 'SSIM'), input.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log =   'Epoch: [{0}][{1}/{2}]\t'\
                'Loss {tl.val:.4f} ({tl.avg:.4f})\t'\
                'Pixel {pl.val:.4f} ({pl.avg:.4f})\t'\
                'Feature {fl.val:.4f} ({fl.avg:.4f})\t'\
                'PSNR {psnr.val:.3f} ({psnr.avg:.3f})\t'\
                'SSIM {ssim.val:.3f} ({ssim.avg:.3f})'.format(
            epoch, i, len(train_loader), tl=losses, psnr=psnr, ssim=ssim, 
            fl=feat_loss, pl=pixel_loss)

        logger_f.info(log)
        if i % print_freq == 0:
            logger_s.info(log)

def train_cnn(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = cfg['CROP_SIZE']

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = QGCNN()
    model = torch.nn.DataParallel(single_model)

    if cfg['FEATS']:
        feat_names, weights = zip(*(tuple(*f.items()) for f in cfg['FEATS']))
    else:
        feat_names, weights = None, None
        
    criterion = ComLoss(cfg['IQA_MODEL'], weights, feat_names, patch_size=cfg['PATCH_SIZE'], pixel_criterion=cfg['CRITERION'])
    criterion.cuda()

    # Data loading
    data_dir = cfg['DATA_DIR']
    list_dir = cfg['LIST_DIR']
    t = [transforms.RandomCrop(crop_size),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor()]
    # Note that the cropsize could have a significant influence, 
    # i.e., with a small cropsize the model would get overfitted 
    # easily thus hard to train
    train_loader = torch.utils.data.DataLoader(
        DataList(data_dir, 'train', transforms.Compose(t),
                list_dir=list_dir),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    # The cropsize of the validation set dramatically affects the 
    # evaluation accuracy, which means the quality of the whole 
    # image might be very different from that of its cropped patches.
    #
    # Try setting batch_size = 1 and no crop (disable RandomCrop)
    # to improve the effect of early stopping. 
    val_loader = DataList(data_dir, 'val', transforms.Compose([
            transforms.ToTensor()]), list_dir=list_dir)

    optimizer = torch.optim.Adam(single_model.parameters(), 
                                lr=args.lr, 
                                betas=(0.9, 0.99), 
                                weight_decay=args.weight_decay)
    
    cudnn.benchmark = True

    weight_dir = join(out_dir, 'weights/')
    if not exists(weight_dir):
        os.mkdir(weight_dir)

    best_prec = 0
    start_epoch = 0

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger_s.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            logger_s.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger_f.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger_f.warning("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model.cuda(), criterion, eval_score=accuracy)
        return

    for epoch in range(start_epoch, args.epochs):

        lr = adjust_learning_rate(args, optimizer, epoch)

        if criterion.weights is not None and (epoch+1) % 100 == 0:
            criterion.weights /= 10.0

        logger_s.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        train(train_loader, model.cuda(), criterion, optimizer, epoch,
              eval_score=accuracy)
             
            
        # Evaluate on validation set
        prec = validate(val_loader, model.cuda(), criterion, eval_score=accuracy)

        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        logger_s.info('current best {:.6f}'.format(best_prec))
        
        checkpoint_path = join(weight_dir, 'checkpoint_latest.pkl')
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
        }, is_best, filename=checkpoint_path)

        if (epoch + 1) % args.store_interval == 0:
            history_path = join(weight_dir, 'checkpoint_{:03d}.pkl'.format(epoch + 1))
            shutil.copyfile(checkpoint_path, history_path)


def test_cnn(args):
    batch_size = 1
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = QGCNN()
    model = torch.nn.DataParallel(single_model).cuda()

    dataset = DataList(cfg['DATA_DIR'], phase, transforms.Compose([
        transforms.ToTensor()
    ]), list_dir=cfg['LIST_DIR'], out_name=True)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # Optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger_s.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            logger_s.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger_f.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger_f.warning("=> no checkpoint found at '{}'".format(args.resume))

    test_dir = join(out_dir, 'test/result_{:03d}_{}'.format(start_epoch, phase))

    test(test_loader, model, output_dir=test_dir, suffix=cfg['TEST_SUFFIX'])


def adjust_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate 
    """
    if args.lr_mode == 'step':
        lr = args.lr * (0.5 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 1.1
    elif args.lr_mode == 'same':
        lr = args.lr
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_output_images(prediction, filename, output_dir, suffix=''):
    im = Image.fromarray(prediction.astype(np.uint8))
    # Force to store in bmp files
    sp = filename.split('.')
    name, ext = '.'.join(sp[:-1]), sp[-1]
    fn = os.path.join(output_dir, '{}_{}.{}'.format(name, suffix, ext))
    out_dir = dirname(fn)
    if not exists(out_dir):
        os.makedirs(out_dir)
    im.save(fn)

def save_checkpoint(state, is_best, filename='checkpoint.pkl'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(dirname(filename), 'model_best.pkl'))


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, 
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--phase', default='test')
    parser.add_argument('--exp-config', default='config.yaml', type=str, help='configuration file for experiment settings')
    parser.add_argument('--store-interval', type=int, default=50)
    args = parser.parse_args()

    return args


def main():
    global cfg, out_dir
    args = parse_args()
    cfg = read_config(args.exp_config)   
    
    for k, v in cfg.items(): print(k.lower(),':',v)

    out_dir = join(cfg['OUT_DIR'], cfg['TEST_SUFFIX'])
    if not exists(out_dir):
        os.makedirs(out_dir)

    log_dir = join(out_dir, 'logs')
    if not exists(log_dir):
        os.mkdir(log_dir)

    # Set loggers
    scrn_handler = logging.StreamHandler()
    scrn_handler.setFormatter(formatter) 
    logger_s.addHandler(scrn_handler)

    file_handler = logging.FileHandler(filename=join(log_dir, 
                                        '{}_{}_{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}.{}'
                                        .format(cfg['TEST_SUFFIX'], args.cmd, 
                                        *localtime()[:6], 'log')))
    file_handler.setFormatter(formatter)
    logger_f.addHandler(file_handler)


    if args.cmd == 'train':
        train_cnn(args)
    elif args.cmd == 'test':
        test_cnn(args)


if __name__ == '__main__':
    main()
