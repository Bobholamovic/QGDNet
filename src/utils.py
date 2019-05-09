import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from pdb import set_trace as db

import yaml
from skimage.measure import compare_psnr, compare_ssim

from iqa_net import IQANet
from data_transforms import to_numpy

# Metrics
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, measure='PSNR'):
    if measure == 'PSNR':
        metric = compare_psnr
    elif measure == 'SSIM':
        from functools import partial
        metric = partial(compare_ssim, multichannel=True)
    else:
        raise ValueError('invalid metric type')
    batch = output.size(0)
    score = 0.0
    for b in range(batch):
        out, tar = output[b], target[b]
        out_np, tar_np = to_numpy(out), to_numpy(tar)
        score += metric(out_np, tar_np)
    return score / batch
    

# Losses
class IQALoss(nn.Module):
    def __init__(self, path_to_model_weight, patch_size, feat_names):
        super(IQALoss, self).__init__()

        self.iqa_model = IQANet(weighted=False)
        self.iqa_model.load_state_dict(torch.load(path_to_model_weight)['state_dict'])

        for p in self.iqa_model.parameters():
            p.requires_grad = False

        self.patch_size = patch_size
        self.feat_names = feat_names

    def forward(self, output, target):
        output_patches = self._extract_patches(output)
        target_patches = self._extract_patches(target)

        self.iqa_model.eval()
        score, features = self.iqa_model(output_patches, target_patches)
        
        sel_feats = [v for k,v in features.items() if k in self.feat_names]

        for i, f in enumerate(sel_feats):
            if isinstance(f, tuple):
                assert len(f) == 2
                sel_feats[i] = F.mse_loss(*f)
            else:
                sel_feats[i] = torch.mean(torch.abs(f))

        return torch.stack(sel_feats, dim=0)
        
    def _extract_patches(self, img):
        h, w = img.shape[-2:]
        nh, nw = h//self.patch_size, w//self.patch_size
        ch, cw = nh*self.patch_size, nw*self.patch_size
        bh, bw = (h-ch)//2, (w-cw)//2

        vpatchs = torch.stack(torch.split(img[...,bh:bh+ch,:], self.patch_size, dim=-2), dim=1)
        patchs = torch.cat(torch.split(vpatchs[...,bw:bw+cw], self.patch_size, dim=-1), dim=1)

        return patchs

    def _renormalize(self, img):
        # The output range of QGCNN exactly fits the input of IQANet 
        # thus no remapping is done here
        pass


class ComLoss(nn.Module):
    def __init__(self, model_path, weights, feat_names, patch_size=32, pixel_criterion='MAE'):
        super(ComLoss, self).__init__()

        if pixel_criterion == 'MAE':
            self.criterion = F.l1_loss
        elif pixel_criterion == 'MSE':
            self.criterion = F.mse_loss
        elif hasattr(pixel_criterion, '__call__'):
            self.criterion = pixel_criterion 
        else:
            raise ValueError('invalid criterion')
            
        self.weights = weights
        if self.weights is not None:
            assert len(weights) == len(feat_names)
            self.weights = torch.FloatTensor(weights)
            if torch.cuda.is_available(): self.weights = self.weights.cuda()
            self.iqa_loss = IQALoss(model_path, patch_size, feat_names)

    def forward(self, output, target):
        pixel_loss = self.criterion(output, target)
        if self.weights is not None and self.training:
            feat_loss = torch.sum(self.weights*self.iqa_loss(output, target))
        else:
            feat_loss = torch.tensor(0.0).type_as(pixel_loss)

        total_loss = pixel_loss + feat_loss

        if self.training:
            return total_loss, pixel_loss, feat_loss
        else:
            return total_loss

# 
def read_config(config_path):
    f = open(config_path, 'r')
    cfg = yaml.load(f.read())
    f.close()
    return cfg

