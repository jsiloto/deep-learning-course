from __future__ import print_function, absolute_import
import torch

__all__ = ['top1pred', 'accuracy']

def top1pred(output):
    _, pred = output.topk(1, 1, True, True)
    return pred.t()[0]
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0))
        return res
