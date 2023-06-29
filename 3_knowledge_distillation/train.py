'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import sys


import argparse
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.eval_classifier import validate
from src.utils import Bar, Logger, AverageMeter, accuracy, savefig, top1pred
from src.boilerplate import get_dataset, get_model, resume_model, LRAdjust

parser = argparse.ArgumentParser(description='Deep Learning Course')
parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('-sc', '--teacher-checkpoint', default='checkpoints/vanilla/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('-c', '--checkpoint', default='checkpoints/distill/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_classifier(args):
    train_loader, val_loader, train_loader_len, val_loader_len = get_dataset(args.batch_size, unlabeled=True)
    student = get_model(num_classes=10).to(device)
    teacher = get_model(num_classes=10).to(device)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    adjuster = LRAdjust(lr=0.001, warmup=5, epochs=args.epochs)

    # optionally resume from a checkpoint
    checkpoint_path = args.checkpoint
    student, start_epoch, best_prec1, best_prec1classes = resume_model(student, args.checkpoint, optimizer, best=False)
    teacher, _ , _ = resume_model(teacher, args.teacher_checkpoint, optimizer, best=False)
    resume = (start_epoch != 0)
    logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title="title", resume=resume)
    logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    ########################################################################################

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))
        train_loss, train_acc = train(train_loader, train_loader_len, student, teacher, criterion, optimizer, adjuster, epoch)
        val_loss, prec1, top1classes = validate(val_loader, val_loader_len, student, criterion)
        lr = optimizer.param_groups[0]['lr']

        # append logger file
        logger.append([epoch, lr, train_loss, val_loss, train_acc, prec1])

        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
            best_prec1classes = top1classes
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': student.state_dict(),
            'best_prec1': best_prec1,
            'best_prec1classes': best_prec1classes,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=checkpoint_path)

    logger.close()
    logger.plot()
    savefig(os.path.join(checkpoint_path, 'log.eps'))

    model, epoch, best_prec1 = resume_model(student, checkpoint_path, optimizer, best=True)
    validate(train_loader, train_loader_len, model, criterion, title='Train Set')
    _, prec1, top1classes = validate(val_loader, val_loader_len, model, criterion, title='Val Set')
    print('Best accuracy:')
    print(prec1)
    print("Classes Accuracy")
    print(top1classes)



def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def train(train_loader, train_loader_len, student, teacher, criterion, optimizer, adjuster, epoch):
    bar = Bar('Train', max=train_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    student.train()
    teacher.eval()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        adjuster.adjust(optimizer, epoch, i, train_loader_len)

        # measure data loading time
        data_time.update(time.time() - end)
        with torch.no_grad():
            target = teacher(input.to(device))
            target = top1pred(target)

        # compute output
        output = student(input.to(device))
        loss = criterion(output, target).to(device)
        # measure accuracy and record loss

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=i + 1,
            size=train_loader_len,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def main():
    args = parser.parse_args()
    train_classifier(args)


if __name__ == '__main__':
    main()
