# import argparse
# import os
# import random
# import shutil
# import time
# import warnings
# import numpy as np
# import pprint
# import math

# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.distributed as dist
# import torch.optim
# import torch.multiprocessing as mp
# import torch.utils.data
# import torch.utils.data.distributed
# import torch.nn.functional as F

# from datasets.cifar10 import CIFAR10_LT
# from datasets.cifar100 import CIFAR100_LT
# from datasets.places import Places_LT
# from datasets.imagenet import ImageNet_LT
# from datasets.ina2018 import iNa2018

# from models import resnet
# from models import resnet_places
# from models import resnet_cifar

# from utils import config, update_config, create_logger
# from utils import AverageMeter, ProgressMeter
# from utils import accuracy, calibration

# from methods import mixup_data, mixup_criterion
# from methods import LabelAwareSmoothing, LearnableWeightScaling


# class ARB_Loss():
#     def __init__(self, reduction='mean'):
#         self.reduction = reduction

#     def __call__(self, outputs: torch.Tensor, labels: torch.Tensor, class_cnt: torch.Tensor) -> torch.Tensor:
#         print('using arb loss', outputs.size(), labels.size())
#         outputs = torch.clamp(outputs, min=-30., max=30.)
#         exp = torch.exp(outputs)
#         num_classes = outputs.size(1)
#         # unique_labels, cnt = torch.unique(labels, return_counts=True)
#         # class_cnt = torch.zeros(outputs.size(1), dtype=outputs.dtype).to(outputs.device)
#         # # batch_cnt = torch.zeros(labels.size(1), dtype=outputs.dtype).to(outputs.device)
#         # class_cnt[unique_labels] = cnt.float()
#         batch_cnt = class_cnt[labels]
#         w = batch_cnt.reshape(-1, 1).repeat(1, num_classes)  # B, 100
#         w = class_cnt.reshape(1, -1) / w
#         _sum = torch.sum(exp * w, dim=1)
#         softmax = exp.gather(1, labels.unsqueeze(-1)).squeeze() / _sum
#         loss = - torch.log(softmax)
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             raise NotImplementedError

# class ARB_Loss_FL():
#     def __init__(self, reduction='mean', alpha=1, gamma=2.):
#         self.reduction = reduction
#         self.alpha = alpha
#         self.gamma = gamma

#     def __call__(self, outputs: torch.Tensor, labels: torch.Tensor, class_cnt: torch.Tensor) -> torch.Tensor:
#         print('using arb loss', outputs.size(), labels.size())
#         outputs = torch.clamp(outputs, min=-30., max=30.)
#         exp = torch.exp(outputs)
#         num_classes = outputs.size(1)
#         # unique_labels, cnt = torch.unique(labels, return_counts=True)
#         # class_cnt = torch.zeros(outputs.size(1), dtype=outputs.dtype).to(outputs.device)
#         # # batch_cnt = torch.zeros(labels.size(1), dtype=outputs.dtype).to(outputs.device)
#         # class_cnt[unique_labels] = cnt.float()
#         batch_cnt = class_cnt[labels]
#         w = batch_cnt.reshape(-1, 1).repeat(1, num_classes)  # B, 100
#         w = class_cnt.reshape(1, -1) / w
#         _sum = torch.sum(exp * w, dim=1)
#         p = exp.gather(1, labels.unsqueeze(-1)).squeeze() / _sum
#         loss = - self.alpha * torch.pow((1 - p), self.gamma) * torch.log(p)
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             raise NotImplementedError

# class WARB_Loss():
#     def __init__(self, reduction='mean', ugcc=False):
#         self.reduction = reduction
#         self.ugcc = ugcc

#     def __call__(self, outputs: torch.Tensor, labels: torch.Tensor, class_cnt: torch.Tensor) -> torch.Tensor:
#         outputs = torch.clamp(outputs, min=-30., max=30.)
#         exp = torch.exp(outputs)
#         num_classes = outputs.size(1)
#         # unique_labels, cnt = torch.unique(labels, return_counts=True)
#         # class_cnt = torch.zeros(outputs.size(1), dtype=outputs.dtype).to(outputs.device)
#         # # batch_cnt = torch.zeros(labels.size(1), dtype=outputs.dtype).to(outputs.device)
#         # class_cnt[unique_labels] = cnt.float()
#         batch_cnt = class_cnt[labels]
#         w = batch_cnt.reshape(-1, 1).repeat(1, num_classes)  # B, 100
#         w = class_cnt.reshape(1, -1) / w
#         _sum = torch.sum(exp * w, dim=1)
#         softmax = exp.gather(1, labels.unsqueeze(-1)).squeeze() / _sum

#         if not self.ugcc:
#             num_classes_in_batch = (class_cnt > 0).sum()
#             mean_num_class = outputs.size(0) / num_classes_in_batch
#             pre_w = mean_num_class / batch_cnt
#         else:
#             mean_num_class = class_cnt.sum() / outputs.size(1)
#             pre_w = mean_num_class / batch_cnt

#         print('using arb loss', pre_w.sum())

#         loss = - pre_w * torch.log(softmax)
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             raise NotImplementedError


# def parse_args():
#     parser = argparse.ArgumentParser(description='MiSLAS training (Stage-2)')
#     parser.add_argument('--cfg',
#                         help='experiment configure file name',
#                         required=True,
#                         type=str)
#     parser.add_argument('opts',
#                         help="Modify config options using the command-line",
#                         default=None,
#                         nargs=argparse.REMAINDER)
#     parser.add_argument('--local_rank',
#                         type=int, default=0)
#     args = parser.parse_args()
#     update_config(config, args)

#     return args


# best_acc1 = 0
# its_ece = 100


# def main():

#     args = parse_args()
#     logger, model_dir = create_logger(config, args.cfg)
#     logger.info('\n' + pprint.pformat(args))
#     logger.info('\n' + str(config))

#     if config.deterministic:
#         seed = 0
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         random.seed(seed)
#         np.random.seed(seed)
#         os.environ['PYTHONHASHSEED'] = str(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)

#     if config.gpu is not None:
#         warnings.warn('You have chosen a specific GPU. This will completely '
#                       'disable data parallelism.')

#     if config.dist_url == "env://" and config.world_size == -1:
#         config.world_size = int(os.environ["WORLD_SIZE"])

#     os.environ["RANK"] = str(args.local_rank)

#     config.distributed = config.world_size > 1 or config.multiprocessing_distributed

#     ngpus_per_node = torch.cuda.device_count()
#     if config.multiprocessing_distributed:
#         raise NotImplementedError
#         # Since we have ngpus_per_node processes per node, the total world_size
#         # needs to be adjusted accordingly
#         config.world_size = ngpus_per_node * config.world_size
#         # Use torch.multiprocessing.spawn to launch distributed processes: the
#         # main_worker process function
#         mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, logger))
#     else:
#         # Simply call main_worker function
#         main_worker(config.gpu, ngpus_per_node, config, logger, model_dir)


# def main_worker(gpu, ngpus_per_node, config, logger, model_dir):
#     global best_acc1, its_ece
#     config.gpu = gpu
# #     start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

#     if config.gpu is not None:
#         logger.info("Use GPU: {} for training".format(config.gpu))

#     if config.distributed:
#         if config.dist_url == "env://" and config.rank == -1:
#             config.rank = int(os.environ["RANK"])
#         if config.multiprocessing_distributed:
#             # For multiprocessing distributed training, rank needs to be the
#             # global rank among all the processes
#             raise NotImplementedError
#             config.rank = config.rank * ngpus_per_node + gpu
#         print(config.dist_backend, config.dist_url, config.world_size, config.rank)
#         dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
#                                 world_size=config.world_size, rank=config.rank)

#         config.gpu = config.rank

#     if config.dataset == 'cifar10' or config.dataset == 'cifar100':
#         model = getattr(resnet_cifar, config.backbone)()
#         classifier = getattr(resnet_cifar, 'Classifier')(feat_in=64, num_classes=config.num_classes)

#     elif config.dataset == 'imagenet' or config.dataset == 'ina2018':
#         model = getattr(resnet, config.backbone)()
#         classifier = getattr(resnet, 'Classifier')(feat_in=2048, num_classes=config.num_classes)

#     elif config.dataset == 'places':
#         model = getattr(resnet_places, config.backbone)(pretrained=True)
#         classifier = getattr(resnet_places, 'Classifier')(feat_in=2048, num_classes=config.num_classes)
#         block = getattr(resnet_places, 'Bottleneck')(2048, 512, groups=1, base_width=64,
#                                                      dilation=1, norm_layer=nn.BatchNorm2d)

#     lws_model = LearnableWeightScaling(num_classes=config.num_classes)

#     if not torch.cuda.is_available():
#         logger.info('using CPU, this will be slow')
#     elif config.distributed:
#         # For multiprocessing distributed, DistributedDataParallel constructor
#         # should always set the single device scope, otherwise,
#         # DistributedDataParallel will use all available devices.
#         if config.gpu is not None:
#             torch.cuda.set_device(f'cuda:{config.gpu}')
#             model.cuda(config.gpu)
#             classifier.cuda(config.gpu)
#             lws_model.cuda(config.gpu)
#             # When using a single GPU per process and per
#             # DistributedDataParallel, we need to divide the batch size
#             # ourselves based on the total number of GPUs we have
#             config.batch_size = int(config.batch_size / ngpus_per_node)
#             config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
#             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
#             classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[config.gpu])
#             lws_model = torch.nn.parallel.DistributedDataParallel(lws_model, device_ids=[config.gpu])

#             if config.dataset == 'places':
#                 block.cuda(config.gpu)
#                 block = torch.nn.parallel.DistributedDataParallel(block, device_ids=[config.gpu])
#         else:
#             model.cuda()
#             classifier.cuda()
#             lws_model.cuda()
#             # DistributedDataParallel will divide and allocate batch_size to all
#             # available GPUs if device_ids are not set
#             model = torch.nn.parallel.DistributedDataParallel(model)
#             classifier = torch.nn.parallel.DistributedDataParallel(classifier)
#             lws_model = torch.nn.parallel.DistributedDataParallel(lws_model)
#             if config.dataset == 'places':
#                 block.cuda()
#                 block = torch.nn.parallel.DistributedDataParallel(block)

#     elif config.gpu is not None:
#         torch.cuda.set_device(f'cuda:{config.gpu}')
#         model = model.cuda(config.gpu)
#         classifier = classifier.cuda(config.gpu)
#         lws_model = lws_model.cuda(config.gpu)
#         if config.dataset == 'places':
#             block.cuda(config.gpu)
#     else:
#         # DataParallel will divide and allocate batch_size to all available GPUs
#         model = torch.nn.DataParallel(model).cuda()
#         classifier = torch.nn.DataParallel(classifier).cuda()
#         lws_model = torch.nn.DataParallel(lws_model).cuda()
#         if config.dataset == 'places':
#             block = torch.nn.DataParallel(block).cuda()

#     # optionally resume from a checkpoint
#     if config.resume:
#         if os.path.isfile(config.resume):
#             logger.info("=> loading checkpoint '{}'".format(config.resume))
#             if config.gpu is None:
#                 checkpoint = torch.load(config.resume)
#             else:
#                 # Map model to be loaded to specified single gpu.
#                 loc = 'cuda:{}'.format(config.gpu)
#                 checkpoint = torch.load(config.resume, map_location=loc)
#             # config.start_epoch = checkpoint['epoch']
#             best_acc1 = checkpoint['best_acc1']
#             its_ece = checkpoint['its_ece']
#             if config.gpu is not None:
#                 # best_acc1 may be from a checkpoint from a different GPU
#                 best_acc1 = best_acc1.to(config.gpu)
#             model.load_state_dict(checkpoint['state_dict_model'])
#             classifier.load_state_dict(checkpoint['state_dict_classifier'])
#             if config.dataset == 'places':
#                 block.load_state_dict(checkpoint['state_dict_block'])
#             logger.info("=> loaded checkpoint '{}' (epoch {})"
#                         .format(config.resume, checkpoint['epoch']))
#         else:
#             logger.info("=> no checkpoint found at '{}'".format(config.resume))

#     # Data loading code
#     if config.dataset == 'cifar10':
#         dataset = CIFAR10_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
#                              batch_size=config.batch_size, num_works=config.workers)

#     elif config.dataset == 'cifar100':
#         dataset = CIFAR100_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
#                               batch_size=config.batch_size, num_works=config.workers)

#     elif config.dataset == 'places':
#         dataset = Places_LT(config.distributed, root=config.data_path,
#                             batch_size=config.batch_size, num_works=config.workers)

#     elif config.dataset == 'imagenet':
#         dataset = ImageNet_LT(config.distributed, root=config.data_path,
#                               batch_size=config.batch_size, num_works=config.workers)

#     elif config.dataset == 'ina2018':
#         dataset = iNa2018(config.distributed, root=config.data_path,
#                           batch_size=config.batch_size, num_works=config.workers)

#     train_loader = dataset.train_balance if not config.train_stage2_instance else dataset.train_instance
#     val_loader = dataset.eval
#     cls_num_list = dataset.cls_num_list
#     if config.distributed:
#         train_sampler = dataset.dist_sampler

#     # define loss function (criterion) and optimizer

#     if config.arbloss:
#         criterion = ARB_Loss()
#     elif config.arbloss_fl:
#         criterion = ARB_Loss_FL()
#     elif config.warbloss:
#         criterion = WARB_Loss(ugcc=config.arb_loss_use_global_class_cnt)
#     else:
#         criterion = LabelAwareSmoothing(cls_num_list=cls_num_list, smooth_head=config.smooth_head,
#                                     smooth_tail=config.smooth_tail).cuda(config.gpu)

#     val_criterion = nn.CrossEntropyLoss().cuda(config.gpu)

#     optimizer = torch.optim.SGD([{"params": classifier.parameters()},
#                                 {'params': lws_model.parameters()}], config.lr,
#                                 momentum=config.momentum,
#                                 weight_decay=config.weight_decay)

#     for epoch in range(config.num_epochs):
#         if config.distributed:
#             train_sampler.set_epoch(epoch)

#         adjust_learning_rate(optimizer, epoch, config)

#         if config.dataset != 'places':
#             block = None
#         # train for one epoch
#         train(train_loader, model, classifier, lws_model, criterion, optimizer, epoch, config, logger, block)

#         # evaluate on validation set
#         acc1, ece = validate(val_loader, model, classifier, lws_model, val_criterion, config, logger, block)
#         # remember best acc@1 and save checkpoint
#         is_best = acc1 > best_acc1
#         best_acc1 = max(acc1, best_acc1)
#         if is_best:
#             its_ece = ece
#         logger.info('Best Prec@1: %.3f%% ECE: %.3f%%\n' % (best_acc1, its_ece))
#         if not config.multiprocessing_distributed or (config.multiprocessing_distributed
#                                                       and config.rank % ngpus_per_node == 0):
#             if config.dataset == 'places':
#                 save_checkpoint({
#                     'epoch': epoch + 1,
#                     'state_dict_model': model.state_dict(),
#                     'state_dict_classifier': classifier.state_dict(),
#                     'state_dict_block': block.state_dict(),
#                     'state_dict_lws_model': lws_model.state_dict(),
#                     'best_acc1': best_acc1,
#                     'its_ece': its_ece,
#                 }, is_best, model_dir)
#             else:
#                 save_checkpoint({
#                     'epoch': epoch + 1,
#                     'state_dict_model': model.state_dict(),
#                     'state_dict_classifier': classifier.state_dict(),
#                     'state_dict_lws_model': lws_model.state_dict(),
#                     'best_acc1': best_acc1,
#                     'its_ece': its_ece,
#                 }, is_best, model_dir)


# def train(train_loader, model, classifier, lws_model, criterion, optimizer, epoch, config, logger, block=None):
#     batch_time = AverageMeter('Time', ':6.3f')
#     data_time = AverageMeter('Data', ':6.3f')
#     losses = AverageMeter('Loss', ':.3f')
#     top1 = AverageMeter('Acc@1', ':6.3f')
#     top5 = AverageMeter('Acc@5', ':6.3f')
#     training_data_num = len(train_loader.dataset)
#     end_steps = int(np.ceil(float(training_data_num) / float(train_loader.batch_size)))
#     progress = ProgressMeter(
#         end_steps,
#         [batch_time, losses, top1, top5],
#         prefix="Epoch: [{}]".format(epoch))

#     # switch to train mode

#     if config.dataset == 'places':
#         model.eval()
#         if config.shift_bn:
#             block.train()
#         else:
#             block.eval()
#     else:
#         if config.shift_bn:
#             model.train()
#         else:
#             model.eval()
#     classifier.train()

#     end = time.time()

#     for i, (images, target) in enumerate(train_loader):
#         if i > end_steps:
#             break

#         # measure data loading time
#         data_time.update(time.time() - end)

#         if torch.cuda.is_available():
#             images = images.cuda(config.gpu, non_blocking=True)
#             target = target.cuda(config.gpu, non_blocking=True)

#         if config.mixup is True:
#             images, targets_a, targets_b, lam = mixup_data(images, target, alpha=config.alpha)
#             with torch.no_grad():
#                 if config.dataset == 'places':
#                     feat = block(model(images))
#                 else:
#                     feat = model(images)
#             output = classifier(feat.detach())
#             output = lws_model(output)

#             if config.arbloss or config.warbloss or config.arbloss_fl:
#                 if config.arb_loss_use_global_class_cnt:
#                     class_cnt = torch.tensor(train_loader.dataset.class_cnt, dtype=output.dtype).to(output.device)
#                 else:
#                     unique_labels, cnt = torch.unique(target, return_counts=True)
#                     class_cnt = torch.zeros(output.size(1), dtype=output.dtype).to(output.device)
#                     class_cnt[unique_labels] = cnt.float()
#             else:
#                 class_cnt = None
#             loss = mixup_criterion(criterion, output, targets_a, targets_b, lam, config, class_cnt)
#         else:
#             # compute output
#             with torch.no_grad():
#                 if config.dataset == 'places':
#                     feat = block(model(images))
#                 else:
#                     feat = model(images)
#             output = classifier(feat.detach())
#             output = lws_model(output)

#             if config.arbloss or config.warbloss or config.arbloss_fl:
#                 if config.arb_loss_use_global_class_cnt:
#                     class_cnt = torch.tensor(train_loader.dataset.class_cnt, dtype=output.dtype).to(output.device)
#                 else:
#                     unique_labels, cnt = torch.unique(target, return_counts=True)
#                     class_cnt = torch.zeros(output.size(1), dtype=output.dtype).to(output.device)
#                     class_cnt[unique_labels] = cnt.float()
#                 loss = criterion(output, target, class_cnt)
#             else:
#                 loss = criterion(output, target)

#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#         losses.update(loss.item(), images.size(0))
#         top1.update(acc1[0], images.size(0))
#         top5.update(acc5[0], images.size(0))

#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if i % config.print_freq == 0:
#             progress.display(i, logger)


# def validate(val_loader, model, classifier, lws_model, criterion, config, logger, block=None):
#     batch_time = AverageMeter('Time', ':6.3f')
#     losses = AverageMeter('Loss', ':.3f')
#     top1 = AverageMeter('Acc@1', ':6.3f')
#     top5 = AverageMeter('Acc@5', ':6.3f')
#     progress = ProgressMeter(
#         len(val_loader),
#         [batch_time, losses, top1, top5],
#         prefix='Eval: ')

#     # switch to evaluate mode
#     model.eval()
#     if config.dataset == 'places':
#         block.eval()
#     classifier.eval()
#     class_num = torch.zeros(config.num_classes).cuda()
#     correct = torch.zeros(config.num_classes).cuda()

#     confidence = np.array([])
#     pred_class = np.array([])
#     true_class = np.array([])

#     with torch.no_grad():
#         end = time.time()
#         for i, (images, target) in enumerate(val_loader):
#             if config.gpu is not None:
#                 images = images.cuda(config.gpu, non_blocking=True)
#             if torch.cuda.is_available():
#                 target = target.cuda(config.gpu, non_blocking=True)

#             # compute output
#             if config.dataset == 'places':
#                 feat = block(model(images))
#             else:
#                 feat = model(images)
#             output = classifier(feat)
#             output = lws_model(output)
#             loss = criterion(output, target)

#             # measure accuracy and record loss
#             acc1, acc5 = accuracy(output, target, topk=(1, 5))
#             losses.update(loss.item(), images.size(0))
#             top1.update(acc1[0], images.size(0))
#             top5.update(acc5[0], images.size(0))

#             _, predicted = output.max(1)
#             target_one_hot = F.one_hot(target, config.num_classes)
#             predict_one_hot = F.one_hot(predicted, config.num_classes)
#             class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
#             correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

#             prob = torch.softmax(output, dim=1)
#             confidence_part, pred_class_part = torch.max(prob, dim=1)
#             confidence = np.append(confidence, confidence_part.cpu().numpy())
#             pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
#             true_class = np.append(true_class, target.cpu().numpy())

#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if i % config.print_freq == 0:
#                 progress.display(i, logger)

#         acc_classes = correct / class_num
#         head_acc = acc_classes[config.head_class_idx[0]:config.head_class_idx[1]].mean() * 100
#         med_acc = acc_classes[config.med_class_idx[0]:config.med_class_idx[1]].mean() * 100
#         tail_acc = acc_classes[config.tail_class_idx[0]:config.tail_class_idx[1]].mean() * 100

#         # logger.info('* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'.format(top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))
#         logger.info(
#             '* Acc@1 {top1.avg:.3f}; Acc@5 {top5.avg:.3f}; many: {head_acc:.3f} ({correct_many:d}/{class_num_many:d}); median {med_acc:.3f} ({correct_median:d}/{class_num_median:d}); low {tail_acc:.3f} ({correct_low:d}/{class_num_low:d}).'.format(
#                 top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc,
#                 correct_many=int(correct[config.head_class_idx[0]:config.head_class_idx[1]].sum().item()),
#                 class_num_many=int(class_num[config.head_class_idx[0]:config.head_class_idx[1]].sum().item()),
#                 correct_median=int(correct[config.med_class_idx[0]:config.med_class_idx[1]].sum().item()),
#                 class_num_median=int(class_num[config.med_class_idx[0]:config.med_class_idx[1]].sum().item()),
#                 correct_low=int(correct[config.tail_class_idx[0]:config.tail_class_idx[1]].sum().item()),
#                 class_num_low=int(class_num[config.tail_class_idx[0]:config.tail_class_idx[1]].sum().item()),
#             ))

#         cal = calibration(true_class, pred_class, confidence, num_bins=15)
#         logger.info('* ECE   {ece:.3f}%.'.format(ece=cal['expected_calibration_error'] * 100))

#     return top1.avg, cal['expected_calibration_error'] * 100


# def save_checkpoint(state, is_best, model_dir):
#     filename = model_dir + '/current.pth.tar'
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, model_dir + '/model_best.pth.tar')


# def adjust_learning_rate(optimizer, epoch, config):
#     """Sets the learning rate"""
#     lr_min = 0
#     lr_max = config.lr
#     lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / config.num_epochs * 3.1415926535))

#     for idx, param_group in enumerate(optimizer.param_groups):
#         if idx == 0:
#             param_group['lr'] = config.lr_factor * lr
#         else:
#             param_group['lr'] = 1.00 * lr


# if __name__ == '__main__':
#     main()
