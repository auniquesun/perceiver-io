import os
from datetime import datetime
import wandb
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from datasets.shapenet_part import ShapeNetPart
from torch.utils.data import DataLoader

from utils import init, calculate_shape_IoU, Logger, AverageMeter
from utils import build_finetune_model, partseg_loss
from parser import args


def setup(rank):
    # initialization for distibuted training on multiple GPUs
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)

    
def cleanup():
    dist.destroy_process_group()


def main(rank, logger_name, log_path, log_file):
    if rank == 0:
        wandb.init(project=args.proj_name, name=args.exp_name)

    logger = Logger(logger_name=logger_name, log_path=log_path, log_file=log_file)

    setup(rank)

    train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_ft_points)
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    
    samples_per_gpu = args.batch_size // args.world_size
    train_loader = DataLoader(
                    train_dataset, 
                    sampler=train_sampler,
                    batch_size=samples_per_gpu, 
                    shuffle=False, 
                    num_workers=args.world_size,
                    pin_memory=True,
                    drop_last=False)

    val_dataset = ShapeNetPart(partition='test', num_points=args.num_ft_points)
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)

    val_samples_per_gpu = args.test_batch_size // args.world_size
    test_loader = DataLoader(val_dataset, 
                            sampler=val_sampler,
                            batch_size=val_samples_per_gpu, 
                            shuffle=False, 
                            num_workers=0, 
                            pin_memory=True, 
                            drop_last=False)
    
    seg_num_all = train_loader.dataset.seg_num_all
    seg_start_index = train_loader.dataset.seg_start_index

    model = build_finetune_model(rank=rank)
    model_ddp = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # ----- load pretrained model
    assert args.resume, 'Finetuning Perceiver_partseg requires pretrained model weights'
    map_location = torch.device('cuda:%d' % rank)
    pretrained = torch.load(args.pc_model_file, map_location=map_location)
    # append `module.` before key
    pretrained = {"module."+key: value for key, value in pretrained.items()}

    model_ddp.load_state_dict(pretrained, strict=False)

    if args.optim == 'sgd':
        optimizer = optim.SGD(
            model_ddp.parameters(),
            lr=args.lr,
            momentum=args.momentum)
    elif args.optim == 'adam':
        optimizer = optim.Adam(
            model_ddp.parameters(),
            lr=args.lr,
            weight_decay=1e-6)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(
            model_ddp.parameters(),
            lr=args.lr)

    logger.write(f'Using {args.optim} optimizer ...', rank=rank)

    if args.scheduler == 'cos':
        lr_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs)
    elif args.scheduler == 'coswarm':
        # lr_scheduler = CosineAnnealingWarmRestarts(
        #     optimizer, 
        #     T_0=args.warm_epochs)
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=args.step_size,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warm_epochs,
            gamma=args.gamma)
    elif args.scheduler == 'plateau':
        lr_scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=args.factor,
            patience=args.patience)
    elif args.scheduler == 'step':
        lr_scheduler = StepLR(
            optimizer, 
            step_size=args.step_size)

    criterion = partseg_loss
    scaler = GradScaler()

    ft_train_best_iou = .0
    ft_test_best_iou = .0
    for epoch in range(args.epochs):
        # ------ Train
        model_ddp.train()

        train_sampler.set_epoch(epoch)

        train_loss = AverageMeter()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []

        # 这个训练代码看起来好乱，numpy和torch混起来用
        for data, label, seg in train_loader:
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                seg = seg - seg_start_index
                data, seg = data.to(rank), seg.to(rank)
                batch_size = data.size()[0]
                seg_pred = model_ddp(data)
                loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss.update(loss.item(), batch_size)
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_label_seg.append(label.reshape(-1))

        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)

        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)

        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)

        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label_seg = np.concatenate(train_label_seg)
        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, args.class_choice)
        train_iou = np.mean(train_ious)

        outstr = 'FT_train (%d/%d), loss: %.6f, ft_acc: %.6f, ft_avg acc: %.6f, ft_iou: %.6f' % \
                 (epoch, args.epochs, train_loss.avg, train_acc, avg_per_class_acc, train_iou)
        logger.write(outstr, rank=rank)

        # ------ Test
        with torch.no_grad():
            # logger.write('Start testing on the %s test set ...' % args.ft_dataset, rank=rank)
            outstr, test_loss, test_acc, test_acc_per_class, test_iou = test(rank, epoch, model_ddp, test_loader, criterion)
            logger.write(outstr, rank=rank)

            if rank == 0:
                if train_iou > ft_train_best_iou:
                    ft_train_best_iou = train_iou

                if test_iou > ft_test_best_iou:
                    ft_test_best_iou = test_iou
                    logger.write(f'Find new highest Mean IoU score: {ft_test_best_iou} !', rank=rank)
                    logger.write('Saving best model ...', rank=rank)
                    save_path = os.path.join('runs', args.proj_name, args.exp_name, 'models', 'model_best.pth')
                    torch.save(model_ddp.module.state_dict(), save_path)
                
                if epoch % args.save_freq == 0:
                    logger.write(f'Saving {epoch}th model ...', rank=rank)
                    save_path = os.path.join('runs', args.proj_name, args.exp_name, 'models', f'model_epoch{epoch}.pth')
                    torch.save(model_ddp.module.state_dict(), save_path)

                wandb_log = dict()
                if args.scheduler == 'coswarm':
                    wandb_log['learning_rate'] = lr_scheduler.get_lr()[0]
                else:
                    wandb_log['learning_rate'] = lr_scheduler.get_last_lr()[0]
                wandb_log["ft_train_loss"] = train_loss.avg
                wandb_log["ft_train_acc"] = train_acc
                wandb_log["ft_train_acc_per_class"] = avg_per_class_acc
                wandb_log["ft_train_mean_iou"] = train_iou
                wandb_log["ft_train_best_iou"] = ft_train_best_iou
                wandb_log["ft_test_loss"] = test_loss
                wandb_log["ft_test_acc"] = test_acc
                wandb_log["ft_test_acc_per_class"] = test_acc_per_class
                wandb_log["ft_test_mean_iou"] = test_iou
                wandb_log["ft_test_best_iou"] = ft_test_best_iou
                wandb.log(wandb_log)

            lr_scheduler.step()

    if rank == 0:
        logger.write('Saving the last model ...', rank=rank)
        save_path = os.path.join('runs', args.proj_name, args.exp_name, 'models', f'model_last.pth')
        torch.save(model_ddp.module.state_dict(), save_path)
        logger.write(f'Final highest Mean IoU score: {ft_test_best_iou} !', rank=rank)
        logger.write('End of DDP finetuning on %s ...' % args.ft_dataset, rank=rank)
        wandb.finish()
    cleanup()


def test(rank, epoch, model, test_loader, criterion):
    model.eval()

    seg_num_all = test_loader.dataset.seg_num_all
    seg_start_index = test_loader.dataset.seg_start_index

    test_loss = AverageMeter()
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []
    for data, label, seg in test_loader:
        seg = seg - seg_start_index
        
        data, seg = data.to(rank), seg.to(rank)
        batch_size = data.size()[0]
        # seg_pred: [batch_size, num_points, seg_num_all]
        seg_pred = model(data)
        loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
        test_loss.update(loss.item(), batch_size)
        pred = seg_pred.max(dim=2)[1]
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(label.reshape(-1))

    # tmp = [1.0 for _ in range(args.world_size)]
    # dist.all_gather_object(tmp, test_loss.avg)
    # test_loss = np.mean(tmp)

    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)

    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    # tmp = [1.0 for _ in range(args.world_size)]
    # dist.all_gather_object(tmp, test_acc)
    # test_acc = np.mean(tmp)

    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    # tmp = [1.0 for _ in range(args.world_size)]
    # dist.all_gather_object(tmp, avg_per_class_acc)
    # test_acc_per_class = np.mean(tmp)

    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    shape_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)

    test_iou_per_gpu = np.mean(shape_ious)
    # tmp = [1.0 for _ in range(args.world_size)]
    # dist.all_gather_object(tmp, test_iou_per_gpu)
    # test_iou = np.mean(tmp)

    outstr = 'FT_test (%d/%d), loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % \
            (epoch, args.epochs, test_loss.avg, test_acc, avg_per_class_acc, test_iou_per_gpu)
    
    return outstr, test_loss.avg, test_acc, avg_per_class_acc, test_iou_per_gpu


if __name__ == "__main__":
    init(args.proj_name, args.exp_name, args.main_program, args.model_name)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    
    logger_name = args.proj_name
    log_path = os.path.join('runs', args.proj_name, args.exp_name)
    log_file = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

    logger = Logger(logger_name=logger_name, log_path=log_path, log_file=log_file)

    if args.cuda:
        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            logger.write('%d GPUs are available and %d of them are used. Ready for DDP finetuning' % (num_devices, args.world_size), rank=0)
            logger.write(str(args), rank=0)
            # Set seed for generating random numbers for all GPUs, and 
            # torch.cuda.manual_seed() is insufficient to get determinism for all GPUs
            mp.spawn(main, args=(logger_name, log_path, log_file), nprocs=args.world_size)
        else:
            logger.write('Only one GPU is available, the process will be much slower', rank=0)
    else:
        logger.write('CUDA is unavailable! Exit', rank=0)
