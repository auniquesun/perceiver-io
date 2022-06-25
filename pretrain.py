import os
import wandb
import numpy as np
from datetime import datetime
from lightly.loss.ntx_ent_loss import NTXentLoss

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader
from datasets.data import ShapeNetRender, ModelNet40SVM

from utils import build_model, transform
from sklearn.svm import SVC

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from utils import init, Logger, AverageMeter
from parser import args


def setup(rank):
    # initialization for distributed training on multiple GPUs
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)

    
def cleanup():
    dist.destroy_process_group()


def main(rank, logger_name, log_path, log_file):
    if rank == 0:
        wandb.init(project=args.proj_name, name=args.exp_name)

    # NOTE: only write logs of the results obtained by
    # the first gpu device whose rank=0, otherwise produce duplicate logs
    logger = Logger(logger_name=logger_name, log_path=log_path, log_file=log_file)

    setup(rank)

    train_set = ShapeNetRender(transform)
    train_sampler = DistributedSampler(train_set, num_replicas=args.world_size, rank=rank)

    assert args.batch_size % args.world_size == 0, \
        'Argument `batch_size` should be divisible by `world_size`'

    samples_per_gpu = args.batch_size // args.world_size
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=samples_per_gpu,
        shuffle=False,
        num_workers=args.world_size,
        pin_memory=True,
        drop_last=False)

    # here set `num_workers=0` because ModelNet40 has much less samples than ShapeNet 
    # thus smaller len(train_val_loader) and len(test_val_loader)
    train_val_loader = DataLoader(
        ModelNet40SVM(partition='train', num_points=args.num_test_points), 
        batch_size=args.test_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_val_loader = DataLoader(
        ModelNet40SVM(partition='test', num_points=args.num_test_points), 
        batch_size=args.test_batch_size, shuffle=True, num_workers=0, pin_memory=True)

    pc_model, img_model = build_model()
    pc_model, img_model = pc_model.to(rank), img_model.to(rank)

    if args.resume:
        path = os.path.join('runs', args.proj_name, args.exp_name, 'models', args.pc_model_file)
        pretrained = torch.load(path)
        pc_model.load_state_dict(pretrained)
        path = os.path.join('runs', args.proj_name, args.exp_name, 'models', args.img_model_file)
        pretrained = torch.load(path)
        img_model.load_state_dict(pretrained)

    pc_model_ddp = DDP(pc_model, device_ids=[rank], find_unused_parameters=False)
    img_model_ddp = DDP(img_model, device_ids=[rank], find_unused_parameters=False)

    parameters = list(pc_model_ddp.parameters()) + list(img_model_ddp.parameters())

    if args.optim == 'sgd':
        optimizer = optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum)
    elif args.optim == 'adam':
        optimizer = optim.Adam(
            parameters,
            lr=args.lr,
            weight_decay=1e-6)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(
            parameters,
            lr=args.lr)
    
    logger.write(f'Using {args.optim} optimizer ...', rank=rank)

    # 这也是要调优的
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

    scaler = GradScaler()
    criterion = NTXentLoss(temperature = 0.1).to(rank)
    best_test_acc = .0
    for epoch in range(args.epochs):
        # ------ Train
        pc_model_ddp.train()
        img_model_ddp.train()
        train_sampler.set_epoch(epoch)

        # average losses across all scanned batches within an epoch
        train_imid_loss = AverageMeter()
        train_cmid_loss = AverageMeter()
        train_loss = AverageMeter()

        for i, ((pc_t1, pc_t2), imgs) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                pc_t1, pc_t2, imgs = pc_t1.to(rank), pc_t2.to(rank), imgs.to(rank)
                imgs = torch.permute(imgs, (0, 2, 3, 1))
                # data.shape: [B, N, C]
                batch_size = pc_t1.shape[0]

                pc = torch.cat([pc_t1, pc_t2], dim=0)
                pc_feats = pc_model_ddp(pc)
                img_feats = img_model_ddp(imgs)

                pc_t1_feats = pc_feats[:batch_size, :]
                pc_t2_feats = pc_feats[batch_size:, :]

                loss_imid = criterion(pc_t1_feats, pc_t2_feats)
                pc_feats = (pc_t1_feats + pc_t2_feats) / 2
                loss_cmid = criterion(pc_feats, img_feats)
                # penalyze more on loss_cmid
                total_loss = loss_imid + args.cmid_weight*loss_cmid

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_imid_loss.update(loss_imid.item(), batch_size)
            train_cmid_loss.update(loss_cmid.item(), batch_size)
            train_loss.update(total_loss.item(), batch_size)

            if i % args.print_freq == 0:
                logger.write(f'Epoch: {epoch}/{args.epochs}, Batch: {i}/{len(train_loader)}, '
                             f'Loss IMID: {loss_imid}, Loss CMID: {loss_cmid} '
                             f'Loss Total: {total_loss}', rank=rank)

        # tmp = [.0 for _ in range(args.world_size)]
        # dist.all_gather_object(tmp, train_imid_loss.avg)
        # overall_train_imid_loss = np.mean(tmp)

        # tmp = [.0 for _ in range(args.world_size)]
        # dist.all_gather_object(tmp, train_cmid_loss.avg)
        # overall_train_cmid_loss = np.mean(tmp)

        # tmp = [.0 for _ in range(args.world_size)]
        # dist.all_gather_object(tmp, train_loss.avg)
        # overall_train_loss = np.mean(tmp)

        # ------ Test
        with torch.no_grad():   # it will disable gradients computation and save memory
            pc_model_ddp.eval()
            img_model_ddp.eval()

            train_feats = []
            train_labels = []

            for i, (data, label) in enumerate(train_val_loader):
                labels = list(map(lambda x: x[0], label.numpy().tolist()))
                data = data.to(rank)
                with torch.no_grad():
                    feats = pc_model_ddp(data)
                feats = feats.detach().cpu().numpy()
                train_feats.extend(feats)
                train_labels.extend(labels)

            train_feats = np.array(train_feats)
            train_labels = np.array(train_labels)

            # ------ LinearSVC vs. SVC
            # The strength of regularization is inversely to C. Must be strictly positive.
            # svm = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, C=0.1, random_state=args.seed, max_iter=1000)
            svm = SVC(C=0.1, kernel='linear')

            logger.write('Training SVM ...', rank=rank)
            svm.fit(train_feats, train_labels)
            
            test_feats = []
            test_labels = []

            for i, (data, label) in enumerate(test_val_loader):
                labels = list(map(lambda x: x[0], label.numpy().tolist()))
                data = data.to(rank)
                with torch.no_grad():
                    feats = pc_model_ddp(data)
                feats = feats.detach().cpu().numpy()
                test_feats.extend(feats)
                test_labels.extend(labels)

            test_feats = np.array(test_feats)
            test_labels = np.array(test_labels)

            logger.write('Testing SVM ...', rank=rank)
            test_accuracy = svm.score(test_feats, test_labels)

            # tmp = [.0 for _ in range(args.world_size)]
            # dist.all_gather_object(tmp, test_accuracy)
            # overall_test_acc = np.mean(tmp)
            overall_test_acc = test_accuracy

            if rank == 0:
                logger.write(f'Test Accuracy of SVM: {overall_test_acc}', rank=rank)

                if overall_test_acc > best_test_acc:
                    best_test_acc = overall_test_acc
                    logger.write(f'Finding new highest test score: {best_test_acc} !', rank=rank)
                    logger.write('Saving best model ...', rank=rank)
                    save_path = os.path.join('runs', args.proj_name, args.exp_name, 'models', 'pc_model_best.pth')
                    torch.save(pc_model_ddp.module.state_dict(), save_path)
                    save_path = os.path.join('runs', args.proj_name, args.exp_name, 'models', 'img_model_best.pth')
                    torch.save(img_model_ddp.module.state_dict(), save_path)

                if epoch % args.save_freq == 0:
                    logger.write(f'Saving {epoch}th model ...', rank=rank)
                    save_path = os.path.join('runs', args.proj_name, args.exp_name, 'models', f'pc_model_epoch{epoch}.pth')
                    torch.save(pc_model_ddp.module.state_dict(), save_path)
                    save_path = os.path.join('runs', args.proj_name, args.exp_name, 'models', f'img_model_epoch{epoch}.pth')
                    torch.save(img_model_ddp.module.state_dict(), save_path)

                wandb_log = dict()
                # NOTE get_lr() vs. get_last_lr(), which should be used? The answer is get_last_lr()
                # https://discuss.pytorch.org/t/whats-the-difference-between-get-lr-and-get-last-lr/121681
                if args.scheduler == 'coswarm':
                    wandb_log['learning_rate'] = lr_scheduler.get_lr()[0]
                else:
                    wandb_log['learning_rate'] = lr_scheduler.get_last_lr()[0]
                wandb_log['Pretrain Loss'] = train_loss.avg
                wandb_log['Pretrain IMID Loss'] = train_imid_loss.avg
                wandb_log['Pretrain CMID Loss'] = train_cmid_loss.avg
                wandb_log['svm_test_acc'] = overall_test_acc
                wandb_log['svm_best_acc'] = best_test_acc
                wandb.log(wandb_log)

            # adjust learning rate before a new epoch
            lr_scheduler.step()

    if rank == 0:
        logger.write(f'Final highest test score: {best_test_acc} !', rank=rank)
        logger.write(f'Saving the last model ...', rank=rank)
        save_path = os.path.join('runs', args.proj_name, args.exp_name, 'models', f'pc_model_last.pth')
        torch.save(pc_model_ddp.module.state_dict(), save_path)
        save_path = os.path.join('runs', args.proj_name, args.exp_name, 'models', f'img_model_last.pth')
        torch.save(img_model_ddp.module.state_dict(), save_path)
        wandb.finish()

    cleanup()


if '__main__' == __name__:
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
            logger.write('%d GPUs are available and %d of them are used. Ready for DDP training' % (num_devices, args.world_size), rank=0)
            logger.write(str(args), rank=0)
            # Set seed for generating random numbers for all GPUs, and 
            # torch.cuda.manual_seed() is insufficient to get determinism for all GPUs
            mp.spawn(main, args=(logger_name, log_path, log_file), nprocs=args.world_size)
        else:
            logger.write('Only one GPU is available, the process will be much slower. Exit', rank=0)
    else:
        logger.write('CUDA is unavailable! Exit', rank=0)
