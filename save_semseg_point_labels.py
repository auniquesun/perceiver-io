import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets.s3dis import S3DISDataset_HDF5
from utils import build_ft_semseg
from parser import args


test_dataset = S3DISDataset_HDF5(split='test', test_area=args.test_area)
# 只需要 test_loader 即可
test_loader = DataLoader(test_dataset, 
                        batch_size=args.test_batch_size, 
                        shuffle=False, 
                        num_workers=0, 
                        pin_memory=True, 
                        drop_last=False)

device = torch.device('cuda:0')
# build_ft_cls 是预训练模型，设成 imc-only，只加载点云模型
model = build_ft_semseg(rank=device)

save_path = os.path.join('runs', args.proj_name, args.exp_name, 'models', args.pc_model_file)
state_dict = torch.load(save_path)
# NOTE: 如果加载的是ft_cls模型，要替换一下key，同时 `load_state_dict` 中 strict 要设为 False
# state_dict = {key.replace("0.", ""): value for key, value in state_dict.items()}
model.load_state_dict(state_dict, strict=True)

with torch.no_grad():
    model = model.eval()

    feats_test = []
    labels_test = []

    print('Forward passing ...')
    for i, (points, sem_label) in enumerate(test_loader):
        # points: [batch, num_points, 9]    --> x,y,z,r,g,b + normal vector
        # sem_label: [batch, num_points], label of object parts

        batch_size, num_points, _ = points.size()
        # sem_label: it is required to convert to `torch.long` type
        points, sem_label = points[:, :, :6].to(device), sem_label.long().to(device)
        # sem_pred: [batch, num_points, num_obj_classes]
        sem_pred = model(points)
        # to get integer label, sem_pred: [batch, num_points]
        sem_pred = sem_pred.max(dim=2)

        for i in batch_size:
            xyz = points[i, :, :3].tolist()
            label = sem_pred[i]
    print('Forward done!')

    state_dict = {'feats_test': np.array(feats_test), 'labels_test': np.array(labels_test)}
    if args.pt_dataset == "ModelNet40":
        torch.save(state_dict, 'visualization/ft_MN_test_feats_labels.pth')
    else:
        torch.save(state_dict, 'visualization/ft_SO_test_feats_labels.pth')
    print('Save done!')
