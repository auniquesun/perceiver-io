import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets.data import ModelNet40SVM, ScanObjectNNSVM
from utils import build_model
from parser import args


if args.pt_dataset == "ModelNet40":
    test_loader = DataLoader(
        ModelNet40SVM(partition='test', num_points=args.num_test_points), 
        batch_size=args.test_batch_size, shuffle=True)
elif args.pt_dataset == "ScanObjectNN":
    test_loader = DataLoader(
        ScanObjectNNSVM(partition='test', num_points=args.num_test_points), 
        batch_size=args.test_batch_size, shuffle=True)

device = torch.device('cuda:0')
# build_model 是预训练模型
model = build_model(rank=device)

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
    for i, (data, label) in enumerate(test_loader):
        if args.pt_dataset == "ModelNet40":
            labels = list(map(lambda x: x[0], label.tolist()))
        elif args.pt_dataset == "ScanObjectNN":
            labels = label.tolist()
        data = data.to(device)
        # model(data)[1] is the features output by CrossFormer backbone
        feats = model(data)[1].tolist()
        feats_test.extend(feats)
        labels_test.extend(labels)
    print('Forward done!')

    state_dict = {'feats_test': np.array(feats_test), 'labels_test': np.array(labels_test)}
    if args.pt_dataset == "ModelNet40":
        torch.save(state_dict, 'visualization/pt_MN_test_feats_labels.pth')
    else:
        torch.save(state_dict, 'visualization/pt_SO_test_feats_labels.pth')
    print('Save done!')
