import os
import numpy as np
from sklearn.svm import SVC

import torch
from torch.utils.data import DataLoader

from datasets.data import ModelNet40SVM, ScanObjectNNSVM

from utils import build_model

from parser import args


device = torch.device("cuda")
save_path = os.path.join('runs', args.proj_name, args.exp_name, 'models', 'pc_model_best.pth')
state_dict = torch.load(save_path)

pc_model, _ = build_model()
model = pc_model.to(device)

model.load_state_dict(state_dict)
model = model.eval()

if args.pt_dataset == "ModelNet40":
    train_loader = DataLoader(ModelNet40SVM(partition='train', num_points=args.num_test_points),
                                batch_size=args.test_batch_size, shuffle=True)
    test_loader = DataLoader(ModelNet40SVM(partition='test', num_points=args.num_test_points),
                                batch_size=args.test_batch_size, shuffle=True)
elif args.pt_dataset == "ScanObjectNN":
    train_loader = DataLoader(ScanObjectNNSVM(partition='train', num_points=args.num_test_points),
                                batch_size=args.test_batch_size, shuffle=True)
    test_loader = DataLoader(ScanObjectNNSVM(partition='test', num_points=args.num_test_points),
                                batch_size=args.test_batch_size, shuffle=True)

feats_train = []
labels_train = []
for i, (data, label) in enumerate(train_loader):
    if args.pt_dataset == "ModelNet40":
        labels = list(map(lambda x: x[0],label.numpy().tolist()))
    elif args.pt_dataset == "ScanObjectNN":
        labels = label.numpy().tolist()
    data = data.to(device)
    with torch.no_grad():
        feats = model(data)
    feats = feats.detach().cpu().numpy()
    feats_train.extend(feats)
    labels_train.extend(labels)

feats_train = np.array(feats_train)
labels_train = np.array(labels_train)
print(feats_train.shape)

feats_test = []
labels_test = []
for i, (data, label) in enumerate(test_loader):
    if args.pt_dataset == "ModelNet40":
        labels = list(map(lambda x: x[0],label.numpy().tolist()))
    elif args.pt_dataset == "ScanObjectNN":
        labels = label.numpy().tolist()
    data = data.to(device)
    with torch.no_grad():
        feats = model(data)
    feats = feats.detach().cpu().numpy()
    feats_test.extend(feats)
    labels_test.extend(labels)

feats_test = np.array(feats_test)
labels_test = np.array(labels_test)
print(feats_test.shape)

# ScanOjbectNN 效果有点差啊，只有57%的准确性
# Linear SVM parameter C, can be tuned
c = args.svm_coff 
model_tl = SVC(C = c, kernel ='linear')
model_tl.fit(feats_train, labels_train)
print(f"C = {c} : {model_tl.score(feats_test, labels_test)}")
