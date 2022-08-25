import torch

from perceiver.model.core import PerceiverEncoder, PerceiverDecoder, PerceiverIO, ClassificationOutputAdapter

from perceiver.model.image import ImageInputAdapter

from perceiver.model.pointcloud import PointCloudInputAdapter

from fvcore.nn import FlopCountAnalysis

from parser import args


# ------ Image
# Fourier-encodes pixel positions and flatten along spatial dimensions
# input_adapter = ImageInputAdapter(
#   image_shape=(137, 137, 3),  # M = 137 * 137
#   num_frequency_bands=64,
# )

# # Generic Perceiver encoder
# encoder = PerceiverEncoder(
#   input_adapter=input_adapter,
#   num_latents=128,  # N
#   num_latent_channels=256,  # D
#   num_cross_attention_qk_channels=256,  # C
#   num_cross_attention_heads=4,
#   num_self_attention_heads=4,
#   num_self_attention_layers_per_block=6,
#   num_self_attention_blocks=1,
#   dropout=0.1,
# )

# pytorch_total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
# print('total parameters:', pytorch_total_params)

# input = torch.randn(1, 137, 137, 3)
# output = encoder(input)
# print('output.shape:', output.shape)

# flops = FlopCountAnalysis(encoder, input)
# print('total flops:', flops.total())

# ------ PointCloud
# input_adapter_pc = PointCloudInputAdapter(
#   pointcloud_shape=(2048, 3),
#   num_input_channels=256,
#   num_groups=128,
#   group_size=32
# )

# encoder_pc = PerceiverEncoder(
#   input_adapter=input_adapter_pc,
#   num_latents=128,  # N
#   num_latent_channels=256,  # D
#   num_cross_attention_qk_channels=input_adapter_pc.num_input_channels,  # C
#   num_cross_attention_heads=1,
#   num_self_attention_heads=4,
#   num_self_attention_layers_per_block=6,
#   num_self_attention_blocks=1,
#   dropout=0.0,
# )

# input_pc = torch.randn(1, 2048, 3)
# output_pc = encoder_pc(input_pc)
# print(output_pc.shape)

# pytorch_total_params = sum(p.numel() for p in encoder_pc.parameters() if p.requires_grad)
# print('total parameters:', pytorch_total_params)

# flops = FlopCountAnalysis(encoder_pc, input_pc)
# print('total flops:', flops.total())

# ------ 经过测试，数据加载没问题，问题出在其他地方
# from torch.utils.data import DataLoader
# from datasets.data import ShapeNetRender, ModelNet40SVM
# from utils import transform

# train_val_loader = DataLoader(
#     ModelNet40SVM(partition='train', num_points=1024), 
#     batch_size=960, shuffle=True)
# print('len(train_val_loader):', len(train_val_loader))
# test_val_loader = DataLoader(
#     ModelNet40SVM(partition='test', num_points=1024), 
#     batch_size=960, shuffle=True)
# print('len(test_val_loader):', len(test_val_loader))

# train_set = ShapeNetRender(img_transform=transform)
# train_loader = DataLoader(
#         train_set,
#         batch_size=210,
#         shuffle=True,
#         num_workers=0,
#         pin_memory=True,
#         drop_last=False
#     )
# for i, ((pc_t1, pc_t2), imgs) in enumerate(train_loader):
#     batch_size = pc_t1.shape[0]
#     pc_t1, pc_t2, imgs = pc_t1.to(0), pc_t2.to(0), imgs.to(0)
#     # imgs = torch.permute(imgs, (0, 2, 3, 1))
#     print('batch_size:', batch_size)
#     print('pc_t1.shape:', pc_t1.shape)
#     print('pc_t2.shape:', pc_t2.shape)
#     print('imgs.shape:', imgs.shape)
#     break


# ------ ShapeNetPart
# from torch.utils.data import DataLoader
# from datasets.shapenet_part import ShapeNetPart
# from utils import part2category

# train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_ft_points, class_choice=args.class_choice)
# train_loader = DataLoader(
#                     train_dataset, 
#                     batch_size=210, 
#                     shuffle=False, 
#                     num_workers=2*args.world_size,
#                     pin_memory=True,
#                     drop_last=False)
# print('len(train_loader):', len(train_loader))
# # print('seg_num_all:', train_loader.dataset.seg_num_all)
# # print('seg_start_index:', train_loader.dataset.seg_start_index)

# num_part_classes = 50
# part2count = dict()
# total = 0
# for i, (points, cls_label, seg_label) in enumerate(train_loader):
# #     print('points:', points.shape)
# #     print('cls_label:', cls_label.shape)
# #     print('seg_label:', seg_label.shape)
# #     break
#         for j in range(num_part_classes):
#             if j not in part2count.keys():
#                 part2count[j] = 0
#             part2count[j] += torch.eq(seg_label, j).sum()
        
#         batch, num_points = seg_label.shape
#         total += batch * num_points

# for part in part2count.keys():
#     print(part2category[part], ':', part2count[part], 'ratio :', part2count[part]/total)


# ------------- Count Parameters and FLOPs
# input_adapter = PointCloudInputAdapter(
#     pointcloud_shape=(2048, 3),
#     num_input_channels=256)

# encoder = PerceiverEncoder(
#     input_adapter=input_adapter,
#     num_latents=128,  # N
#     num_latent_channels=256,  # D
#     num_cross_attention_heads=4,
#     num_cross_attention_qk_channels=input_adapter.num_input_channels,  # C
#     num_cross_attention_v_channels=None,
#     num_cross_attention_layers=1,
#     first_cross_attention_layer_shared=False,
#     cross_attention_widening_factor=2,
#     num_self_attention_heads=4,
#     num_self_attention_qk_channels=None,
#     num_self_attention_v_channels=None,
#     num_self_attention_layers_per_block=6,
#     num_self_attention_blocks=1,
#     first_self_attention_block_shared=True,
#     self_attention_widening_factor=2,
#     max_dpr=0.0,
#     atten_drop=0.1,
#     mlp_drop=0.5)

# output_adapter = ClassificationOutputAdapter(
#     num_classes=40,
#     num_output_queries=1,
#     num_output_query_channels=256)
# decoder = PerceiverDecoder(
#     output_adapter=output_adapter,
#     num_latent_channels=256,  # D
#     num_cross_attention_heads=4,
#     num_cross_attention_qk_channels=256,
#     num_cross_attention_v_channels=None,
#     cross_attention_widening_factor=2,
#     num_self_attention_heads=4,
#     num_self_attention_qk_channels=None,
#     num_self_attention_v_channels=None,
#     num_self_attention_layers_per_block=2,  # In decoder, set `num_sa_layers=2`
#     self_attention_widening_factor=2,
#     atten_drop=0.1,
#     mlp_drop=0.5)

# model = PerceiverIO(encoder, decoder)

# input = torch.randn(2,2048,3)
# output = model(input)
# print('output.shape:', output.shape)
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('total parameters:', pytorch_total_params)
# flops = FlopCountAnalysis(model, input)
# print('fvcore - total flops:', flops.total())


# ------------- 
# from datasets.shapenet_part import ShapeNetPart
# from torch.utils.data import DataLoader

# train_dataset = ShapeNetPart(
#     partition='trainval', 
#     num_points=args.num_ft_points, 
#     class_choice=args.class_choice)

# train_loader = DataLoader(
#     train_dataset, 
#     batch_size=args.batch_size, 
#     shuffle=False, 
#     num_workers=args.world_size,
#     pin_memory=True, 
#     drop_last=False)

# print('len(train_loader):', len(train_loader))

# val_dataset = ShapeNetPart(
#     partition='test', 
#     num_points=args.num_ft_points, 
#     class_choice=args.class_choice)

# test_loader = DataLoader(
#     val_dataset, 
#     batch_size=args.test_batch_size, 
#     shuffle=False, 
#     num_workers=0, 
#     pin_memory=True, 
#     drop_last=False)

# print('len(test_loader):', len(test_loader))


# ----------- partseg model
# from perceiver.model.core import PerceiverEncoder_partseg

# input_adapter = PointCloudInputAdapter(
#         pointcloud_shape=(args.num_pt_points, 3),
#         num_input_channels=args.num_latent_channels,
#         num_groups=args.num_groups,
#         group_size=args.group_size)
# output_adapter = ClassificationOutputAdapter(
#     num_classes=args.num_obj_classes,
#     num_output_queries=args.output_seq_length,
#     num_output_query_channels=args.num_latent_channels)
# model = PerceiverEncoder_partseg(
#     input_adapter=input_adapter,
#     output_adapter=output_adapter,
#     num_latents=args.num_pc_latents,  # N
#     num_latent_channels=args.num_latent_channels,  # D
#     num_cross_attention_heads=args.num_ca_heads,
#     num_cross_attention_qk_channels=input_adapter.num_input_channels,  # C
#     num_cross_attention_v_channels=None,
#     num_cross_attention_layers=args.num_ca_layers,
#     first_cross_attention_layer_shared=False,
#     cross_attention_widening_factor=args.mlp_widen_factor,
#     num_self_attention_heads=args.num_sa_heads,
#     num_self_attention_qk_channels=None,
#     num_self_attention_v_channels=None,
#     num_self_attention_layers_per_block=args.num_sa_layers_per_block,
#     num_self_attention_blocks=args.num_sa_blocks,
#     first_self_attention_block_shared=True,
#     self_attention_widening_factor=args.mlp_widen_factor,
#     dropout=args.atten_drop)

# print('num_pc_latents:', args.num_pc_latents)
# print('num_latent_channels:', args.num_latent_channels)
# x = torch.randn(2, 1024, 3)
# cls_label = torch.randn(2, 16)
# output = model(x, cls_label)
# print('output.shape:', output.shape)

# ----------- partseg model
# from perceiver.model.pointcloud.partseg import CrossFormer
# from torch.nn import CrossEntropyLoss

# input_adapter = PointCloudInputAdapter(
#         pointcloud_shape=(2048, 3),
#         num_input_channels=384,
#         num_groups=128,
#         group_size=32)

# model = CrossFormer(
#         input_adapter=input_adapter,
#         num_latents=128,
#         num_latent_channels=384,
#         group_size=32,
#         num_cross_attention_layers=1,
#         num_cross_attention_heads=6,
#         num_self_attention_layers=11,
#         num_self_attention_heads=6,
#         mlp_widen_factor=4,
#         max_dpr=0.1,
#         atten_drop=.0,
#         mlp_drop=.0,
#         layer_idx=[3,7,11],
#         num_part_classes=50
#         )
# criterion = CrossEntropyLoss()

# points = torch.randn(30, 2048, 3)
# obj_cls_labels = torch.randn(30, 16)
# output = model(points, obj_cls_labels)
# target = torch.randint(0, 50, (30,2048))
# loss = criterion(output.reshape(-1, 50), target.reshape(-1))

# print('output.shape:', output.shape)
# print('loss:', loss.item())

# print(args.layer_idx)


# ----------- semseg model
# from perceiver.model.pointcloud.semseg import CrossFormer_semseg
# from torch.nn import CrossEntropyLoss

# input_adapter = PointCloudInputAdapter(
#         pointcloud_shape=(2048, 6),
#         num_input_channels=384,
#         num_groups=128,
#         group_size=32)

# model = CrossFormer_semseg(
#         input_adapter=input_adapter,
#         point_channels=6,
#         num_latents=128,
#         num_latent_channels=384,
#         group_size=32,
#         num_cross_attention_layers=1,
#         num_cross_attention_heads=6,
#         num_self_attention_layers=11,
#         num_self_attention_heads=6,
#         mlp_widen_factor=4,
#         max_dpr=0.1,
#         atten_drop=.0,
#         mlp_drop=.0,
#         layer_idx=[3,7,11],
#         num_obj_classes=13
#         )
# criterion = CrossEntropyLoss()

# points = torch.randn(10, 2048, 6)
# output = model(points)
# target = torch.randint(0, 13, (10,2048))
# loss = criterion(output.reshape(-1, 13), target.reshape(-1))

# print('output.shape:', output.shape)
# print('loss:', loss.item())


# ------ CrossFormer_img_mp
# from perceiver.model.pointcloud import CrossFormer_img_mp

# model = CrossFormer_img_mp()

# input = torch.randn(2, 144, 144, 3)
# x_latent_feats, backbone_feats = model(input)

# print('x_latent_feats.shape:', x_latent_feats.shape)
# print('backbone_feats.shape:', backbone_feats.shape)


# ------ CrossFormer_pc_mp_ft
from perceiver.model.pointcloud import CrossFormer_pc_mp_ft

input_adapter = PointCloudInputAdapter(
    pointcloud_shape=(1024, 3),
    num_input_channels=384)
model = CrossFormer_pc_mp_ft(input_adapter=input_adapter, num_latents=96, mlp_widen_factor=2)

input = torch.randn(2, 1024, 3)
output = model(input)

print('output.shape:', output.shape)
