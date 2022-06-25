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

# train_set = ShapeNetRender(transform)
# train_loader = DataLoader(
#         train_set,
#         batch_size=4,
#         shuffle=True,
#         num_workers=0,
#         pin_memory=True,
#         drop_last=False
#     )

# print('len(train_loader):', len(train_loader))

# for i, ((pc_t1, pc_t2), imgs) in enumerate(train_loader):
#     batch_size = pc_t1.shape[0]
#     pc_t1, pc_t2, imgs = pc_t1.to(0), pc_t2.to(0), imgs.to(0)
#     imgs = torch.permute(imgs, (0, 2, 3, 1))
#     print('batch_size:', batch_size)
#     print('pc_t1.shape:', pc_t1.shape)
#     print('pc_t2.shape:', pc_t2.shape)
#     print('imgs.shape:', imgs.shape)
#     break

# ------------- 
input_adapter = PointCloudInputAdapter(
    pointcloud_shape=(2048, 3),
    num_input_channels=256,
    num_groups=128,
    group_size=32)

encoder = PerceiverEncoder(
    input_adapter=input_adapter,
    num_latents=128,  # N
    num_latent_channels=256,  # D
    num_cross_attention_heads=4,
    num_cross_attention_qk_channels=input_adapter.num_input_channels,  # C
    num_cross_attention_v_channels=None,
    num_cross_attention_layers=1,
    first_cross_attention_layer_shared=False,
    cross_attention_widening_factor=2,
    num_self_attention_heads=4,
    num_self_attention_qk_channels=None,
    num_self_attention_v_channels=None,
    num_self_attention_layers_per_block=6,
    num_self_attention_blocks=1,
    first_self_attention_block_shared=True,
    self_attention_widening_factor=2,
    dropout=0.1)

output_adapter = ClassificationOutputAdapter(
    num_classes=50,
    num_output_queries=2048,
    num_output_query_channels=256)
decoder = PerceiverDecoder(
    output_adapter=output_adapter,
    num_latent_channels=256,  # D
    num_cross_attention_heads=4,
    num_cross_attention_qk_channels=256,
    num_cross_attention_v_channels=None,
    cross_attention_widening_factor=2,
    dropout=0.1)

model = PerceiverIO(encoder, decoder)

input = torch.randn(1,2048,3)
output = model(input)
print('output.shape:', output.shape)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total parameters:', pytorch_total_params)
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