import torch

from perceiver.model.core import PerceiverEncoder

from perceiver.model.image import ImageInputAdapter

from perceiver.model.pointcloud import PointCloudInputAdapter


# # ------ Image
# # Fourier-encodes pixel positions and flatten along spatial dimensions
# input_adapter = ImageInputAdapter(
#   image_shape=(224, 224, 3),  # M = 224 * 224
#   num_frequency_bands=64,
# )

# # Generic Perceiver encoder
# encoder = PerceiverEncoder(
#   input_adapter=input_adapter,
#   num_latents=512,  # N
#   num_latent_channels=256,  # D
#   num_cross_attention_qk_channels=input_adapter.num_input_channels,  # C
#   num_cross_attention_heads=1,
#   num_self_attention_heads=8,
#   num_self_attention_layers_per_block=6,
#   num_self_attention_blocks=8,
#   dropout=0.0,
# )

# input = torch.randn(2, 224, 224, 3)
# output: [2, 512, 512]
# output = encoder(input)
# print(output.shape)

# from torch.utils.data import DataLoader
# from datasets.data import ShapeNetRender
# from utils import transform

# ------ test PointCloud dataloader
# train_set = ShapeNetRender(transform)
# train_loader = DataLoader(
#         train_set,
#         batch_size=4,
#         shuffle=True,
#         num_workers=0,
#         pin_memory=True,
#         drop_last=False
#     )

# for i, ((pc_t1, pc_t2), imgs) in enumerate(train_loader):
#     batch_size = pc_t1.shape[0]
#     pc_t1, pc_t2, imgs = pc_t1.to(0), pc_t2.to(0), imgs.to(0)
#     imgs = torch.permute(imgs, (0, 2, 3, 1))
#     print('batch_size:', batch_size)
#     print('pc_t1.shape:', pc_t1.shape)
#     print('pc_t2.shape:', pc_t2.shape)
#     print('imgs.shape:', imgs.shape)
#     break

# ------ PointCloud
input_adapter_pc = PointCloudInputAdapter(
  pointcloud_shape=(2048, 3),
  num_input_channels=256,
  num_groups=128,
  group_size=32
)

encoder_pc = PerceiverEncoder(
  input_adapter=input_adapter_pc,
  num_latents=128,  # N
  num_latent_channels=256,  # D
  num_cross_attention_qk_channels=input_adapter_pc.num_input_channels,  # C
  num_cross_attention_heads=1,
  num_self_attention_heads=8,
  num_self_attention_layers_per_block=6,
  num_self_attention_blocks=8,
  dropout=0.0,
)

input_pc = torch.randn(2, 2048, 3)
# output_pc: [2, 128, 512]
output_pc = encoder_pc(input_pc)
print(output_pc.shape)

pytorch_total_params = sum(p.numel() for p in encoder_pc.parameters() if p.requires_grad)
print('total parameters:', pytorch_total_params)
