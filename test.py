import torch

from perceiver.model.core import PerceiverEncoder

from perceiver.model.image import ImageInputAdapter

from perceiver.model.pointcloud import PointCloudInputAdapter


# ------ Image
# Fourier-encodes pixel positions and flatten along spatial dimensions
input_adapter = ImageInputAdapter(
  image_shape=(224, 224, 3),  # M = 224 * 224
  num_frequency_bands=64,
)

# Generic Perceiver encoder
encoder = PerceiverEncoder(
  input_adapter=input_adapter,
  num_latents=512,  # N
  num_latent_channels=256,  # D
  num_cross_attention_qk_channels=input_adapter.num_input_channels,  # C
  num_cross_attention_heads=1,
  num_self_attention_heads=8,
  num_self_attention_layers_per_block=6,
  num_self_attention_blocks=8,
  dropout=0.0,
)

# 怎么输入数据呢，其实就传个 (batch_size, 224,224,3) 的张量就行吧
input = torch.randn(2, 224, 224, 3)
# output: [2, 512, 512]
output = encoder(input)
# results: [2, 512]
results = output.mean(1)
print(results.shape)

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
# results_pc: [2, 512]
results_pc = output_pc.mean(1)
print(results_pc.shape)