_base_ = ['./minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti.py']

# 2025-02-16 Jinzheng Guang 消融实验，更少计算量
# configs/minkunet/minkunet18_w16_torchsparse_8xb2-amp-15e_semantickitti.py
# model = dict(
#     backbone=dict(
#         base_channels=16,
#         encoder_channels=[16, 32, 64, 128],
#         decoder_channels=[128, 64, 48, 48]),
#     decode_head=dict(channels=48))


# NOTE: Due to TorchSparse backend, the model performance is relatively
# dependent on random seeds, and if random seeds are not specified the
# model performance will be different (± 1.5 mIoU).
# randomness = dict(seed=1588147245)


train_dataloader = dict(
    batch_size=5,
    num_workers=5)


test_dataloader = dict(
    batch_size=1,
    num_workers=4)


