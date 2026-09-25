import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models.unet import UnetN2N
from utils.data_loader import (
    DenoisingFolderN2N,
    fluore_to_tensor,
)


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = torch.device("cpu")


# --------------------------------------------------
# 2. Original four-crop preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.FiveCrop(256),

    transforms.Lambda(
        lambda crops: torch.stack([
            fluore_to_tensor(crop)
            for crop in crops[:4]
        ])
    ),

    transforms.Lambda(
        lambda x: x.float().div(255).sub(0.5)
    ),
])


# --------------------------------------------------
# 3. Dataset
# --------------------------------------------------

dataset = DenoisingFolderN2N(
    root="./dataset",
    noise_levels=[1, 2, 4, 8, 16],
    types=["Confocal_BPAE_B"],
    test_fov=19,
    captures=2,
    transform=transform,
    target_transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
)


# --------------------------------------------------
# 4. Model, optimizer, and loss
# --------------------------------------------------

model = UnetN2N(1, 1).to(device)
model.train()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)

criterion = torch.nn.MSELoss()


# --------------------------------------------------
# 5. Load one real batch
# --------------------------------------------------

noisy_input, noisy_target, ground_truth = next(iter(loader))

print("Input batch shape:", noisy_input.shape)
print("Target batch shape:", noisy_target.shape)
print("Ground-truth batch shape:", ground_truth.shape)
print("Input dtype:", noisy_input.dtype)
print("Input range:", noisy_input.min().item(), noisy_input.max().item())


# --------------------------------------------------
# 6. Four-crop batch preparation
# --------------------------------------------------

# Dataset batch shape:
# [batch_size, 4, 1, 256, 256]

# Merge batch and crop dimensions:
# [batch_size * 4, 1, 256, 256]

noisy_input = noisy_input.flatten(0, 1)
noisy_target = noisy_target.flatten(0, 1)

noisy_input = noisy_input.to(device)
noisy_target = noisy_target.to(device)


# --------------------------------------------------
# 7. Genuine training step
# --------------------------------------------------

optimizer.zero_grad()

prediction = model(noisy_input)

loss = criterion(prediction, noisy_target)

loss.backward()

optimizer.step()


# --------------------------------------------------
# 8. Report
# --------------------------------------------------

gradient_count = sum(
    1 for parameter in model.parameters()
    if parameter.grad is not None
)

print("Prediction shape:", prediction.shape)
print("Loss:", loss.item())
print("Parameters with gradients:", gradient_count)
print("Real-data training step: SUCCESS")