import torch
from utils.data_loader import load_denoising_test_mix
from torchvision import transforms
from utils.data_loader import fluore_to_tensor

transform = transforms.Compose([
    transforms.FiveCrop(256),
    transforms.Lambda(
        lambda crops: torch.stack([
            fluore_to_tensor(crop) for crop in crops[:4]
        ])
    ),
    transforms.Lambda(
        lambda x: x.float().div(255).sub(0.5)
    )
])

loader = load_denoising_test_mix(
    root="./dataset",
    batch_size=2,
    noise_levels=[1],
    transform=transform,
    patch_size=256
)

print("Number of test samples:", len(loader.dataset))

noisy, clean = next(iter(loader))

print("Noisy shape:", noisy.shape)
print("Clean shape:", clean.shape)
print("Noisy dtype:", noisy.dtype)
print("Clean dtype:", clean.dtype)
print("Noisy range:", noisy.min().item(), noisy.max().item())
print("Clean range:", clean.min().item(), clean.max().item())