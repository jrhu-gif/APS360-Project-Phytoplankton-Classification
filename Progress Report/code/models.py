"""Models: HAB-Net (own two-stream design) and the ResNet-18 baseline (B2)."""
import torch
import torch.nn as nn


def conv_stage(c_in, c_out):
    """[Conv3x3 - BN - ReLU] x 2 + MaxPool 2x2 (halves H and W)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class HABNet(nn.Module):
    """
    Image stream : 1x96x256 -> 32x48x128 -> 64x24x64 -> 128x12x32 -> 256x6x16 -> GAP -> 256
    Morph stream : 8 -> 32 -> 32                       (absolute size/shape from original image)
    Fusion       : concat 288 -> FC 128 -> ReLU -> Dropout 0.3
    Heads        : species (38 logits) + HAB (1 logit)
    """
    def __init__(self, n_classes=38, n_morph=8, morph=True, hab_head=True, pixel_mean=0.5, pixel_std=0.25):
        super().__init__()
        self.register_buffer("px", torch.tensor([pixel_mean, pixel_std]))   # train-set normalization
        self.image = nn.Sequential(conv_stage(1, 32), conv_stage(32, 64), conv_stage(64, 128),
                                   conv_stage(128, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.morph = nn.Sequential(nn.Linear(n_morph, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
                                   nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True)) if morph else None
        self.fuse = nn.Sequential(nn.Linear(256 + (32 if morph else 0), 128), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.species = nn.Linear(128, n_classes)
        self.hab = nn.Linear(128, 1) if hab_head else None

    def forward(self, x, m=None):          # x: raw [0,1] grayscale, 1 channel
        z = self.image((x - self.px[0]) / self.px[1])
        if self.morph is not None:
            z = torch.cat([z, self.morph(m)], 1)
        z = self.fuse(z)
        return self.species(z), (self.hab(z).squeeze(1) if self.hab is not None else None)


class ResNet18(nn.Module):
    """B2: ImageNet-pretrained ResNet-18, all layers fine-tuned (as in Kraft et al.)."""
    def __init__(self, n_classes=38):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Linear(512, n_classes)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, m=None):          # x: raw [0,1] grayscale, 1 channel
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        return self.net(x), None


def build_model(cfg, n_classes, n_morph, pixel_mean, pixel_std):
    if cfg["model"] == "resnet18":
        return ResNet18(n_classes)
    return HABNet(n_classes, n_morph, cfg["morph"], cfg["hab_head"], pixel_mean, pixel_std)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
