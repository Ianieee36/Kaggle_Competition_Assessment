"""Model definitions for the ENGE707 cat/dog facial feature classifier (report Task 3).

Two architectures:
  - BaselineCNN:         a small CNN trained from scratch, used as a baseline
  - ResNet50Classifier:  ResNet-50 pretrained on ImageNet (transfer learning), either with a
                         frozen backbone (only the new output layer trains) or fully fine-tuned

Both take a batch of 3 x 128 x 128 images and output 8 raw scores (logits), one per class.
No softmax is applied inside the models: the cross-entropy loss applies it internally,
and softmax is applied at prediction time to obtain class probabilities.
"""

import torch
from torch import nn
from torchvision import models as tv_models

NUM_CLASSES = 8


# ---------------------------------------------------------------------------
# Baseline CNN (trained from scratch)
# ---------------------------------------------------------------------------
def conv_block(in_ch, out_ch):
    """Two 3x3 convolutions, each followed by batch normalisation and ReLU, then 2x2 max pooling.
    Batch normalisation stabilises training; pooling halves the spatial size."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    """Four convolutional blocks (32 -> 64 -> 128 -> 256 channels), global average pooling,
    dropout and a linear output layer.

    Spatial size: 128 -> 64 -> 32 -> 16 -> 8, then pooled to 1 x 1.
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)          # one value per channel, any input size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),                     # randomly drops features to reduce overfitting
            nn.Linear(256, num_classes),             # output layer: 8 logits
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# ---------------------------------------------------------------------------
# ResNet-50 (transfer learning)
# ---------------------------------------------------------------------------
class ResNet50Classifier(nn.Module):
    """ResNet-50 pretrained on ImageNet with its 1000-class output layer replaced by an 8-class one.

    freeze_backbone=True:  only the new output layer is trained (feature extraction)
    freeze_backbone=False: every layer is trained (fine-tuning)
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False, dropout=0.2):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.net = tv_models.resnet50(weights=weights)
        in_features = self.net.fc.in_features        # 2048 features from the backbone
        self.net.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),     # new output layer: 8 logits
        )
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for name, p in self.net.named_parameters():
                p.requires_grad = name.startswith("fc.")

    def train(self, mode=True):
        """When the backbone is frozen, its batch-normalisation layers stay in evaluation mode,
        so their ImageNet statistics are not overwritten during training."""
        super().train(mode)
        if self.freeze_backbone and mode:
            for name, m in self.net.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_model(name, pretrained=True):
    """Builds a model by name: 'baseline_cnn', 'resnet50_frozen' or 'resnet50_finetune'."""
    if name == "baseline_cnn":
        return BaselineCNN()
    if name == "resnet50_frozen":
        return ResNet50Classifier(pretrained=pretrained, freeze_backbone=True)
    if name == "resnet50_finetune":
        return ResNet50Classifier(pretrained=pretrained, freeze_backbone=False)
    raise ValueError(f"Unknown model name: {name}")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
