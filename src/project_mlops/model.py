import torch
from torch import nn
import timm

class PretrainedResNet(nn.Module):
    """My awesome model."""

    def __init__(self, num_classes = 53) -> None:
        super(PretrainedResNet, self).__init__()
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)

        # Freeze earlier layers (layer1 and layer2)
        for param in self.model.layer1.parameters():
            param.requires_grad = False
        for param in self.model.layer2.parameters():
            param.requires_grad = False

        # Fine-tune layer3, layer4, and fc layer
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Change the number of output classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


if __name__ == "__main__":
    model = PretrainedResNet()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")