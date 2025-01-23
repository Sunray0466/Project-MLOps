import torch
from torch import nn
from transformers import ResNetForImageClassification


class ModelConvolution(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.3),
        )
        self.flat = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear((224 // 2**3) ** 2 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 53),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


def hugging_face_resnet():
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50", num_labels=53, ignore_mismatched_sizes=True
    )
    return model


if __name__ == "__main__":
    # model = ModelConvolution()
    # print(f"Model architecture: {model}")
    # print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    hugging_face_resnet()
    # dummy_input = torch.randn(1, 1, 224, 224)
    # output = model(dummy_input)
    # print(f"Output shape: {output.shape}")
