from torch import nn
import timm
import torch
# from transformers import ResNetForImageClassification


class CNN(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.3),
        )
        self.flat = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear((224 // 2**3) ** 2 * 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 53),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


# def resnet50():
#     model = ResNetForImageClassification.from_pretrained(
#         "microsoft/resnet-50", num_labels=53, ignore_mismatched_sizes=True
#     )
#     # freeze all layers except the last 10
#     for param in list(model.parameters())[:-6]:
#         param.requires_grad = False
#     return model
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



def model_list(model_type: str = "cnn"):
    "Returns given model as well as a function to extract the model prediction"
    model_type = model_type.lower()
    model_options = {"cnn": (CNN, lambda x: x), "resnet50": (resnet50, lambda x: x.logits)}
    # Model type check
    str_sep = lambda x: ("\n\t- " + str(x))
    if model_type not in model_options:
        raise Exception(
            f"[Model error]\n\tInvalid model type ({model_type})\n\tAvailable models:{''.join(map(str_sep, model_options))}"
        )
    # get model
    model, pred_func = model_options[model_type]
    return model(), pred_func


if __name__ == "__main__":
    model = PretrainedResNet()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

