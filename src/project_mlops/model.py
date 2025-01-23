import timm
import torch
from torch import nn
from transformers import ResNetForImageClassification

class CNN(nn.Module):
    """My awesome model."""

    def __init__(self, num_classes=53) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(24),
            nn.Dropout(p=0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 36, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(36, 36, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(36),
            nn.Dropout(p=0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(36, 48, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48),
            nn.Dropout(p=0.3),
        )
        self.flat = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear((224 // 2**3) ** 2 * 48, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


def HuggingfaceResnet(num_classes=53):
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    # freeze all layers except the last 10
    for param in list(model.parameters())[:1]:
        param.requires_grad = False
    model.classifier.append(nn.Linear(1000, num_classes))
    return model

def TimmResNet(num_classes=53):
    model = timm.create_model("resnet18", pretrained=True, num_classes=num_classes)
    # Freeze earlier layers (layer1 and layer2)
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    # Fine-tune layer3, layer4, and fc layer
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Change the number of output classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def model_list(model_type: str = "cnn"):
    "Returns given model as well as a function to extract the model prediction"
    assert model_type == model_type.lower(), "Error: model type is not lowercase!"
    model_options = {
        "cnn": (CNN, lambda x: x), 
        "huggingresnet": (HuggingfaceResnet, lambda x: x.logits),
        "timmresnet": (TimmResNet, lambda x: x),
        }
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
    model = HuggingfaceResnet()
    # model = TimmResNet()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
