import io

import torch
import typer
from model import model_list
from PIL import Image
from torchvision.transforms import Compose, PILToTensor, Resize

from data import normalize, playing_cards

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model.
    Input example: model_yyyy-mm-dd_hh-mm-ss.pth
    """
    print(model_checkpoint)
    model_type = model_checkpoint.split("_")[0]

    model, pred_func = model_list(model_type)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load("models/" + model_checkpoint, weights_only=True))

    *_, test_set = playing_cards(".")
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = pred_func(model(img))
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total * 100:.4f}")


def predict_input(model_checkpoint: str, img_tensor: torch.Tensor) -> list[str]:
    """Evaluate a trained model on input image
    Input example: model_yyyy-mm-dd_hh-mm-ss.pth
    """
    print(model_checkpoint)
    model_type = model_checkpoint.split("_")[0]

    model, pred_func = model_list(model_type)
    model = model.to(DEVICE)
    img_tensor = img_tensor.to(DEVICE)
    model.load_state_dict(torch.load("models/" + model_checkpoint, weights_only=True))
    idx2labels = torch.load(f"data/processed/cards-dataset/label_converter.pt", weights_only=True)

    model.eval()
    label_indexes = pred_func(model(img_tensor)).argmax(dim=1)

    labels = []
    for label_idx in label_indexes:
        labels.append(idx2labels[label_idx.item()])
    return labels


def predict_bimg(model_checkpoint: str, byte_img: str) -> list[str]:
    """Evaluate a trained model on input "byte" image
    Input example: model_yyyy-mm-dd_hh-mm-ss.pth
    """
    img = Image.open(io.BytesIO(byte_img))
    transform = Compose(
        [
            PILToTensor(),
            Resize((224, 224)),
        ]
    )
    img = torch.unsqueeze(normalize(transform(img).float()), 0)
    return predict_input(model_checkpoint, img)
    # import matplotlib.pyplot as plt
    # plt.imshow(img.permute(1,2,0).numpy())
    # plt.show()


if __name__ == "__main__":
    typer.run(evaluate)
    # print(predict_input("cnn_2025-01-21_22-10-09.pth", img))
