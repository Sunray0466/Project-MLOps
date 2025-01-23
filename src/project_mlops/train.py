import logging
import os

import hydra
import matplotlib.pyplot as plt
import torch
import hydra
import os
from data import playing_cards
from model import model_list
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(config_name="hyperparams.yaml", config_path=f"{os.getcwd()}/configs")
def train(chp) -> None:
    """Train a model on playing cards."""
    # var
    model_type  = chp.model
    batch_size  = chp.batch_size
    lr          = chp.get("lr", chp.default[model_type].lr)
    epochs      = chp.epochs
    seed        = chp.seed
    project_dir = hydra.utils.get_original_cwd()
    log = logging.getLogger(__name__)
    log.info(f"{model_type=}, {batch_size=}, {lr=}, {epochs=}, {seed=} {project_dir=}")
    
    # model/data
    print(DEVICE)
    torch.manual_seed(seed)
    model,pred_func = model_list(model_type)
    model = model.to(DEVICE)
    train_set, valid_set, _ = playing_cards(project_dir)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    statistics = {"train_loss": [], "train_accuracy": [], "valid_loss": [], "valid_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = pred_func(model(img))
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if (i + 1) % 100 == 0:
                model.eval()
                valid_l = []
                valid_a = []
                with torch.no_grad():
                    for img, target in valid_dataloader:
                        img, target = img.to(DEVICE), target.to(DEVICE)
                        y_pred = pred_func(model(img))
                        vloss = loss_fn(y_pred, target)

                        accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
                        valid_l.append(vloss.item())
                        valid_a.append(accuracy)
                    valid_loss = sum(valid_l) / len(valid_l)
                    valid_accuracy = sum(valid_a) / len(valid_a)
                    statistics["valid_loss"].append(valid_loss)
                    statistics["valid_accuracy"].append(valid_accuracy)
                log.info(
                    f"Epoch {epoch:>2}, iter {i + 1:>4}, train-loss: {loss.item():.4f}, valid-loss: {valid_loss:.4f}, valid-accuracy: {valid_accuracy * 100:.2f}%"
                )
                model.train()

    log.info("Training completed")
    # save model + score
    prefix = os.getcwd().split("outputs\\")[-1].replace("\\","_") # yyyy-mm-dd_hh-mm-ss
    model_save_path = f"{project_dir}/models/{model_type}_{prefix}.pth"
    score_save_path = f"{os.getcwd()}/training_{prefix}.png"
    
    torch.save(model.state_dict(), model_save_path) # model_{prefix}.pth
    fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    axs = axs.flat
    axs[0].plot(statistics["train_loss"])
    axs[1].plot(statistics["train_accuracy"])
    axs[2].plot(statistics["valid_loss"])
    axs[3].plot(statistics["valid_accuracy"])
    axs[0].set_title("Train loss")
    axs[1].set_title("Train accuracy")
    axs[2].set_title("Valid loss")
    axs[3].set_title("Valid accuracy")
    fig.savefig(score_save_path) # training_{prefix}.pth
    print(f"      Model saved to: {model_save_path}")
    print(f"Performance saved to: {score_save_path}")


if __name__ == "__main__":
    train()
