import logging
import os
import sys

# import logging
from datetime import datetime

import hydra
import matplotlib.pyplot as plt
import omegaconf
import torch
import typer
import utils
from loguru import logger as log
from model import model_list
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score

import wandb
from data import playing_cards

# Replace underscores with dashes in CLI arguments
sys.argv = [arg.replace("_", "-") if "--" in arg else arg for arg in sys.argv]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(config_name="hyperparams.yaml", config_path=f"{os.getcwd()}/configs", version_base="1.1")
def train(cfg) -> None:
    """Train a model on playing cards."""
    # var
    model_type = cfg.model
    batch_size = cfg.batch_size
    lr = cfg.lr if cfg.lr != None else cfg.default[model_type].lr
    epochs = cfg.epochs
    seed = cfg.seed
    project_dir = "/gcs/dtumlops-bucket-group35"  # hydra.utils.get_original_cwd()

    log = logging.getLogger(__name__)
    log.info(f"{model_type=}, {batch_size=}, {lr=}, {epochs=}, {seed=} {project_dir=}")
    wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(project="playing_cards", settings=wandb.Settings(start_method="thread"))

    # model/data
    log.info(f"Using Device: {DEVICE}")
    torch.manual_seed(seed)
    model, pred_func = model_list(model_type)
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

        preds, targets = [], []

        # Training loop
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

            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

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
                # add a plot of the input images
                images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                wandb.log({"images": images})

                # add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads.cpu())})

                model.train()

        # add a custom matplotlib plot of the ROC curves
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)
        fig, axes = plt.subplots(4, 13, figsize=(24, 18))
        axes = list(axes.flat)
        axes.append(fig.add_subplot(5, 6, 1))

        for class_id in range(53):
            one_hot = torch.zeros_like(targets)
            one_hot[targets == class_id] = 1
            _ = RocCurveDisplay.from_predictions(
                one_hot.cpu().numpy(),
                preds[:, class_id].cpu().numpy(),
                name=f"ROC curve for {class_id}",
                plot_chance_level=(class_id == 2),
                ax=axes[class_id],
            )
            axes[class_id].axis("off")

        # alternatively use wandb.log({"roc": wandb.Image(plt)}
        wandb.log({"roc": wandb.Image(plt)})
        plt.close()  # close the plot to avoid memory leaks and overlapping figures

    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    log.info("Training completed")
    # Get the current date and time and format to string
    current_datetime = datetime.now()
    prefix = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = f"{project_dir}/models/{model_type}_{prefix}.pth"
    score_save_path = f"{os.getcwd()}/training_{prefix}.png"

    torch.save(model.state_dict(), model_save_path)  # model_{prefix}.pth

    artifact = wandb.Artifact(
        name="playing_cards_model",
        type="model",
        description="A model trained to classify playing cards.",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file(model_save_path)
    run.log_artifact(artifact)

    for vl,va in zip(statistics["valid_loss"],statistics["valid_accuracy"]):
        wandb.log({"valid_loss": vl, "valid_accuracy": va})

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
    fig.savefig(score_save_path)  # training_{prefix}.pth
    print(f"      Model saved to: {model_save_path}")
    print(f"Performance saved to: {score_save_path}")


if __name__ == "__main__":
    train()
    # typer.run(train)
