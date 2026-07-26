import argparse
import copy
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from PIL import Image
import matplotlib.pyplot as plt

from cnn_dice_detection_model import DiceClassifier

LEARNING_RATE = 0.001
BATCH_SIZE = 32
PATIENCE = 15
IMG_SIZE = 64
CLASSES = 6
OUTPUT_DIR = "runs/tuning"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = {
    "baseline": {
        "dropout_rate": 0.5,
        "weight_decay": 1e-4,
        "augment": True,
        "adjust_augment": False,
    },
    "adjusted_augment": {
        "dropout_rate": 0.5,
        "weight_decay": 1e-4,
        "augment": False,
        "adjust_augment": True,
    },
    "no_dropout": {
        "dropout_rate": 0.0,
        "weight_decay": 1e-4,
        "augment": True,
        "adjust_augment": False,
    },
    "no_weight_decay": {
        "dropout_rate": 0.5,
        "weight_decay": 0.0,
        "augment": True,
        "adjust_augment": False,
    },
    "no_augmentation": {
        "dropout_rate": 0.5,
        "weight_decay": 1e-4,
        "augment": False,
        "adjust_augment": False,
    },
    "no_dropout_no_augmentation": {
        "dropout_rate": 0.0,
        "weight_decay": 1e-4,
        "augment": False,
        "adjust_augment": False,
    },
    "no_dropout_adjusted_augmentation": {
        "dropout_rate": 0.0,
        "weight_decay": 1e-4,
        "augment": False,
        "adjust_augment": True,
    },
}

class DiceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label in range(1, 7):
            class_dir = os.path.join(root_dir, str(label))
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, fname), label - 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# brightness: 0 black, 1 unchanged, 2 double
# contrast: 0 grey, 1 unchanged, 2 inc
# saturation: 0 black and white, 1 unchanged, 2 highly saturated
# hue: 0 - 0.5

AUGMENTED_TRANSFORM = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.RandomRotation(360),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5096, 0.4681, 0.3996], std=[0.2762, 0.2643, 0.2510])
])

ADJUSTED_AUGMENTED_TRANSFORM = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.RandomRotation(15),
    v2.ColorJitter(brightness=.4, contrast=0, saturation=2, hue=0),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5096, 0.4681, 0.3996], std=[0.2762, 0.2643, 0.2510])
])

PLAIN_TRANSFORM = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5096, 0.4681, 0.3996], std=[0.2762, 0.2643, 0.2510])
])

def train_one_config(config_name, config, epochs, train_loader, val_loader):
    print(f"\n{'=' * 55}")
    print(f"  Config: {config_name}  {config}")
    print(f"{'=' * 55}")

    model = DiceClassifier(
        classes=CLASSES,
        dropout_rate=config["dropout_rate"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, cooldown=2
    )

    best_val_loss = float('inf')
    best_model_weights = None
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"  Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}%")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"— no improvement for {PATIENCE} epochs")
                break

    return {
        "config": config,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "epochs_trained": len(train_losses),
    }

def plot_overlay(all_results, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10.colors

    for i, (name, result) in enumerate(all_results.items()):
        color = colors[i % len(colors)]
        ax1.plot(result["train_losses"], color=color, linestyle='-',
                 label=f"{name} (train)")
        ax1.plot(result["val_losses"], color=color, linestyle='--',
                 label=f"{name} (val)")

        ax2.plot(result["train_accuracies"], color=color, linestyle='-',
                 label=f"{name} (train)")
        ax2.plot(result["val_accuracies"], color=color, linestyle='--',
                 label=f"{name} (val)")

    ax1.set_title("Loss — solid=train, dashed=val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    ax2.set_title("Accuracy — solid=train, dashed=val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "tuning_overlay.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nTuning overlay plot saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Run RollCall overfitting tuning study"
    )
    parser.add_argument("--epochs", type=int, default=60,
                         help="Max epochs per config (early stopping still applies, default: 60)")
    parser.add_argument("--configs", type=str, nargs="+",
                         default=list(CONFIGS.keys()),
                         choices=list(CONFIGS.keys()),
                         help=f"Which configs to run (default: all — {list(CONFIGS.keys())})")
    args = parser.parse_args()

    print(f"Running on: {device}")

    val_dataset = DiceDataset("data/cropped/validation", transform=PLAIN_TRANSFORM)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_results = {}

    for config_name in args.configs:
        config = CONFIGS[config_name]
        if config["adjust_augment"]:
            train_transform = ADJUSTED_AUGMENTED_TRANSFORM
        elif config["augment"]:
            train_transform = AUGMENTED_TRANSFORM
        else:
            train_transform = AUGMENTED_TRANSFORM

        train_dataset = DiceDataset("data/cropped/train", transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        result = train_one_config(config_name, config, args.epochs, train_loader, val_loader)
        all_results[config_name] = result
        config_dir = os.path.join(OUTPUT_DIR, config_name)
        os.makedirs(config_dir, exist_ok=True)
        with open(os.path.join(config_dir, "metrics.json"), "w") as f:
            json.dump(result, f, indent=2)

    plot_overlay(all_results, OUTPUT_DIR)

    print(f"\n{'=' * 55}")
    print("  TEST SUMMARY")
    print(f"{'=' * 55}")
    print(f"  {'Config':<20} {'Best Val Loss':>15} {'Epochs Trained':>16}")
    for name, result in all_results.items():
        print(f"  {name:<20} {result['best_val_loss']:>15.4f} {result['epochs_trained']:>16}")

if __name__ == "__main__":
    main()