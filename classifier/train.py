import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from PIL import Image
import os
import json
from cnn_dice_detection_model import DiceClassifier
import matplotlib.pyplot as plt
from pathlib import Path

CURRENT_DIR = Path.cwd()
TRAINING_DATA_DIR = CURRENT_DIR / "data/cropped/train" # Directory where training data is stored. Should correlate to the value from crop_script.py
VALIDATION_DATA_DIR = CURRENT_DIR / "data/cropped/validation" # Directory where validation data is stored. Should correlate to the value from crop_script.py
RUN_DATA_DIR = CURRENT_DIR / "runs" # Directory to store models, outputs, plots etc for the runs of different models
CURR_MODEL_CLASS_DIR = RUN_DATA_DIR / "custom" # Directory to store data and plots from the custom model
CURRENT_RUN_DIR = CURR_MODEL_CLASS_DIR / "run_20" # Directory to break data into runs
PARAMETERS_FILE = CURRENT_RUN_DIR / "hyperparameters.txt"
BEST_MODEL_FILE = CURRENT_RUN_DIR / "best_model.pt"
METRICS_FILE = CURRENT_RUN_DIR / "metrics.json"
TRAINING_CURVES_FILE = CURRENT_RUN_DIR / "training_curves.png"

# Hyperparameters
LEARNING_RATE = 0.001 # Tune if saw pattern
BATCH_SIZE = 32 # Lower if memory becomes an issue.
EPOCHS = 200 # High epoch ceiling to allow early stopping
DROPOUT_RATE = 0.5 # If under-fitting -> lower. If overfitting -> increase
PATIENCE = 15 # Patients for early stopping
IMG_SIZE = 64 # Size of input cropped image. Should correlate to the value from crop_script.py
NUM_CLASSES = 6 # Configured classes of dice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

os.makedirs(RUN_DATA_DIR, exist_ok=True)
os.makedirs(CURR_MODEL_CLASS_DIR, exist_ok=True)
os.makedirs(CURRENT_RUN_DIR, exist_ok=True)

with open(PARAMETERS_FILE, "w") as file:
    file.write(
        "Initial Hyperparameters\n"
        f"Learning Rate:.......{LEARNING_RATE}\n"
        f"Batch Size:..........{BATCH_SIZE}\n"
        f"Epochs:..............{EPOCHS}\n"
        f"Dropout Rate:........{DROPOUT_RATE}\n"
        f"Patients:............{PATIENCE}\n"
        f"Image Size:..........{IMG_SIZE}\n"
        f"Number of classes:...{NUM_CLASSES}\n"
    )

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
                    self.samples.append((
                        os.path.join(class_dir, fname),
                        label - 1
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB') # Guard against non-RGB images that would break the pipeline
        if self.transform:
            image = self.transform(image)
        return image, label

#
training_data_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.RandomRotation(15),     # Add rotation invariance. Allows 360 degrees of rotation of original image
    v2.ColorJitter(            # Add lighting variation simulations
        brightness=0.4,
        contrast=0,
        saturation=2,
        hue=0
    ),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean = [0.5096, 0.4681, 0.3996],
        std  = [0.2762, 0.2643, 0.2510]
    )
])

validation_data_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean = [0.5096, 0.4681, 0.3996],
        std  = [0.2762, 0.2643, 0.2510]
    )
])

training_dataset = DiceDataset(TRAINING_DATA_DIR, transform=training_data_transform)
validation_dataset = DiceDataset(VALIDATION_DATA_DIR, transform=validation_data_transform)

training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(training_loader)}")
print(f"Validation samples: {len(validation_loader)}")

model = DiceClassifier(classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, cooldown=2)

best_validation_loss = float('inf')
best_validation_loss_accuracy = 0.0
best_validation_loss_training_loss = 0.0
best_validation_loss_training_accuracy = 0.0
best_model_weights = None
epochs_no_improve = 0

training_losses = []
validation_losses = []
training_accuracies = []
validation_accuracies = []

assert PATIENCE > 2 * scheduler.patience + scheduler.cooldown, "Early stopping patience too low - will interfere with LR scheduler"

for epoch in range(EPOCHS):
# ========== Training ==========
    model.train()
    training_loss = 0.0
    training_correct = 0
    training_total = 0

    for images, labels in training_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        _, predicted = outputs.max(1)
        training_total += labels.size(0)
        training_correct += predicted.eq(labels).sum().item()

    avg_training_loss = training_loss / len(training_loader)
    training_accuracy = 100. * training_correct / training_total
    training_losses.append(avg_training_loss)
    training_accuracies.append(training_accuracy)

# ========== Validation  ==========
    model.eval()
    validation_loss = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            _, predicted = outputs.max(1)
            validation_total += labels.size(0)
            validation_correct += predicted.eq(labels).sum().item()

    avg_validation_loss = validation_loss / len(validation_loader)
    validation_accuracy = 100. * validation_correct / validation_total

    validation_losses.append(avg_validation_loss)
    validation_accuracies.append(validation_accuracy)

    print(f"Epoch [{epoch+1} of {EPOCHS}] "
          f"Training Loss: {avg_training_loss:.4f} Accuracy: {training_accuracy:.2f}% | "
          f"Validation Loss: {avg_validation_loss:.4f} Accuracy: {validation_accuracy:.2f}%")

    scheduler.step(avg_validation_loss)

    if avg_validation_loss < best_validation_loss:
        best_validation_loss = avg_validation_loss
        best_validation_loss_accuracy = validation_accuracy
        best_validation_loss_training_loss = avg_training_loss
        best_validation_loss_training_accuracy = training_accuracy
        best_model_weights = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        torch.save(best_model_weights, BEST_MODEL_FILE)
        print(f"  New best model saved  at {BEST_MODEL_FILE} (val loss: {best_validation_loss:.4f})")


    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEARLY STOPPING at epoch {epoch+1}, no improvement for {PATIENCE} epochs")
            break

model.load_state_dict(best_model_weights)
print(f"\nTraining complete. Best validation loss: {best_validation_loss:.4f}")

with open(PARAMETERS_FILE, "a") as file:
    file.write(
        "\nBest Model Metrics\n"
        f"Validation Loss:.......{best_validation_loss}\n"
        f"Validation Accuracy:...{best_validation_loss_accuracy}\n"
        f"Training Loss:.........{best_validation_loss_training_loss}\n"
        f"Training Accuracy:.....{best_validation_loss_training_accuracy}\n"
    )
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(training_losses, label='Training Loss')
ax1.plot(validation_losses, label='Validation Loss')
ax1.set_title('Loss Curves')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(training_accuracies, label='Training Accuracy')
ax2.plot(validation_accuracies, label='Validation Accuracy')
ax2.set_title('Accuracy Curves')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True)

plt.savefig(TRAINING_CURVES_FILE, dpi=150, bbox_inches='tight')
plt.close()
print(f"Training curves saved to {TRAINING_CURVES_FILE}")

metrics = {
    "training_losses": training_losses,
    "validation_losses": validation_losses,
    "training_accuracies": training_accuracies,
    "validation_accuracies": validation_accuracies,
    "best_validation_loss": best_validation_loss,
    "epochs_trained": len(training_losses)
}
with open(METRICS_FILE, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {METRICS_FILE}")