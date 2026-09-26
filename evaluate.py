
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\USER\Desktop\cnn_image\mammals_clean"
MODEL_PATH = r"C:\Users\USER\Desktop\cnn_image\best_vgg7.pth"
OUTPUT_DIR = r"C:\Users\USER\Desktop\cnn_image\evaluation_results"

IMAGE_SIZE = 224
BATCH_SIZE = 64
VAL_SPLIT = 0.20
RANDOM_SEED = 42
NUM_WORKERS = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("VGG7 MODEL EVALUATION")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============================================================
# 4. CHECK FILES
# ============================================================

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model checkpoint not found:\n{MODEL_PATH}"
    )

# ============================================================
# 5. LOAD CHECKPOINT
# ============================================================

print("\nLoading trained model...")

# Only load checkpoints you trust, since this uses
# weights_only=False to load the saved class-name list.
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

classes = checkpoint["classes"]
NUM_CLASSES = checkpoint["num_classes"]

print("Number of classes:", NUM_CLASSES)
print("Best validation accuracy during training:",
      f'{checkpoint["best_val_accuracy"]:.2f}%')
print("Best epoch:", checkpoint["epoch"])

if len(classes) != NUM_CLASSES:
    raise ValueError(
        "Checkpoint class count does not match."
    )

# ============================================================
# 6. DEFINE SAME VGG7 ARCHITECTURE
# ============================================================

class VGG7(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 7th convolution
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================
# 7. CREATE MODEL AND LOAD WEIGHTS
# ============================================================

model = VGG7(NUM_CLASSES).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nModel weights loaded successfully.")

# ============================================================
# 8. VALIDATION TRANSFORM
# Must match train.py validation transform
# ============================================================

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# 9. COLLECT DATASET
# Use checkpoint's exact class order
# ============================================================

image_extensions = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
)

all_samples = []

for label, class_name in enumerate(classes):

    class_folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.isdir(class_folder):
        raise FileNotFoundError(
            f"Class folder missing: {class_folder}"
        )

    for filename in sorted(os.listdir(class_folder)):

        if filename.lower().endswith(image_extensions):

            image_path = os.path.join(
                class_folder,
                filename
            )

            all_samples.append(
                (image_path, label)
            )

print("\nTotal images:", len(all_samples))
print("Total classes:", len(classes))

# ============================================================
# 10. RECREATE STRATIFIED SPLIT
# Must match train.py exactly
# ============================================================

def stratified_split(samples, val_split, seed=42):

    rng = np.random.default_rng(seed)

    class_samples = {}

    for path, label in samples:
        class_samples.setdefault(label, []).append(
            (path, label)
        )

    train_samples = []
    val_samples = []

    for label in sorted(class_samples.keys()):

        items = class_samples[label].copy()
        rng.shuffle(items)

        val_count = max(
            1,
            int(len(items) * val_split)
        )

        val_samples.extend(items[:val_count])
        train_samples.extend(items[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples


train_samples, val_samples = stratified_split(
    all_samples,
    VAL_SPLIT,
    RANDOM_SEED
)

print("\nTrain images:", len(train_samples))
print("Validation images:", len(val_samples))

# ============================================================
# 11. VALIDATION DATASET
# ============================================================

class AnimalDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


val_dataset = AnimalDataset(
    val_samples,
    transform=val_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ============================================================
# 12. EVALUATION
# ============================================================

print("\nEvaluating model...")

criterion = nn.CrossEntropyLoss()

all_predictions = []
all_labels = []
total_loss = 0.0
total_correct = 0
total_images = 0

use_amp = DEVICE.type == "cuda"

with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(val_loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=use_amp
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_images += labels.size(0)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        if (batch_idx + 1) % 10 == 0:
            print(
                f"Processed {total_images}/"
                f"{len(val_dataset)} images"
            )

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

val_loss = total_loss / total_images
accuracy = 100.0 * total_correct / total_images

# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

confusion = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)

for true_label, predicted_label in zip(
    all_labels,
    all_predictions
):
    confusion[true_label, predicted_label] += 1

# ============================================================
# 14. PER-CLASS METRICS
# ============================================================

print("\nCalculating per-class metrics...")

class_results = []

for i, class_name in enumerate(classes):

    true_positive = confusion[i, i]

    actual_count = confusion[i, :].sum()
    predicted_count = confusion[:, i].sum()

    precision = (
        true_positive / predicted_count
        if predicted_count > 0 else 0.0
    )

    recall = (
        true_positive / actual_count
        if actual_count > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    class_accuracy = (
        100.0 * true_positive / actual_count
        if actual_count > 0 else 0.0
    )

    class_results.append({
        "class": class_name,
        "images": int(actual_count),
        "correct": int(true_positive),
        "accuracy": class_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

macro_precision = np.mean([
    item["precision"] for item in class_results
])

macro_recall = np.mean([
    item["recall"] for item in class_results
])

macro_f1 = np.mean([
    item["f1"] for item in class_results
])

# ============================================================
# 15. PRINT OVERALL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL EVALUATION RESULTS")
print("=" * 70)

print(f"Validation images: {total_images}")
print(f"Validation loss:   {val_loss:.4f}")
print(f"Accuracy:          {accuracy:.2f}%")
print(f"Macro precision:   {macro_precision:.4f}")
print(f"Macro recall:      {macro_recall:.4f}")
print(f"Macro F1-score:    {macro_f1:.4f}")

# ============================================================
# 16. PRINT PER-CLASS RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("PER-CLASS PERFORMANCE")
print("=" * 100)

print(
    f"{'Class':<22}"
    f"{'Images':>9}"
    f"{'Correct':>10}"
    f"{'Accuracy':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>10}"
    f"{'F1':>10}"
)

print("-" * 100)

for item in class_results:

    print(
        f"{item['class']:<22}"
        f"{item['images']:>9}"
        f"{item['correct']:>10}"
        f"{item['accuracy']:>11.2f}%"
        f"{item['precision']:>12.4f}"
        f"{item['recall']:>10.4f}"
        f"{item['f1']:>10.4f}"
    )

# ============================================================
# 17. SAVE PER-CLASS CSV
# ============================================================

import csv

csv_path = os.path.join(
    OUTPUT_DIR,
    "per_class_metrics.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "class",
            "images",
            "correct",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )

    writer.writeheader()
    writer.writerows(class_results)

print("\nPer-class metrics saved:")
print(csv_path)

# ============================================================
# 18. SAVE CONFUSION MATRIX
# ============================================================

confusion_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

fig, ax = plt.subplots(figsize=(22, 20))

image = ax.imshow(
    confusion,
    interpolation="nearest",
    aspect="auto"
)

ax.set_title("VGG7 Confusion Matrix")
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")

ax.set_xticks(np.arange(NUM_CLASSES))
ax.set_yticks(np.arange(NUM_CLASSES))

ax.set_xticklabels(
    classes,
    rotation=90,
    fontsize=7
)

ax.set_yticklabels(
    classes,
    fontsize=7
)

fig.colorbar(image, ax=ax)

fig.tight_layout()

fig.savefig(
    confusion_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("\nConfusion matrix saved:")
print(confusion_path)

# ============================================================
# 19. SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("VGG7 MODEL EVALUATION\n")
    file.write("=" * 50 + "\n")

    file.write(f"Model: {MODEL_PATH}\n")
    file.write(f"Dataset: {DATASET_PATH}\n")
    file.write(f"Classes: {NUM_CLASSES}\n")
    file.write(f"Validation images: {total_images}\n")
    file.write(f"Validation loss: {val_loss:.4f}\n")
    file.write(f"Accuracy: {accuracy:.2f}%\n")
    file.write(f"Macro precision: {macro_precision:.4f}\n")
    file.write(f"Macro recall: {macro_recall:.4f}\n")
    file.write(f"Macro F1-score: {macro_f1:.4f}\n")

print("\nSummary saved:")
print(summary_path)

print("\n")
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)