import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import VGG7


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "mammals"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_vgg7_wide.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

NUM_CLASSES = 44

TRAIN_SPLIT = 0.80
RANDOM_SEED = 42


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("VGG7 TRAINING")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

print("=" * 60)


# ============================================================
# DATA TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

full_dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=train_transform
)

print("Dataset path:", DATASET_PATH)
print("Total images:", len(full_dataset))
print("Number of classes:", len(full_dataset.classes))

print("\nClasses:")

for i, class_name in enumerate(full_dataset.classes):
    print(f"{i:2d}: {class_name}")


# ============================================================
# CHECK NUMBER OF CLASSES
# ============================================================

if len(full_dataset.classes) != NUM_CLASSES:

    raise ValueError(
        f"\nExpected {NUM_CLASSES} classes, "
        f"but found {len(full_dataset.classes)} classes."
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_size = int(
    TRAIN_SPLIT * len(full_dataset)
)

val_size = len(full_dataset) - train_size


generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)


# ============================================================
# USE VALIDATION TRANSFORM
# ============================================================

val_dataset.dataset = copy.deepcopy(
    full_dataset
)

val_dataset.dataset.transform = val_transform


print("\nDataset split:")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating wider VGG7 model...")

model = VGG7(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# ============================================================
# PRINT MODEL
# ============================================================

print("\nModel architecture:")
print(model)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


# ============================================================
# TRAINING VARIABLES
# ============================================================

best_val_accuracy = 0.0

best_model_state = None

best_epoch = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        # Clear gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        # Statistics
        running_loss += (
            loss.item() * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        100.0 * correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            val_running_loss += (
                loss.item() * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_running_loss / val_total
    )

    val_accuracy = (
        100.0 * val_correct / val_total
    )


    # --------------------------------------------------------
    # LEARNING RATE UPDATE
    # --------------------------------------------------------

    scheduler.step(
        val_accuracy
    )


    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    print(
        f"Learning Rate: {current_lr:.6f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_epoch = epoch + 1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )


        checkpoint = {

            "model_state_dict":
                best_model_state,

            "num_classes":
                NUM_CLASSES,

            "classes":
                full_dataset.classes,

            "best_val_accuracy":
                best_val_accuracy,

            "epoch":
                best_epoch
        }


        torch.save(
            checkpoint,
            MODEL_PATH
        )


        print("\n*** BEST MODEL SAVED ***")

        print(
            f"Validation Accuracy: "
            f"{best_val_accuracy:.2f}%"
        )

        print(
            f"Saved to: {MODEL_PATH}"
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"Model saved at:\n{MODEL_PATH}"
)

print("=" * 60)