# ============================================================
# VGG7 MAMMAL IMAGE RECOGNITION - PREDICTION
# ============================================================

import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_vgg7.pth"

IMAGE_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# VGG7 MODEL
# ============================================================

class VGG7(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 7th convolution
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
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
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 60)
    print("LOADING TRAINED VGG7 MODEL")
    print("=" * 60)

    print()

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        print("=" * 60)
        print("ERROR")
        print("=" * 60)

        print("Model checkpoint not found:")
        print(MODEL_PATH)

        print()
        print("Make sure best_vgg7.pth is inside:")
        print(BASE_DIR)

        print("=" * 60)

        return None, None

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    try:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

        # ----------------------------------------------------
        # Get class names
        # ----------------------------------------------------

        classes = checkpoint["classes"]

        num_classes = checkpoint["num_classes"]

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        model = VGG7(
            num_classes=num_classes
        )

        # ----------------------------------------------------
        # Load trained weights
        # ----------------------------------------------------

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # ----------------------------------------------------
        # Move model to GPU / CPU
        # ----------------------------------------------------

        model = model.to(DEVICE)

        # ----------------------------------------------------
        # Evaluation mode
        # ----------------------------------------------------

        model.eval()

        print("Model loaded successfully.")
        print()

        print("Model path  :", MODEL_PATH)
        print("Device      :", DEVICE)

        if torch.cuda.is_available():

            print(
                "GPU         :",
                torch.cuda.get_device_name(0)
            )

            print(
                "CUDA Version:",
                torch.version.cuda
            )

        else:

            print("GPU         : Not available")

        print("Classes     :", num_classes)

        if "best_val_accuracy" in checkpoint:

            print(
                "Best Val Acc:",
                f"{checkpoint['best_val_accuracy']:.2f}%"
            )

        if "epoch" in checkpoint:

            print(
                "Best Epoch  :",
                checkpoint["epoch"]
            )

        print()

        print("=" * 60)

        return model, classes

    except Exception as e:

        print("=" * 60)
        print("ERROR WHILE LOADING MODEL")
        print("=" * 60)

        print(e)

        print("=" * 60)

        return None, None


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(model, classes, image_path):

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not os.path.exists(image_path):

        print()
        print("ERROR: Image not found.")
        print(image_path)

        return

    try:

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        # ----------------------------------------------------
        # Transform image
        # ----------------------------------------------------

        image_tensor = transform(image)

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        image_tensor = image_tensor.unsqueeze(0)

        # ----------------------------------------------------
        # Move to GPU / CPU
        # ----------------------------------------------------

        image_tensor = image_tensor.to(DEVICE)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

        # ----------------------------------------------------
        # Get highest probability
        # ----------------------------------------------------

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

        predicted_index = predicted_index.item()

        confidence = confidence.item() * 100

        predicted_class = classes[predicted_index]

        # ----------------------------------------------------
        # Top 5 predictions
        # ----------------------------------------------------

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=min(5, len(classes)),
            dim=1
        )

        print()
        print("=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)

        print()
        print("Image       :", image_path)
        print("Prediction  :", predicted_class)
        print(
            "Confidence  :",
            f"{confidence:.2f}%"
        )

        print()
        print("Top 5 Predictions")
        print("-" * 60)

        for i in range(len(top_indices[0])):

            index = top_indices[0][i].item()

            probability = (
                top_probabilities[0][i].item()
                * 100
            )

            print(
                f"{i + 1}. "
                f"{classes[index]:25s} "
                f"{probability:.2f}%"
            )

        print()
        print("=" * 60)

    except Exception as e:

        print()
        print("=" * 60)
        print("ERROR DURING PREDICTION")
        print("=" * 60)

        print(e)

        print("=" * 60)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    model, classes = load_model()

    if model is not None:

        print()
        print("=" * 60)
        print("ENTER IMAGE PATH")
        print("=" * 60)

        print()
        print("Example:")
        print(
            r"C:\Users\USER\Desktop\cnn_image\test.jpg"
        )

        print()

        image_path = input(
            "Enter image path: "
        ).strip().strip('"')

        predict_image(
            model,
            classes,
            image_path
        )