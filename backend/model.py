
import torch
import torch.nn as nn
from torchvision import transforms

import config


# ============================================================
# VGG7 MODEL
# ============================================================

class VGG7(nn.Module):

    def __init__(self, num_classes=44):

        super().__init__()

        self.features = nn.Sequential(

            # ==================================================
            # BLOCK 1
            # ==================================================

            # Conv 1
            # Input: 3 channels
            # Output: 32 feature maps
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),


            # Conv 2
            # Input: 32 channels
            # Output: 32 feature maps
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),


            # Max Pool 1
            nn.MaxPool2d(2),


            # ==================================================
            # BLOCK 2
            # ==================================================

            # Conv 3
            # Input: 32 channels
            # Output: 64 feature maps
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),


            # Conv 4
            # Input: 64 channels
            # Output: 64 feature maps
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),


            # Max Pool 2
            nn.MaxPool2d(2),


            # ==================================================
            # BLOCK 3
            # ==================================================

            # Conv 5
            # Input: 64 channels
            # Output: 128 feature maps
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),


            # Conv 6
            # Input: 128 channels
            # Output: 128 feature maps
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),


            # ==================================================
            # BLOCK 4
            # ==================================================

            # Conv 7
            # Input: 128 channels
            # Output: 256 feature maps
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),


            # Max Pool 3
            nn.MaxPool2d(2),
        )


        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(

            # ------------------------------------------------
            # Adaptive Average Pooling
            # ------------------------------------------------
            # 28 × 28 × 256
            #       ↓
            # 1 × 1 × 256
            nn.AdaptiveAvgPool2d((1, 1)),


            # ------------------------------------------------
            # Flatten
            # ------------------------------------------------
            # 256 × 1 × 1
            #       ↓
            # 256
            nn.Flatten(),


            # ------------------------------------------------
            # Dense Layer
            # ------------------------------------------------
            # 256 → 512
            nn.Linear(
                256,
                512
            ),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),


            # ------------------------------------------------
            # Dropout
            # ------------------------------------------------
            nn.Dropout(0.4),


            # ------------------------------------------------
            # Output Layer
            # ------------------------------------------------
            # 512 → 44 classes
            nn.Linear(
                512,
                num_classes
            )
        )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, x):

        # Feature extraction
        x = self.features(x)

        # Classification
        x = self.classifier(x)

        return x


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (
            config.IMAGE_SIZE,
            config.IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=config.MEAN,
        std=config.STD
    )
])


# ============================================================
# DEFAULT CLASS LIST
# ============================================================

DEFAULT_CLASSES = [

    "african_elephant",
    "alpaca",
    "american_bison",
    "anteater",
    "arctic_fox",
    "armadillo",
    "baboon",
    "badger",
    "blue_whale",
    "brown_bear",
    "camel",
    "dolphin",
    "giraffe",
    "groundhog",
    "highland_cattle",
    "horse",
    "jackal",
    "kangaroo",
    "koala",
    "manatee",
    "mongoose",
    "mountain_goat",
    "opossum",
    "orangutan",
    "otter",
    "polar_bear",
    "porcupine",
    "red_panda",
    "rhinoceros",
    "sea_lion",
    "snow_leopard",
    "squirrel",
    "sugar_glider",
    "tapir",
    "vampire_bat",
    "vicuna",
    "walrus",
    "warthog",
    "water_buffalo",
    "weasel",
    "wildebeest",
    "wombat",
    "yak",
    "zebra"
]


# ============================================================
# MODEL MANAGER
# ============================================================

class ModelManager:

    def __init__(self):

        self.model = None

        self.classes = DEFAULT_CLASSES

        self.loaded = False

        self.best_accuracy = config.VALIDATION_ACCURACY

        self.best_epoch = config.BEST_EPOCH


    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        print()
        print("=" * 60)
        print("LOADING VGG7 MODEL")
        print("=" * 60)


        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        print(
            f"Device: {config.DEVICE}"
        )


        # ----------------------------------------------------
        # GPU information
        # ----------------------------------------------------

        if torch.cuda.is_available():

            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

            print(
                "CUDA:",
                torch.version.cuda
            )

        else:

            print("GPU: CPU")


        # ----------------------------------------------------
        # Check model checkpoint
        # ----------------------------------------------------

        if not config.MODEL_PATH.exists():

            raise FileNotFoundError(

                "\nCould not find best_vgg7.pth\n\n"

                "Expected model checkpoint at:\n"

                f"{config.MODEL_PATH}"
            )


        print(
            "Model checkpoint:",
            config.MODEL_PATH
        )


        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(

            config.MODEL_PATH,

            map_location=config.DEVICE,

            weights_only=False
        )


        # ----------------------------------------------------
        # Determine classes and number of classes
        # ----------------------------------------------------

        if isinstance(checkpoint, dict):

            # If class names were saved inside checkpoint
            if "classes" in checkpoint:

                self.classes = checkpoint["classes"]


            # If number of classes was saved
            if "num_classes" in checkpoint:

                num_classes = checkpoint["num_classes"]

            else:

                num_classes = len(self.classes)

        else:

            num_classes = len(self.classes)


        print(
            "Number of classes:",
            num_classes
        )


        # ----------------------------------------------------
        # Create VGG7 model
        # ----------------------------------------------------

        self.model = VGG7(

            num_classes=num_classes
        )


        # ----------------------------------------------------
        # Extract state dictionary
        # ----------------------------------------------------

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                state_dict = checkpoint[
                    "model_state_dict"
                ]

            elif "state_dict" in checkpoint:

                state_dict = checkpoint[
                    "state_dict"
                ]

            else:

                state_dict = checkpoint

        else:

            state_dict = checkpoint


        # ----------------------------------------------------
        # Load trained weights
        # ----------------------------------------------------

        self.model.load_state_dict(
            state_dict
        )


        # ----------------------------------------------------
        # Move model to GPU / CPU
        # ----------------------------------------------------

        self.model.to(
            config.DEVICE
        )


        # ----------------------------------------------------
        # Evaluation mode
        # ----------------------------------------------------

        self.model.eval()

        self.loaded = True


        # ----------------------------------------------------
        # Checkpoint information
        # ----------------------------------------------------

        self.best_accuracy = (
            config.VALIDATION_ACCURACY
        )

        self.best_epoch = (
            config.BEST_EPOCH
        )


        # ----------------------------------------------------
        # Display information
        # ----------------------------------------------------

        print()
        print("Model loaded successfully.")

        print(
            f"Classes: {len(self.classes)}"
        )

        print(
            f"Validation accuracy: "
            f"{self.best_accuracy:.2f}%"
        )

        print(
            f"Best epoch: {self.best_epoch}"
        )

        print("=" * 60)
        print()


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, image):

        if not self.loaded:

            raise RuntimeError(
                "Model has not been loaded."
            )


        # ----------------------------------------------------
        # Convert image to RGB
        # ----------------------------------------------------

        if image.mode != "RGB":

            image = image.convert("RGB")


        # ----------------------------------------------------
        # Apply image transformation
        # ----------------------------------------------------

        tensor = transform(image)


        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        tensor = tensor.unsqueeze(0)


        # ----------------------------------------------------
        # Move image to GPU / CPU
        # ----------------------------------------------------

        tensor = tensor.to(
            config.DEVICE
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = self.model(
                tensor
            )


            # Convert logits into probabilities
            probabilities = torch.softmax(

                outputs,

                dim=1
            )


        # ----------------------------------------------------
        # Get top 3 predictions
        # ----------------------------------------------------

        top_probabilities, top_indices = torch.topk(

            probabilities,

            k=min(
                3,
                len(self.classes)
            ),

            dim=1
        )


        predictions = []


        # ----------------------------------------------------
        # Convert predictions to readable format
        # ----------------------------------------------------

        for probability, index in zip(

            top_probabilities[0],

            top_indices[0]

        ):

            class_index = index.item()


            confidence = (
                probability.item() * 100
            )


            predictions.append({

                "animal":
                    self.classes[class_index],

                "confidence":
                    round(
                        confidence,
                        2
                    )
            })


        return predictions


# ============================================================
# GLOBAL MODEL MANAGER
# ============================================================

model_manager = ModelManager()
