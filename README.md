\# 🐾 AnimalLens - AI Animal Recognition



\*\*AnimalLens\*\* is an AI-powered animal image recognition web application built using a \*\*custom VGG7 Convolutional Neural Network (CNN)\*\* and a \*\*Flask backend\*\*.



Users can upload an animal image through the web interface, and the trained deep learning model analyzes the image and returns the predicted animal class along with confidence scores and the \*\*Top 3 predictions\*\*.



\---



\## 📌 Project Overview



AnimalLens combines deep learning and web development to create an end-to-end image classification system.



The system follows this pipeline:



```text

User uploads image

&#x20;       ↓

Frontend

HTML + CSS + JavaScript

&#x20;       ↓

Flask REST API

&#x20;       ↓

Image preprocessing

&#x20;       ↓

Custom VGG7 CNN

&#x20;       ↓

Feature extraction

&#x20;       ↓

Fully connected layers

&#x20;       ↓

44-class classification

&#x20;       ↓

Prediction probabilities

&#x20;       ↓

Top 3 predictions

&#x20;       ↓

Frontend displays results

```



The project was developed using \*\*PyTorch\*\* for deep learning and \*\*Flask\*\* for serving the trained model through a web API.



\---



\# ✨ Features



\* 🐾 Animal image classification

\* 🧠 Custom VGG7 CNN architecture

\* 🔬 Model trained from scratch

\* 🖼️ Image upload through a web interface

\* 📊 Top 3 prediction results

\* 📈 Confidence scores

\* ⚡ GPU-accelerated inference using CUDA

\* 🌐 Flask REST API

\* 💻 HTML, CSS and JavaScript frontend

\* 🔍 Model health/status endpoint

\* 📋 Model configuration endpoint

\* 🧪 Separate training, evaluation, prediction and EDA scripts



\---



\# 🧠 Deep Learning Model



AnimalLens uses a \*\*custom VGG7 Convolutional Neural Network\*\*.



The architecture was implemented specifically for this project rather than using a pretrained VGG model.



\### Model configuration



| Parameter           |                         Value |

| ------------------- | ----------------------------: |

| Architecture        |                   Custom VGG7 |

| Framework           |                       PyTorch |

| Input size          |                 224 × 224 × 3 |

| Number of classes   |                            44 |

| Validation accuracy |                    \*\*83.99%\*\* |

| Best epoch          |                        \*\*50\*\* |

| GPU                 | NVIDIA RTX PRO 4000 Blackwell |

| CUDA                |                          12.8 |



\---



\# 🔬 VGG7 Architecture



The model follows the fundamental CNN pipeline of progressively extracting visual features from the input image.



```text

224 × 224 × 3

&#x20;     │

&#x20;     ▼

Convolution

&#x20;     │

&#x20;     ▼

Activation

&#x20;     │

&#x20;     ▼

Max Pooling

&#x20;     │

&#x20;     ▼

Convolution Blocks

&#x20;     │

&#x20;     ▼

Deeper Feature Extraction

&#x20;     │

&#x20;     ▼

Max Pooling

&#x20;     │

&#x20;     ▼

Flatten

&#x20;     │

&#x20;     ▼

Fully Connected Layer

&#x20;     │

&#x20;     ▼

Fully Connected Layer

&#x20;     │

&#x20;     ▼

44 Output Classes

```



\### Why CNN?



A CNN is particularly suitable for image recognition because it can learn spatial patterns directly from images.



Early layers can learn simple visual structures such as:



\* Edges

\* Lines

\* Corners

\* Textures



Deeper layers can combine these patterns into more meaningful structures such as:



\* Fur patterns

\* Eyes

\* Ears

\* Body shapes

\* Facial structures

\* Distinctive animal characteristics



The final classification layer maps the learned representation to the \*\*44 animal classes\*\*.



\---



\# 📐 Input Image



The model expects images with a spatial resolution of:



```text

224 × 224 pixels

```



with three color channels:



```text

224 × 224 × 3

```



where:



```text

224 = image height

224 = image width

3   = RGB channels

```



Images are resized and normalized before being passed into the neural network.



The normalization uses ImageNet-style statistics:



```text

Mean:

\[0.485, 0.456, 0.406]



Standard Deviation:

\[0.229, 0.224, 0.225]

```



\---



\# 🗂️ Dataset



The project uses an animal image dataset containing \*\*44 animal classes\*\*.



The final dataset contains approximately:



```text

Total images: 13,203

Training images: 10,579

Validation images: 2,624

Classes: 44

```



The dataset was divided using an \*\*80/20 training-validation split\*\*.



The dataset itself is \*\*not included in this repository\*\* because of its size and dataset distribution/licensing considerations.



\---



\# 🧪 Exploratory Data Analysis



The project includes an EDA script:



```text

eda.py

```



The analysis was used to investigate characteristics of the dataset, including:



\* Number of images

\* Number of classes

\* Images per class

\* Class distribution

\* Minimum and maximum class counts

\* Dataset statistics

\* Duplicate images



This helped identify dataset characteristics before training the CNN.



\---



\# 🏋️ Training



The model is trained using:



```text

train.py

```



The training process includes:



1\. Loading the dataset

2\. Creating training and validation sets

3\. Applying image transformations

4\. Creating PyTorch DataLoaders

5\. Initializing the custom VGG7 model

6\. Forward propagation

7\. Computing classification loss

8\. Backpropagation

9\. Updating model weights

10\. Evaluating validation performance

11\. Saving the best model checkpoint



The best-performing model is saved as:



```text

best\_vgg7.pth

```



Large model files are intentionally excluded from the Git repository.



\---



\# 📊 Model Evaluation



Model evaluation is performed using:



```text

evaluate.py

```



The evaluation process measures classification performance on the validation dataset.



The final reported evaluation results include:



```text

Validation Accuracy: 83.99%



Macro Precision: 0.8452

Macro Recall:    0.8379

Macro F1 Score:  0.8385

```



The model achieved its best reported validation accuracy at:



```text

Epoch: 50

Accuracy: 83.99%

```



\---



\# 🔮 Prediction



Single-image prediction is handled by:



```text

predict.py

```



The prediction pipeline is:



```text

Input Image

&#x20;    ↓

Resize to 224 × 224

&#x20;    ↓

Convert to Tensor

&#x20;    ↓

Normalize

&#x20;    ↓

CNN Forward Pass

&#x20;    ↓

Output Logits

&#x20;    ↓

Softmax

&#x20;    ↓

Class Probabilities

&#x20;    ↓

Top 3 Predictions

```



The system returns the predicted animal classes along with their confidence scores.



\---



\# 🌐 Web Application



AnimalLens provides a web interface where users can upload an image and receive predictions from the trained CNN.



\## Frontend



The frontend is built using:



\* HTML

\* CSS

\* JavaScript



Frontend files:



```text

frontend/

├── index.html

├── script.js

└── style.css

```



\### `index.html`



Provides the structure of the AnimalLens user interface.



\### `style.css`



Controls the visual design, layout and responsive styling.



\### `script.js`



Handles frontend interactions and communication with the Flask API.



\---



\# ⚙️ Backend



The backend is implemented using \*\*Flask\*\*.



Backend files:



```text

backend/

├── app.py

├── config.py

├── model.py

└── \_\_init\_\_.py

```



\### `app.py`



Main Flask application responsible for:



\* Starting the web server

\* Handling API requests

\* Receiving uploaded images

\* Running predictions

\* Returning prediction results



\### `model.py`



Contains the custom VGG7 neural network implementation and model loading logic.



\### `config.py`



Contains application and model configuration such as:



\* Model path

\* Number of classes

\* Image size

\* Device

\* CUDA configuration

\* Validation accuracy

\* Best epoch



\### `\_\_init\_\_.py`



Marks the backend directory as a Python package.



\---



\# 🔌 API Endpoints



The Flask backend provides several endpoints.



\## Health Check



```http

GET /health

```



Used to verify that the backend is running and the model has loaded successfully.



Example response:



```json

{

&#x20;   "status": "running",

&#x20;   "model\_loaded": true

}

```



\---



\## Model Configuration



```http

GET /config

```



Returns information about the loaded model and configuration.



\---



\## Image Prediction



```http

POST /predict

```



Accepts an image upload and returns the model's predictions.



The prediction response contains the predicted animal classes and confidence scores.



\---



\# 💻 Technology Stack



| Category                | Technology         |

| ----------------------- | ------------------ |

| Programming Language    | Python             |

| Deep Learning           | PyTorch            |

| Neural Network          | Custom VGG7 CNN    |

| Web Framework           | Flask              |

| Frontend                | HTML               |

| Styling                 | CSS                |

| Client-side Logic       | JavaScript         |

| GPU Acceleration        | CUDA               |

| Development Environment | Visual Studio Code |

| Operating System        | Windows            |



\---



\# 📁 Project Structure



```text

cnn\_image/

│

├── backend/

│   ├── app.py

│   ├── config.py

│   ├── model.py

│   └── \_\_init\_\_.py

│

├── frontend/

│   ├── index.html

│   ├── script.js

│   └── style.css

│

├── eda.py

├── train.py

├── evaluate.py

├── predict.py

│

├── requirements.txt

├── .gitignore

└── README.md

```



The dataset and large model checkpoint files are intentionally excluded from Git.



\---



\# 🚀 Installation



\## 1. Clone the repository



```bash

git clone <YOUR\_GITHUB\_REPOSITORY\_URL>

cd cnn\_image

```



\---



\## 2. Create a virtual environment



On Windows:



```powershell

python -m venv .venv

```



\---



\## 3. Activate the virtual environment



```powershell

.venv\\Scripts\\activate

```



\---



\## 4. Install dependencies



```powershell

pip install -r requirements.txt

```



\---



\# ▶️ Running the Application



Start the Flask backend from the project root:



```powershell

python backend/app.py

```



The application runs locally at:



```text

http://127.0.0.1:5000

```



Open the address in a web browser to access AnimalLens.



\---



\# 🖼️ Using AnimalLens



The typical workflow is:



\### Step 1



Open the AnimalLens web application.



\### Step 2



Upload an animal image.



\### Step 3



The frontend sends the image to the Flask `/predict` endpoint.



\### Step 4



The backend preprocesses the image.



\### Step 5



The VGG7 model performs inference.



\### Step 6



The model generates probabilities for all 44 classes.



\### Step 7



The application selects the Top 3 predictions.



\### Step 8



The frontend displays the predicted animals and confidence scores.



\---



\# 📸 Screenshots



Screenshots of the completed application can be added here.



Example:



```text

screenshots/

├── homepage.png

├── prediction.png

└── model-info.png

```



Add screenshots to demonstrate:



\* AnimalLens homepage

\* Image upload

\* Prediction results

\* Top 3 predictions

\* Model information



\---



\# 📈 Results



The final model achieved:



```text

83.99% validation accuracy

```



with a macro F1 score of:



```text

0.8385

```



The results demonstrate that a custom CNN can learn meaningful visual representations for multi-class animal recognition.



Performance can vary between animal categories, particularly when different species have visually similar characteristics.



\---



\# 🔐 Files Excluded from Git



The repository intentionally excludes files that are unnecessary or impractical to store directly in GitHub.



Examples include:



```text

.venv/

\_\_pycache\_\_/

\*.pth

\*.pt

dataset/

mammals\_clean/

mammals\_modify/

```



The trained model checkpoint is excluded because model files can be large and are better distributed separately through an appropriate model-storage mechanism.



\---



\# 🔮 Future Improvements



Potential improvements include:



\* Increasing the size and diversity of the training dataset

\* Improving performance on visually similar animal classes

\* Experimenting with stronger data augmentation

\* Hyperparameter optimization

\* Learning-rate scheduling

\* Class-specific error analysis

\* Grad-CAM visualization

\* Model explainability

\* Docker deployment

\* Cloud deployment

\* Mobile-friendly deployment

\* Model versioning

\* Automated testing and CI/CD



\---



\# 🎓 Learning Objectives



This project demonstrates practical understanding of:



\* Convolutional Neural Networks

\* VGG-style architectures

\* Image preprocessing

\* Feature extraction

\* Convolution and pooling

\* Fully connected layers

\* Multi-class classification

\* Softmax probabilities

\* Model training

\* Validation

\* Precision, recall and F1 score

\* GPU acceleration

\* PyTorch

\* Flask REST APIs

\* Frontend-backend integration

\* End-to-end machine learning deployment



\---



\# 👩‍💻 Project



\*\*AnimalLens - AI Animal Recognition\*\*



Built as an end-to-end deep learning and web application project using a custom VGG7 CNN.



\---



