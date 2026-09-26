from pathlib import Path

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory
)

from flask_cors import CORS
from PIL import Image
import io

import model as model_lib


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


# ============================================================
# CHECK FRONTEND FILES
# ============================================================

print("=" * 60)
print("FRONTEND FILE CHECK")
print("=" * 60)

print("BASE DIR:")
print(BASE_DIR)

print("FRONTEND DIR:")
print(FRONTEND_DIR)

print("Frontend exists:")
print(FRONTEND_DIR.exists())

print("index.html:")
print((FRONTEND_DIR / "index.html").exists())

print("style.css:")
print((FRONTEND_DIR / "style.css").exists())

print("script.js:")
print((FRONTEND_DIR / "script.js").exists())

print("=" * 60)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR)
)

CORS(app)


# ============================================================
# SERVE CSS
# ============================================================

@app.route("/static/style.css")
def serve_css():

    print("Serving style.css")

    return send_from_directory(
        str(FRONTEND_DIR),
        "style.css"
    )


# ============================================================
# SERVE JAVASCRIPT
# ============================================================

@app.route("/static/script.js")
def serve_js():

    print("Serving script.js")

    return send_from_directory(
        str(FRONTEND_DIR),
        "script.js"
    )


# ============================================================
# SERVE OTHER FRONTEND FILES
# ============================================================

@app.route("/static/<path:filename>")
def serve_static(filename):

    print("Serving static file:", filename)

    return send_from_directory(
        str(FRONTEND_DIR),
        filename
    )


# ============================================================
# SERVER STARTING
# ============================================================

print("=" * 60)
print("SERVER STARTING...")
print("=" * 60)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model_lib.model_manager.load_model()

    print()
    print("=" * 60)
    print("VGG7 ANIMAL IMAGE RECOGNITION")
    print("=" * 60)

    print("Server URL:")
    print("http://127.0.0.1:5000")

    print("Model:")
    print("VGG7")

    print("Number of classes:")
    print("44")

    print("Image size:")
    print("224 x 224")

    print("Validation accuracy:")
    print("83.99%")

    print("Best epoch:")
    print("50")

    print("Device:")
    print("cuda")

    print("=" * 60)

except Exception as e:

    print("MODEL LOADING ERROR:")
    print(e)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "model": "VGG7",
        "classes": 44,
        "image_size": 224,
        "device": "cuda"
    })


# ============================================================
# CONFIG
# ============================================================

@app.route("/config")
def config():

    return jsonify({
        "model": "VGG7",
        "num_classes": 44,
        "image_size": 224,
        "validation_accuracy": 83.99,
        "best_epoch": 50,
        "device": "cuda",
        "gpu": "NVIDIA RTX PRO 4000 Blackwell",
        "cuda": "12.8"
    })


# ============================================================
# PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image uploaded."
            }), 400


        file = request.files["image"]


        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "No image selected."
            }), 400


        image_bytes = file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")


        result = model_lib.model_manager.predict(image)


        print()
        print("=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(result)
        print("=" * 60)


        return jsonify({
            "success": True,
            "predictions": result
        })


    except Exception as e:

        print()
        print("=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)
        print(e)
        print("=" * 60)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )