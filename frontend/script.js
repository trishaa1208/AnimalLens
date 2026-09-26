// ============================================================
// ANIMAL LENS - FRONTEND
// ============================================================

const API_URL = "http://127.0.0.1:5000";


// ============================================================
// ELEMENTS
// ============================================================

const imageInput = document.getElementById("imageInput");
const chooseImageBtn = document.getElementById("chooseImageBtn");
const predictBtn = document.getElementById("predictBtn");
const resetBtn = document.getElementById("resetBtn");

const imagePreview = document.getElementById("imagePreview");
const previewContainer = document.getElementById("previewContainer");

const selectedAnimal = document.getElementById("selectedAnimal");

const loading = document.getElementById("loading");

const resultsSection = document.getElementById("resultsSection");
const predictionsContainer = document.getElementById("predictionsContainer");


// ============================================================
// SELECT IMAGE
// ============================================================

if (chooseImageBtn) {

    chooseImageBtn.addEventListener("click", () => {

        imageInput.click();

    });

}


// ============================================================
// IMAGE SELECTED
// ============================================================

if (imageInput) {

    imageInput.addEventListener("change", () => {

        const file = imageInput.files[0];

        if (!file) {
            return;
        }


        // Check file type

        const allowedTypes = [
            "image/jpeg",
            "image/png",
            "image/webp"
        ];


        if (!allowedTypes.includes(file.type)) {

            alert("Please select a JPG, PNG, or WEBP image.");

            imageInput.value = "";

            return;
        }


        // Preview image

        const reader = new FileReader();


        reader.onload = function (event) {

            imagePreview.src = event.target.result;

            if (previewContainer) {
                previewContainer.style.display = "block";
            }

        };


        reader.readAsDataURL(file);


        // Display filename

        if (selectedAnimal) {

            selectedAnimal.textContent = file.name;

        }


        // Enable prediction

        if (predictBtn) {

            predictBtn.disabled = false;

        }

    });

}


// ============================================================
// PREDICT
// ============================================================

if (predictBtn) {

    predictBtn.addEventListener("click", predictAnimal);

}


async function predictAnimal() {

    const file = imageInput.files[0];


    // --------------------------------------------------------
    // CHECK FILE
    // --------------------------------------------------------

    if (!file) {

        alert("Please choose an animal image first.");

        return;

    }


    // --------------------------------------------------------
    // SHOW LOADING
    // --------------------------------------------------------

    predictBtn.disabled = true;


    if (loading) {

        loading.style.display = "block";

    }


    if (resultsSection) {

        resultsSection.style.display = "none";

    }


    // --------------------------------------------------------
    // CREATE FORM DATA
    // --------------------------------------------------------

    const formData = new FormData();

    formData.append("image", file);


    try {

        console.log("Sending image to:", `${API_URL}/predict`);


        // ----------------------------------------------------
        // SEND REQUEST
        // ----------------------------------------------------

        const response = await fetch(
            `${API_URL}/predict`,
            {
                method: "POST",
                body: formData
            }
        );


        console.log("Response status:", response.status);


        const data = await response.json();


        console.log("Prediction response:", data);


        // ----------------------------------------------------
        // CHECK ERROR
        // ----------------------------------------------------

        if (!response.ok || data.success === false) {

            throw new Error(
                data.error || "Prediction failed."
            );

        }


        // ----------------------------------------------------
        // GET PREDICTIONS
        // ----------------------------------------------------

        let predictions = data.predictions;


        /*
         * Some versions of model.py may return:
         *
         * [
         *   {
         *      animal: "rhino",
         *      confidence: 91.2
         *   }
         * ]
         *
         * Other versions may return:
         *
         * {
         *   predictions: [...]
         * }
         *
         * Handle both.
         */


        if (!Array.isArray(predictions)) {

            if (
                predictions &&
                Array.isArray(predictions.predictions)
            ) {

                predictions = predictions.predictions;

            } else {

                throw new Error(
                    "Invalid prediction format returned by model."
                );

            }

        }


        // ----------------------------------------------------
        // SHOW TOP 3
        // ----------------------------------------------------

        displayPredictions(
            predictions.slice(0, 3)
        );


    } catch (error) {

        console.error("Prediction error:", error);


        alert(
            "Could not predict the image.\n\n" +
            error.message
        );


    } finally {

        predictBtn.disabled = false;


        if (loading) {

            loading.style.display = "none";

        }

    }

}


// ============================================================
// DISPLAY PREDICTIONS
// ============================================================

function displayPredictions(predictions) {

    if (!predictionsContainer) {

        console.error(
            "predictionsContainer not found in HTML."
        );

        return;

    }


    predictionsContainer.innerHTML = "";


    if (!predictions || predictions.length === 0) {

        predictionsContainer.innerHTML = `
            <p>No predictions returned.</p>
        `;

        return;

    }


    predictions.forEach((prediction, index) => {

        const animal =
            prediction.animal ||
            prediction.class ||
            prediction.label ||
            "Unknown";


        const confidence =
            Number(
                prediction.confidence ||
                prediction.probability ||
                0
            );


        const card = document.createElement("div");

        card.className = "prediction-card";


        card.innerHTML = `

            <div class="prediction-rank">
                #${index + 1}
            </div>

            <div class="prediction-animal">
                ${formatAnimalName(animal)}
            </div>

            <div class="prediction-confidence">
                ${confidence.toFixed(2)}%
            </div>

            <div class="confidence-bar">

                <div
                    class="confidence-fill"
                    style="width: ${Math.min(confidence, 100)}%"
                ></div>

            </div>

        `;


        predictionsContainer.appendChild(card);

    });


    // Show results

    if (resultsSection) {

        resultsSection.style.display = "block";

    }

}


// ============================================================
// FORMAT ANIMAL NAME
// ============================================================

function formatAnimalName(name) {

    return String(name)
        .replaceAll("_", " ")
        .replace(/\b\w/g, letter => letter.toUpperCase());

}


// ============================================================
// RESET
// ============================================================

if (resetBtn) {

    resetBtn.addEventListener("click", resetPage);

}


function resetPage() {

    imageInput.value = "";


    if (imagePreview) {

        imagePreview.src = "";

    }


    if (previewContainer) {

        previewContainer.style.display = "none";

    }


    if (selectedAnimal) {

        selectedAnimal.textContent = "Selected animal";

    }


    if (resultsSection) {

        resultsSection.style.display = "none";

    }


    if (predictionsContainer) {

        predictionsContainer.innerHTML = "";

    }


    if (predictBtn) {

        predictBtn.disabled = true;

    }

}


// ============================================================
// CHECK SERVER
// ============================================================

async function checkServer() {

    try {

        const response = await fetch(
            `${API_URL}/health`
        );


        const data = await response.json();


        console.log(
            "Server status:",
            data
        );


    } catch (error) {

        console.error(
            "Cannot connect to Flask:",
            error
        );

    }

}


checkServer();