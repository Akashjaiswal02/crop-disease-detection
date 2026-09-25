from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import json
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("model/crop_disease_model.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

treatments = {
    "Potato___Early_blight": "Use crop rotation, remove infected leaves, and use proper fungicide.",
    "Potato___Late_blight": "Avoid excess moisture, remove infected plants, and apply fungicide.",
    "Potato___healthy": "Crop is healthy. Maintain proper watering, sunlight, and soil care.",

    "Early_blight": "Use crop rotation, remove infected leaves, and use proper fungicide.",
    "Late_blight": "Avoid excess moisture, remove infected plants, and apply fungicide.",
    "healthy": "Crop is healthy. Maintain proper watering, sunlight, and soil care.",

    "Potato Early Blight": "Use crop rotation, remove infected leaves, and use proper fungicide.",
    "Potato Late Blight": "Avoid excess moisture, remove infected plants, and apply fungicide.",
    "Potato healthy": "Crop is healthy. Maintain proper watering, sunlight, and soil care.",
    "Healthy": "Crop is healthy. Maintain proper watering, sunlight, and soil care."
}

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) 
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def home():
    return {
        "message": "Real Potato Disease Detection API is running"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    index = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]) * 100)

    disease = class_names[index]
    display_disease = disease.replace("___", " ").replace("_", " ")

    treatment = treatments.get(disease)
    if treatment is None:
        treatment = treatments.get(display_disease, "Treatment information not available.")

    if confidence < 85:
        return {
            "filename": file.filename,
            "disease": "Uncertain / Not Sure",
            "confidence": round(confidence, 2),
            "treatment": "Image is not clear or not similar to training data. Please upload a clear single potato leaf image."
        }

    return {
        "filename": file.filename,
        "disease": display_disease,
        "confidence": round(confidence, 2),
        "treatment": treatment
    }