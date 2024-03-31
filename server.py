import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
import cv2
import numpy as np
import torch
import joblib
import albumentations
import cnn_models

app = FastAPI()

# Load the label binarizer
lb = joblib.load('lb.pkl')

# Load the model
model = cnn_models.CustomCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

# Define the augmentation pipeline
aug = albumentations.Compose([
    albumentations.Resize(224, 224, always_apply=True),
])

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define function for sign language prediction on a single frame
def predict_sign_language_single_frame(frame):
    # Apply the augmentation
    frame = aug(image=frame)['image']
    frame = np.transpose(frame, (2, 0, 1)).astype(np.float32)
    frame = torch.tensor(frame, dtype=torch.float)
    frame = frame.unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(frame)
    _, preds = torch.max(outputs, 1)
    predicted_class = lb.classes_[preds]

    # Log the prediction
    logging.info(f"Prediction: {predicted_class}")
    
    return predicted_class

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this based on your requirements
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
async def hello():
    return{"message":"Hello SignHope!"}

# Add endpoint for predicting sign language from an image
@app.post("/predict")
async def predict_sign_language(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Make prediction on the frame
        prediction = predict_sign_language_single_frame(frame)
        prediction_text = str(prediction)  # Convert prediction to string

        # Return prediction as JSON response
        return JSONResponse(content={"prediction": prediction_text})
    except Exception as e:
        # Log the error
        logging.error(f"Error: {e}")
        return JSONResponse(content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
