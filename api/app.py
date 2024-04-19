import joblib
import numpy as np
import torch
from fastapi import FastAPI, Request
from sklearn.preprocessing import StandardScaler

from ..model.model import BackPainNN

MODEL_PATH = "D:/Development/MLPractice/Kaggle/backpain/model/BackPain.pth"
SCALER_PATH = "D:/Development/MLPractice/Kaggle/backpain/model/Backpain_Scaler.bin"

# Create the network instance
backpain_model = BackPainNN(12, 64, 1)

# Load the weights
weights = torch.load(MODEL_PATH)

# Populate the weights on the network
backpain_model.load_state_dict(weights)

# Set the network in evaluation mode
backpain_model.eval()

# Load the scalar which was saved during training
scaler : StandardScaler = joblib.load(SCALER_PATH)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/predict")
async def predict(request: Request):
    
    # Example input to the API. This is similar to the records in Dataset_spine.csv
    # X = {
    #     "col1": 63.0278175,
    #     "col2": 22.55258597,
    #     "col3": 39.60911701,
    #     "col4": 40.47523153,
    #     "col5": 98.67291675,
    #     "col6": -0.254399986,
    #     "col7": 0.744503464,
    #     "col8": 12.5661,
    #     "col9": 14.5386,
    #     "col10": 15.30468,
    #     "col11": -28.658501,
    #     "col12": 43.5123
    # }
    
    # Get the input from the request body
    X = await request.json()
    
    # Convert it to numpy array
    array = np.zeros(shape=(1, len(X)), dtype=float)
    array[0] = np.asarray(list(X.values()), dtype=float)
    
    # Transform using the scaler which was saved while training
    array = torch.FloatTensor(scaler.transform(array))
    
    prediction = None
    
    # Predict and return the output
    with torch.inference_mode():
        y_prediction = backpain_model(array)
        
        # Map to the class
        prediction = "Normal" if torch.round(torch.sigmoid(y_prediction)).squeeze().item() == 0 else "Abnormal"    
    
    return prediction