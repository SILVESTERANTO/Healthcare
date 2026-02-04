from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import uvicorn
from pydantic import BaseModel
from typing import List
import shutil

app = FastAPI(title="Cyber Threat Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# MODEL ARCHITECTURE
# ============================================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class CNNLSTMAttention(nn.Module):
    def __init__(self, input_size, num_classes, seq_length):
        super(CNNLSTMAttention, self).__init__()
        
        # CNN Branch
        self.conv1 = nn.Conv1d(in_channels=seq_length, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(0.3)
        
        # LSTM Branch
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, 
                           batch_first=True, dropout=0, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, num_layers=1, 
                           batch_first=True, dropout=0, bidirectional=True)
        
        # Attention
        self.attention = AttentionLayer(hidden_size=128)
        
        # Fusion & Classification
        self.fc1 = nn.Linear(128 + 128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # CNN Path
        cnn_out = self.relu(self.bn1(self.conv1(x)))
        cnn_out = self.pool(cnn_out)
        cnn_out = self.relu(self.bn2(self.conv2(cnn_out)))
        cnn_out = self.pool(cnn_out)
        cnn_out = self.dropout_cnn(cnn_out)
        cnn_out = torch.mean(cnn_out, dim=2)
        
        # LSTM Path
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)
        
        # Attention
        attended_out, attn_weights = self.attention(lstm_out)
        
        # Fusion
        fused = torch.cat([attended_out, cnn_out], dim=1)
        
        # Classification
        out = self.relu(self.bn3(self.fc1(fused)))
        out = self.dropout1(out)
        out = self.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)
        
        return out

# ============================================
# LOAD MODEL
# ============================================
SEQ_LENGTH = 10
NUM_FEATURES = 35  # Adjust based on your PCA features
NUM_CLASSES = 2

model = CNNLSTMAttention(NUM_FEATURES, NUM_CLASSES, SEQ_LENGTH).to(device)

try:
    checkpoint = torch.load('models/cnn_lstm_attention_complete.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    print("Please place 'cnn_lstm_attention_complete.pth' in the 'models' folder")

# ============================================
# PYDANTIC MODELS
# ============================================
class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    threat_level: str
    probabilities: dict

# ============================================
# HELPER FUNCTIONS
# ============================================
def create_sequences(X, seq_len=10):
    if len(X) < seq_len:
        padding = np.zeros((seq_len - len(X), X.shape[1]))
        X = np.vstack([padding, X])
    
    sequences = []
    for i in range(len(X) - seq_len + 1):
        sequences.append(X[i:i + seq_len])
    return np.array(sequences)

def predict_threats(data):
    """Make predictions on network traffic data"""
    model.eval()
    
    # Create sequences
    X_seq = create_sequences(data, SEQ_LENGTH)
    X_tensor = torch.FloatTensor(X_seq).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    confidence = np.max(probabilities, axis=1)
    
    # Generate threat levels
    threat_levels = []
    for pred, conf in zip(predictions, confidence):
        if pred == 0:
            threat_levels.append('Normal')
        else:
            if conf >= 0.95:
                threat_levels.append('High Threat')
            elif conf >= 0.80:
                threat_levels.append('Medium Threat')
            else:
                threat_levels.append('Low Threat')
    
    results = {
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist(),
        'confidence': confidence.tolist(),
        'threat_levels': threat_levels
    }
    
    return results

# ============================================
# ROUTES
# ============================================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/predict")
async def predict_file(file: UploadFile = File(...)):
    try:
        # Check file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Save uploaded file temporarily
        temp_path = f"uploads/{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load and process data
        df = pd.read_csv(temp_path)
        
        # Remove label column if exists
        if 'Label' in df.columns:
            df = df.drop('Label', axis=1)
        if 'Attack Type' in df.columns:
            df = df.drop('Attack Type', axis=1)
        
        data = df.values
        
        # Make predictions
        results = predict_threats(data)
        
        # Calculate statistics
        n_attacks = sum(1 for p in results['predictions'] if p == 1)
        n_normal = len(results['predictions']) - n_attacks
        avg_confidence = float(np.mean(results['confidence']))
        
        response = {
            'success': True,
            'total_samples': len(results['predictions']),
            'normal_count': n_normal,
            'attack_count': n_attacks,
            'avg_confidence': avg_confidence,
            'predictions': results['predictions'],
            'confidence': results['confidence'],
            'threat_levels': results['threat_levels'],
            'probabilities': results['probabilities']
        }
        
        # Clean up
        os.remove(temp_path)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict_single", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    try:
        features = request.features
        
        if len(features) != NUM_FEATURES:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {NUM_FEATURES} features, got {len(features)}"
            )
        
        # Convert to numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        results = predict_threats(features_array)
        
        response = PredictionResponse(
            success=True,
            prediction='Attack' if results['predictions'][0] == 1 else 'Normal',
            confidence=float(results['confidence'][0]),
            threat_level=results['threat_levels'][0],
            probabilities={
                'normal': float(results['probabilities'][0][0]),
                'attack': float(results['probabilities'][0][1])
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
