# 🛡️ Real-Time Cyber Threat Detection System

A comprehensive web application for detecting cyber threats in network traffic using CNN-LSTM-Attention deep learning architecture.

## 🌟 Features

- **Real-time threat detection** using advanced AI model
- **Interactive dashboard** with Tailwind CSS
- **File upload support** for batch analysis (CSV files)
- **Visual analytics** with charts and statistics
- **Download results** as CSV
- **FastAPI backend** for high performance
- **Responsive design** for all devices

## 🏗️ Architecture

- **Backend**: FastAPI (Python)
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Model**: CNN-LSTM-Attention Fusion Architecture
- **Framework**: PyTorch

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Trained model file: `cnn_lstm_attention_final.pth`

## 🚀 Installation

### Step 1: Clone or Navigate to Project Directory

```bash
cd CyberThreat_Detection_WebApp
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add Your Trained Model

1. Place your trained model file `cnn_lstm_attention_final.pth` in the `models/` folder
2. Make sure the model file name matches exactly

## ▶️ Running the Application

### Start the Server

```bash
python app.py
```

Or alternatively:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Access the Application

Open your web browser and go to:
```
http://localhost:8000
```

## 📁 Project Structure

```
CyberThreat_Detection_WebApp/
│
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
│
├── models/               # Model files directory
│   └── cnn_lstm_attention_final.pth  (Place your model here)
│
├── templates/            # HTML templates
│   └── index.html        # Main dashboard
│
├── static/               # Static files (CSS, JS, images)
│   └── (empty for now - Tailwind CDN used)
│
└── uploads/              # Temporary upload directory
```

## 🔧 Configuration

### Model Parameters (in app.py)

```python
SEQ_LENGTH = 10           # Sequence length for LSTM
NUM_FEATURES = 36         # Number of input features (PCA components)
NUM_CLASSES = 2           # Binary classification (Normal/Attack)
```

Adjust these parameters based on your model training configuration.

## 📊 Usage Guide

### 1. Upload CSV File

- Click "Browse Files" or drag & drop your CSV file
- File should contain network traffic features (36 PCA components)
- Labels column will be automatically removed if present

### 2. Analyze

- Click "Analyze" button after file upload
- Wait for processing (may take a few moments)

### 3. View Results

- **Statistics Cards**: Total samples, normal traffic, attacks, confidence
- **Charts**: Threat distribution and confidence distribution
- **Detailed Table**: Individual predictions with threat levels
- **Download**: Export results as CSV

## 🎯 API Endpoints

### Health Check
```
GET /api/health
```

### Batch Prediction (File Upload)
```
POST /api/predict
Content-Type: multipart/form-data
Body: file (CSV)
```

### Single Prediction
```
POST /api/predict_single
Content-Type: application/json
Body: {"features": [array of 36 values]}
```

## 🔐 Model Information

**Architecture**: CNN-LSTM-Attention Fusion
- **CNN Branch**: Spatial feature extraction
- **Bidirectional LSTM**: Temporal modeling
- **Attention Mechanism**: Focus on important features
- **Fusion Layer**: Combines CNN and LSTM outputs

## 📈 Performance Metrics

Based on your training results:
- **Accuracy**: 95.25%
- **Precision**: 95.28%
- **Recall**: 95.25%
- **F1-Score**: 95.25%

## 🛠️ Troubleshooting

### Model Not Loading
```
⚠️ Error loading model
```
**Solution**: Ensure `cnn_lstm_attention_final.pth` is in the `models/` folder

### Port Already in Use
```
ERROR: Address already in use
```
**Solution**: Change port in app.py or kill process using port 8000
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:8000 | xargs kill -9
```

### File Upload Error
```
Only CSV files are allowed
```
**Solution**: Ensure your file has `.csv` extension

### Feature Count Mismatch
```
Expected 36 features, got X
```
**Solution**: Verify your data preprocessing matches training (PCA with 36 components)

## 🎨 Customization

### Change Theme Colors

Edit the gradient in `templates/index.html`:
```css
.gradient-bg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```

### Modify Threat Levels

Edit threat level logic in `app.py`:
```python
if conf >= 0.95:
    threat_levels.append('High Threat')
elif conf >= 0.80:
    threat_levels.append('Medium Threat')
else:
    threat_levels.append('Low Threat')
```

## 📝 Notes

- Maximum file upload size: 16MB
- Table displays first 100 results (download CSV for complete data)
- Model runs on GPU if available, otherwise CPU
- CORS enabled for development (disable in production)

## 🚀 Deployment

### For Production:

1. Disable debug mode in `app.py`
2. Configure CORS properly
3. Use production WSGI server (gunicorn, etc.)
4. Set up SSL/HTTPS
5. Use environment variables for sensitive config

## 📞 Support

For issues or questions, contact:
- **Developer**: tanish.qriocity@gmail.com
- **Client**: iamcodepilot@gmail.com
- **Support**: Gandhi.qriocity@gmail.com

## 📄 License

Proprietary - Cyber Threat Detection Project

## 🙏 Acknowledgments

- CICIDS2017 Dataset
- CICIDS2018 Dataset
- Gotham2025 Dataset
- PyTorch Framework
- FastAPI Framework
- Tailwind CSS

---

**Created**: December 2025
**Version**: 1.0.0
**Status**: Production Ready ✅
