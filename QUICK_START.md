# ⚡ QUICK START GUIDE

## 🎯 Get Started in 3 Steps

### 1️⃣ Place Your Model File
```
Copy your trained model: cnn_lstm_attention_final.pth
Paste it into: models/ folder
```

### 2️⃣ Install & Run (Windows)
```
Double-click: run.bat
```

### 2️⃣ Install & Run (Manual)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

### 3️⃣ Open Browser
```
Go to: http://localhost:8000
```

---

## 🎮 How to Use

1. **Upload CSV** → Click "Browse Files" or drag & drop
2. **Analyze** → Click "Analyze" button
3. **View Results** → See statistics, charts, and detailed predictions
4. **Download** → Export results as CSV

---

## ✅ Requirements Checklist

- [ ] Python 3.8+ installed
- [ ] Model file in `models/` folder
- [ ] CSV file with 36 PCA features ready
- [ ] Internet connection (for Tailwind CSS CDN)

---

## 🆘 Quick Troubleshooting

**Problem**: Model not loading
- **Fix**: Check if `cnn_lstm_attention_final.pth` is in `models/` folder

**Problem**: Port 8000 in use
- **Fix**: Change port in `app.py` line: `uvicorn.run(app, host="0.0.0.0", port=8000)`

**Problem**: Module not found
- **Fix**: Run `pip install -r requirements.txt`

---

## 📝 CSV File Format

Your CSV should have:
- **36 columns** (PCA features: PC1, PC2, ..., PC36)
- Optional: "Label" or "Attack Type" column (will be ignored)
- No headers required (but recommended)

Example:
```
PC1,PC2,PC3,...,PC36
-2.31,0.52,0.47,...,0.45
1.42,-0.39,0.82,...,-0.23
...
```

---

## 🎉 That's It!

You're ready to detect cyber threats! 🛡️

**Need Help?** Check the full README.md for detailed documentation.
