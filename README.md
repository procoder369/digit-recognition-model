# 🔢 Digit Recognition AI

A web app that recognizes handwritten digits (0–9) in real time. Draw a digit on the canvas, and a **Scikit-Learn K-Nearest Neighbors (KNN)** classifier served through **FastAPI** predicts what you drew, along with per-class probabilities and a view of the 8×8 image the model actually sees.

<!-- Add a screenshot: save it as docs/screenshot.png and uncomment the line below -->
<!-- ![App Screenshot](docs/screenshot.png) -->

---

## ✨ Features

- 🎨 **Drawing canvas** (280×280) with adjustable stroke width
- ⚡ **Live Predict** toggle: predictions update as you draw
- 🖼️ **8×8 preprocessed grid** visualization (the exact features fed to the model)
- 📊 **Class probability bars** for digits 0–9
- 🎯 **Confidence score** and inference time display
- 🔢 **Quick preset digits** (0–9) for fast testing
- 🟢 **Backend status indicator** showing whether the API is online

---

## 🧠 How It Works

The app follows a five-step pipeline:

| Step | Stage | Description |
|------|-------|-------------|
| 1 | **Draw Digit** | User draws on the HTML canvas |
| 2 | **Grayscale & Crop** | Image is converted to grayscale and cropped around the digit |
| 3 | **Resize 8×8** | Image is downsampled to an 8×8 grid, scaled to the 0–16 range |
| 4 | **KNN Model** | The 64-value feature vector is passed to the KNeighborsClassifier (K=5, uniform weights) |
| 5 | **Predicted Result** | Predicted digit, confidence and class probabilities are returned |

---

## 🛠️ Tech Stack

**Backend**
- Python
- FastAPI + Uvicorn
- Scikit-Learn (KNeighborsClassifier)
- NumPy / Pillow

**Frontend**
- HTML5, CSS3, Vanilla JavaScript
- HTML Canvas API

---

## 📁 Project Structure

```
.
├── backend/
│   ├── main.py             # FastAPI application
│   ├── train_model.py      # Trains the KNN model and saves model.pkl
│   ├── test_pipeline.py    # Tests for the preprocessing/prediction pipeline
│   ├── model.pkl           # Trained model
│   ├── requirements.txt    # Python dependencies
│   ├── run_app.bat         # Windows one-click launcher
│   └── README.md
│
└── frontend/
    ├── index.html          # UI layout
    ├── script.js           # Canvas drawing + API calls
    └── style.css           # Styling
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Set up the backend

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. (Optional) Train the model

A trained `model.pkl` is already included. To retrain it:

```bash
python train_model.py
```

### 4. Run the backend

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.
Interactive docs: `http://127.0.0.1:8000/docs`

> 💡 **Windows users:** you can simply double-click `run_app.bat` instead.

### 5. Run the frontend

Open `frontend/index.html` in your browser, or serve it locally:

```bash
cd frontend
python -m http.server 5500
```

Then visit `http://localhost:5500`.

Make sure the backend is running. The header badge will show **FastAPI Online** when the frontend can reach it.

---

## 🧪 Testing

```bash
cd backend
python test_pipeline.py
```

---

## 📸 Usage

1. Draw a digit on the black canvas with your mouse or touch.
2. With **Live Predict** on, the result updates automatically. Otherwise, click **Predict**.
3. Check the predicted digit, confidence, 8×8 grid and probability bars.
4. Click **Clear** to start over, or use the **Quick Preset Digits** to try samples.

---

## 📈 Model Details

| Property | Value |
|----------|-------|
| Algorithm | K-Nearest Neighbors |
| K | 5 |
| Weights | Uniform |
| Input features | 64 (8×8 grayscale pixels) |
| Feature scale | 0.0 – 16.0 |
| Classes | 10 (digits 0–9) |

---

## 🔮 Future Improvements

- Swap KNN for a CNN trained on MNIST for higher accuracy on messy handwriting
- Deploy the backend (Render, Railway, etc.) and host the frontend on GitHub Pages
- Add mobile touch optimizations
- Add model comparison (SVM, Random Forest, CNN)

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome. Feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. Add a `LICENSE` file if you'd like to use it.

## 👤 Author
procoder369
