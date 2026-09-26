# Digit Recognition AI ✏️🔢

A web app that recognizes hand-drawn digits (0–9) in real time using a **Scikit-Learn K-Nearest Neighbors (KNN)** classifier served through a **FastAPI** backend.

![Status](https://img.shields.io/badge/FastAPI-Online-brightgreen) ![Model](https://img.shields.io/badge/Model-KNeighborsClassifier-blue)

## Overview

Draw a digit on the canvas and the app predicts what number it is, showing the confidence score and full class-probability breakdown. The pipeline downsamples your drawing to an 8×8 grayscale grid (the same feature format used by the classic `sklearn.datasets.load_digits` dataset) before feeding it into the KNN model.

**Pipeline:**

1. **Draw Digit** — sketch a digit on the 280×280 canvas
2. **Grayscale & Crop** — the drawing is converted to grayscale and cropped
3. **Resize 8×8** — downsampled to an 8×8 pixel grid (64 features)
4. **KNN Model** — the 1×64 feature vector is passed to the trained KNN classifier
5. **Predicted Result** — the predicted digit and per-class probabilities are returned

## Features

- 🎨 Live drawing canvas with adjustable stroke width
- ⚡ Optional **Live Predict** mode (predicts as you draw)
- 🔢 Quick preset digit buttons (0–9) for testing
- 📊 Visualization of the preprocessed 8×8 feature grid
- 📈 Class probability bar chart (K and voting mode shown live, read from the trained model)
- 🚀 Fast inference (~20ms) via FastAPI + Uvicorn

## Tech Stack

| Component        | Technology                     |
|-------------------|--------------------------------|
| Backend           | FastAPI / Uvicorn              |
| Model             | Scikit-Learn `KNeighborsClassifier` |
| Feature Resolution| 8×8 grayscale pixels (64 features) |
| Frontend          | HTML/CSS/JS (served from `static/`) |

## Project Structure

```
digit-recognition-model/
├── backend/
│   ├── main.py            # FastAPI app entry point / API routes
│   ├── model.pkl          # Pre-trained KNN model
│   ├── train_model.py     # Script to train and export the KNN model
│   ├── test_pipeline.py   # Tests for the preprocessing/inference pipeline
│   ├── run_app.bat        # Windows script to launch the app
│   └── README.md
└── frontend/
    ├── index.html         # Drawing canvas UI
    ├── script.js
    └── style.css
```

`main.py` serves the `frontend/` folder as static files, so run everything
from `backend/` (e.g. `run_app.bat` or `python -m uvicorn main:app`) and open:

URL: http://127.0.0.1:8000/
## Author

**procoder369**
