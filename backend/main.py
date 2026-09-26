import io
from contextlib import asynccontextmanager
import os
import base64
import joblib
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Global model container
model = None

def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"KNN Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Warning: {MODEL_PATH} not found. Train the model first using train_model.py")
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    get_model()
    yield
    # (cleanup on shutdown can go here if needed)

# Initialize FastAPI App
app = FastAPI(
    title="Digit Recognition API",
    description="Digit Recognition Web App using KNN Classifier and 8x8 image preprocessing",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    image: str  # Base64 string or data URL

class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    probabilities: List[float]
    preprocessed_grid: List[List[float]]

def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, list[list[float]]]:
    """
    Image preprocessing pipeline as specified in architecture:
    1. Grayscale conversion
    2. Centering & aspect-ratio-preserving padding
    3. Resize to 8x8
    4. Convert to pixels normalized to 0..16 range (matching sklearn load_digits)
    """
    # 1. Open with PIL and handle RGBA transparency
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGBA")
    
    # Composite over a solid black background
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    composite = Image.alpha_composite(bg, img)
    
    # Convert to Grayscale
    gray = composite.convert("L")
    arr = np.array(gray, dtype=np.float32)
    
    # Invert if user drew dark strokes on light background
    if np.mean(arr) > 127:
        arr = 255.0 - arr

    # Detect stroke pixels (threshold to remove minor noise)
    threshold = 20
    coords = np.argwhere(arr > threshold)
    
    if len(coords) == 0:
        # Blank canvas: return all zeros
        grid = np.zeros((8, 8), dtype=np.float32)
        return grid.reshape(1, -1), grid.tolist()
    
    # Get bounding box of the drawn digit
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)
    
    # Crop stroke to bounding box
    stroke_img = Image.fromarray(arr.astype(np.uint8)[ymin:ymax+1, xmin:xmax+1])
    
    # Center stroke in a square canvas with ~20% border padding (matching load_digits distribution)
    w, h = stroke_img.size
    max_dim = max(w, h)
    pad = max(int(max_dim * 0.22), 2)
    padded_dim = max_dim + 2 * pad
    
    square_canvas = Image.new("L", (padded_dim, padded_dim), color=0)
    offset_x = (padded_dim - w) // 2
    offset_y = (padded_dim - h) // 2
    square_canvas.paste(stroke_img, (offset_x, offset_y))
    
    # Resize to 8x8
    resized_8x8 = square_canvas.resize((8, 8), Image.Resampling.BILINEAR)
    
    # Convert to pixel array and scale values to 0.0 - 16.0 range
    pixel_arr = np.array(resized_8x8, dtype=np.float32)
    if pixel_arr.max() > 0:
        pixel_arr = (pixel_arr / pixel_arr.max()) * 16.0
    
    pixel_arr = np.clip(pixel_arr, 0.0, 16.0)
    features = pixel_arr.reshape(1, -1)
    
    return features, pixel_arr.round(2).tolist()

@app.get("/health")
def health_check():
    """Health check endpoint to verify API and model status"""
    current_model = get_model()
    return {
        "status": "ok",
        "model_loaded": current_model is not None,
        "model_type": type(current_model).__name__ if current_model else None,
        # Report the *actual* trained hyperparameters instead of a hardcoded
        # guess, so the frontend can display what the model really is.
        "n_neighbors": getattr(current_model, "n_neighbors", None),
        "weights": getattr(current_model, "weights", None),
        "classes": list(range(10))
    }

@app.post("/predict", response_model=PredictResponse)
async def predict_digit(request: PredictRequest):
    """
    Receives base64 image data from frontend, pre-processes to 8x8,
    and runs KNN model inference.
    """
    current_model = get_model()
    if current_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Run train_model.py first.")
    
    raw_image_data = request.image
    if not raw_image_data:
        raise HTTPException(status_code=400, detail="Empty image data provided.")
    
    try:
        # Strip data URL header if present (e.g. data:image/png;base64,...)
        if "," in raw_image_data:
            raw_image_data = raw_image_data.split(",", 1)[1]
        
        image_bytes = base64.b64decode(raw_image_data)
        features, grid_8x8 = preprocess_image(image_bytes)
        
        # Check if the canvas was completely empty
        if np.all(features == 0):
            return PredictResponse(
                prediction=-1,
                confidence=0.0,
                probabilities=[0.0] * 10,
                preprocessed_grid=grid_8x8
            )
        
        # Run KNN model prediction
        prediction = int(current_model.predict(features)[0])
        
        # Calculate probabilities across classes 0-9
        probabilities = [0.0] * 10
        if hasattr(current_model, "predict_proba"):
            raw_probs = current_model.predict_proba(features)[0]
            model_classes = current_model.classes_
            for cls_idx, prob in zip(model_classes, raw_probs):
                if 0 <= cls_idx < 10:
                    probabilities[int(cls_idx)] = round(float(prob), 4)
            confidence = round(float(probabilities[prediction]), 4)
        else:
            confidence = 1.0
            probabilities[prediction] = 1.0

        return PredictResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            preprocessed_grid=grid_8x8
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing or prediction failed: {str(e)}")

# Mount static folder for serving web frontend.
#
# The project ships the UI in a sibling "frontend/" folder (next to
# "backend/", where this file lives) rather than a "static/" folder inside
# "backend/". Point StaticFiles there so index.html/script.js/style.css are
# actually found. If that layout isn't present (e.g. a different deployment),
# fall back to a local "static/" folder next to this file, which preserves
# the original behavior.
_frontend_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(_frontend_dir):
    static_dir = _frontend_dir
else:
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "Digit Recognition API is running. Frontend static/index.html not created yet."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
