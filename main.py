from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
import os
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stroke Detection API",
    description="API for detecting stroke symptoms in facial images using YOLO",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the model
model = None

# Class names - using COCO classes for demo (pretrained model)
# In production, replace with your stroke detection classes
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def load_model():
    """Load the YOLO model"""
    global model
    try:
        # Use pretrained model - it will be downloaded automatically
        model = YOLO('yolo11n.pt')  # This will download if not present
        logger.info("YOLO11n model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.warning("API starting without model - prediction endpoints will not work")
        model = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Stroke Detection API is running",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES,
        "demo": "/demo"
    }

@app.get("/demo")
async def demo():
    """Serve the demo page"""
    return FileResponse("demo.html")

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {"status": "healthy", "model_loaded": model is not None}

def process_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded image to numpy array"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def format_predictions(results) -> List[Dict[str, Any]]:
    """Format YOLO predictions into JSON response"""
    predictions = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                box = boxes[i]
                prediction = {
                    "class_id": int(box.cls[0]),
                    "class_name": CLASS_NAMES[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                }
                predictions.append(prediction)
    
    return predictions

@app.post("/predict")
async def predict_stroke(file: UploadFile = File(...)):
    """
    Predict stroke symptoms from uploaded image
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Process image
        image_array = process_image(image_bytes)
        
        # Run prediction
        results = model(image_array)
        
        # Format predictions
        predictions = format_predictions(results)
        
        # Analyze results for stroke indicators
        stroke_indicators = analyze_stroke_indicators(predictions)
        
        return {
            "filename": file.filename,
            "predictions": predictions,
            "stroke_analysis": stroke_indicators,
            "total_detections": len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def analyze_stroke_indicators(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze predictions to determine stroke indicators - DEMO VERSION"""
    # This is a demo version using COCO classes
    # In production, replace with actual stroke detection logic
    
    person_count = 0
    face_related = 0
    
    for pred in predictions:
        class_name = pred["class_name"]
        confidence = pred["confidence"]
        
        if class_name == "person":
            person_count += 1
        # For demo, we'll simulate face detection
        if class_name in ["person"]:  # In real app, this would be actual face/stroke classes
            face_related += 1
    
    # Demo logic - this would be replaced with real stroke detection
    assessment = {
        "stroke_detected": False,  # Demo always shows false for safety
        "confidence_level": "demo",
        "max_severity": "none",
        "affected_areas": [],
        "recommendation": "This is a DEMO using pretrained COCO model. Replace with trained stroke detection model for real use."
    }
    
    if person_count > 0:
        assessment["recommendation"] = f"Demo detected {person_count} person(s). Replace with trained stroke model for medical analysis."
    
    return assessment

def get_severity(class_name: str) -> str:
    """Extract severity level from class name"""
    if "severe" in class_name.lower():
        return "severe"
    elif "mid" in class_name.lower():
        return "moderate"
    elif "weak" in class_name.lower():
        return "mild"
    else:
        return "unknown"

@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Predict stroke symptoms from multiple uploaded images
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "error": "Invalid file type"
                })
                continue
            
            # Read and process image
            image_bytes = await file.read()
            image_array = process_image(image_bytes)
            
            # Run prediction
            predictions_results = model(image_array)
            predictions = format_predictions(predictions_results)
            stroke_analysis = analyze_stroke_indicators(predictions)
            
            results.append({
                "filename": file.filename,
                "predictions": predictions,
                "stroke_analysis": stroke_analysis,
                "total_detections": len(predictions)
            })
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"batch_results": results, "total_processed": len(results)}

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": "YOLO11n",
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "model_loaded": True
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)