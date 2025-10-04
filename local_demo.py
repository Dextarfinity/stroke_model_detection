from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import numpy as np
import io
from PIL import Image
import os
import logging
import random
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stroke Detection API - Local Demo",
    description="Local demo API for testing stroke detection interface",
    version="1.0.0-local"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulated stroke detection classes
CLASS_NAMES = [
    'normalEye', 'normalMouth', 'strokeEyeMid', 'strokeEyeSevere', 
    'strokeEyeWeak', 'strokeMouthMid', 'strokeMouthSevere', 'strokeMouthWeak'
]

def simulate_predictions(image_array: np.ndarray) -> List[Dict[str, Any]]:
    """Simulate YOLO predictions for demo purposes"""
    height, width = image_array.shape[:2]
    
    # Generate random predictions to simulate model output
    num_predictions = random.randint(1, 3)
    predictions = []
    
    for i in range(num_predictions):
        # Randomly select a class
        class_id = random.randint(0, len(CLASS_NAMES) - 1)
        class_name = CLASS_NAMES[class_id]
        
        # Generate random bounding box
        x1 = random.randint(0, width // 2)
        y1 = random.randint(0, height // 2)
        x2 = random.randint(x1 + 50, width)
        y2 = random.randint(y1 + 50, height)
        
        # Generate confidence score
        confidence = random.uniform(0.5, 0.95)
        
        prediction = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            }
        }
        predictions.append(prediction)
    
    return predictions

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Stroke Detection API - Local Demo",
        "model_loaded": True,
        "classes": CLASS_NAMES,
        "demo": "/demo",
        "note": "This is a local demo with simulated predictions"
    }

@app.get("/demo")
async def demo():
    """Serve the demo page"""
    return FileResponse("demo.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True, "mode": "local_demo"}

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

def analyze_stroke_indicators(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze predictions to determine stroke indicators"""
    stroke_count = 0
    normal_count = 0
    stroke_types = []
    
    for pred in predictions:
        class_name = pred["class_name"]
        confidence = pred["confidence"]
        
        if "stroke" in class_name.lower():
            stroke_count += 1
            stroke_types.append({
                "type": class_name,
                "confidence": confidence,
                "severity": get_severity(class_name)
            })
        elif "normal" in class_name.lower():
            normal_count += 1
    
    # Determine overall assessment
    if stroke_count > 0:
        severity_levels = [st["severity"] for st in stroke_types]
        max_severity = max(severity_levels, key=lambda x: ["mild", "moderate", "severe"].index(x)) if severity_levels else "mild"
        
        assessment = {
            "stroke_detected": True,
            "confidence_level": "high" if stroke_count > normal_count else "moderate",
            "max_severity": max_severity,
            "affected_areas": list(set([st["type"] for st in stroke_types])),
            "recommendation": "⚠️ DEMO: Simulated stroke detection. Consult medical professional for real diagnosis."
        }
    else:
        assessment = {
            "stroke_detected": False,
            "confidence_level": "high" if normal_count > 0 else "low",
            "max_severity": "none",
            "affected_areas": [],
            "recommendation": "✅ DEMO: No stroke indicators detected in simulation."
        }
    
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

@app.post("/predict")
async def predict_stroke(file: UploadFile = File(...)):
    """
    Predict stroke symptoms from uploaded image - LOCAL DEMO VERSION
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Process image
        image_array = process_image(image_bytes)
        
        # Simulate predictions (in real version, this would be model(image_array))
        predictions = simulate_predictions(image_array)
        
        # Analyze results for stroke indicators
        stroke_indicators = analyze_stroke_indicators(predictions)
        
        return {
            "filename": file.filename,
            "predictions": predictions,
            "stroke_analysis": stroke_indicators,
            "total_detections": len(predictions),
            "note": "🔬 LOCAL DEMO: These are simulated results for testing purposes"
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Predict stroke symptoms from multiple uploaded images - LOCAL DEMO
    """
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
            
            # Simulate predictions
            predictions = simulate_predictions(image_array)
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
    
    return {
        "batch_results": results, 
        "total_processed": len(results),
        "note": "🔬 LOCAL DEMO: These are simulated results for testing purposes"
    }

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_type": "Local Demo Simulator",
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "model_loaded": True,
        "note": "This is a local demo with simulated predictions"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("🚀 Starting Local Stroke Detection Demo API...")
    print(f"📱 Demo URL: http://localhost:{port}/demo")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print("⚠️  Note: This is a demo with simulated predictions")
    uvicorn.run(app, host="0.0.0.0", port=port)