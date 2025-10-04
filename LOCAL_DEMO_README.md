# 🧠 Local Stroke Detection Demo

This is a simplified local version of the Stroke Detection API that works without complex model dependencies. Perfect for testing the interface and API functionality!

## 🚀 Quick Start

### Option 1: Using Batch File (Easiest)
```bash
# Double-click this file or run in command prompt:
start_local_demo.bat
```

### Option 2: Manual Start
```bash
# Install light dependencies
pip install -r requirements_local.txt

# Start the demo
python local_demo.py
```

## 🎯 Features

- ✅ **Working Demo Interface**: Full web UI at `/demo`
- ✅ **Simulated Predictions**: Realistic stroke detection simulation
- ✅ **All API Endpoints**: Complete API functionality
- ✅ **No Heavy Dependencies**: Runs without YOLO/PyTorch
- ✅ **Fast Startup**: Loads in seconds
- ✅ **Educational**: See how the real API would work

## 📱 Access Points

Once running, visit:
- **Demo Page**: http://localhost:8000/demo
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🔬 What It Does

### Simulated Detection Classes:
- `normalEye` - Normal eye appearance
- `normalMouth` - Normal mouth appearance  
- `strokeEyeMid` - Moderate stroke symptoms in eye
- `strokeEyeSevere` - Severe stroke symptoms in eye
- `strokeEyeWeak` - Mild stroke symptoms in eye
- `strokeMouthMid` - Moderate stroke symptoms in mouth
- `strokeMouthSevere` - Severe stroke symptoms in mouth
- `strokeMouthWeak` - Mild stroke symptoms in mouth

### Simulation Logic:
- Generates 1-3 random detections per image
- Assigns random confidence scores (50-95%)
- Creates realistic bounding boxes
- Provides stroke analysis with severity assessment

## 🧪 Testing the Interface

1. **Upload Images**: Drag and drop any images
2. **View Results**: See simulated stroke detection
3. **Batch Processing**: Test multiple images
4. **API Testing**: Use the Swagger docs at `/docs`

## 📊 Sample Response

```json
{
  "filename": "test_image.jpg",
  "predictions": [
    {
      "class_id": 2,
      "class_name": "strokeEyeMid", 
      "confidence": 0.85,
      "bbox": {"x1": 100, "y1": 150, "x2": 200, "y2": 250}
    }
  ],
  "stroke_analysis": {
    "stroke_detected": true,
    "confidence_level": "high",
    "max_severity": "moderate", 
    "affected_areas": ["strokeEyeMid"],
    "recommendation": "⚠️ DEMO: Simulated stroke detection..."
  },
  "total_detections": 1,
  "note": "🔬 LOCAL DEMO: These are simulated results"
}
```

## ⚠️ Important Notes

- **Demo Only**: This generates fake predictions for testing
- **No Real Analysis**: Not for actual medical diagnosis
- **Interface Testing**: Perfect for UI/UX validation
- **API Development**: Test integration before real deployment

## 🔄 Next Steps

1. **Test Locally**: Verify interface works as expected
2. **Deploy to Railway**: Use the main deployment files
3. **Add Real Model**: Replace simulation with trained model
4. **Production Ready**: Full stroke detection capability

## 🛠️ Troubleshooting

**Port Already in Use?**
```bash
# Change port in local_demo.py line:
port = int(os.environ.get("PORT", 8001))  # Use 8001 instead
```

**Module Not Found?**
```bash
# Install missing packages:
pip install fastapi uvicorn python-multipart Pillow numpy
```

---

**Ready to test? Run `start_local_demo.bat` and visit http://localhost:8000/demo! 🎉**