# 🧠 Stroke Detection API

A FastAPI-based web service for detecting stroke symptoms in facial images using YOLO object detection.

## 🚀 Quick Start

This repository is ready for immediate deployment to Railway!

### 🌐 Deploy to Railway

1. **Upload this folder to GitHub:**
   - Create a new repository on GitHub
   - Upload all files from this `railway_deployment` folder
   - Commit the changes

2. **Deploy to Railway:**
   - Go to [railway.app](https://railway.app)
   - Sign in with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will automatically detect the Dockerfile and deploy

3. **Access your API:**
   - Railway provides a URL like: `https://your-app.railway.app`
   - Demo page: `https://your-app.railway.app/demo`
   - API docs: `https://your-app.railway.app/docs`

## 🎯 Features

### API Endpoints
- **`GET /`** - Health check and API info
- **`GET /demo`** - Interactive web demo
- **`GET /health`** - Health status
- **`POST /predict`** - Single image analysis
- **`POST /batch_predict`** - Multiple image analysis (max 10)
- **`GET /model_info`** - Model information
- **`GET /docs`** - Swagger API documentation

### Detection Classes
The model detects 8 stroke-related facial features:
- `normalEye` - Normal eye appearance
- `normalMouth` - Normal mouth appearance
- `strokeEyeMid` - Moderate stroke symptoms in eye
- `strokeEyeSevere` - Severe stroke symptoms in eye
- `strokeEyeWeak` - Mild stroke symptoms in eye
- `strokeMouthMid` - Moderate stroke symptoms in mouth
- `strokeMouthSevere` - Severe stroke symptoms in mouth
- `strokeMouthWeak` - Mild stroke symptoms in mouth

## 📱 Usage Examples

### Web Demo
Visit `/demo` on your deployed app for a drag-and-drop interface.

### API Usage (Python)
```python
import requests

# Single image prediction
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('https://your-app.railway.app/predict', files=files)
    result = response.json()
    print(result)
```

### API Usage (JavaScript)
```javascript
const formData = new FormData();
formData.append('file', imageFile);

fetch('https://your-app.railway.app/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

### API Usage (cURL)
```bash
curl -X POST "https://your-app.railway.app/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

## 📊 Response Format

```json
{
  "filename": "image.jpg",
  "predictions": [
    {
      "class_id": 2,
      "class_name": "strokeEyeMid",
      "confidence": 0.85,
      "bbox": {
        "x1": 100, "y1": 150,
        "x2": 200, "y2": 250
      }
    }
  ],
  "stroke_analysis": {
    "stroke_detected": true,
    "confidence_level": "high",
    "max_severity": "moderate",
    "affected_areas": ["strokeEyeMid"],
    "recommendation": "Immediate medical attention recommended"
  },
  "total_detections": 1
}
```

## 🔧 Technical Details

- **Framework:** FastAPI with async support
- **ML Model:** YOLO11n (optimized for cloud deployment)
- **Image Processing:** OpenCV + Pillow
- **Container:** Docker with Python 3.11
- **Deployment:** Railway with automatic scaling

## ⚠️ Medical Disclaimer

**Important:** This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice and diagnosis.

## 🛠️ Local Development

If you want to run locally:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python main.py
   ```

3. **Access locally:**
   - API: `http://localhost:8000`
   - Demo: `http://localhost:8000/demo`
   - Docs: `http://localhost:8000/docs`

## 📈 Railway Deployment Benefits

- ✅ Automatic HTTPS
- ✅ Global CDN
- ✅ Automatic scaling
- ✅ Built-in monitoring
- ✅ Easy custom domains
- ✅ Environment variables support
- ✅ Continuous deployment from GitHub

## 💰 Cost Information

- **Free Tier:** $5 credit monthly (sufficient for testing)
- **Usage-based:** Pay for actual compute time
- **Scaling:** Automatic based on traffic

## 🔄 Updates

To update your deployment:
1. Make changes to your code
2. Push to GitHub
3. Railway automatically redeploys

## 📞 Support

- **Railway Docs:** [docs.railway.app](https://docs.railway.app)
- **Railway Discord:** Active community support
- **Issues:** Create GitHub issues for bugs

---

**Ready to deploy? Just upload this folder to GitHub and connect it to Railway! 🚀**