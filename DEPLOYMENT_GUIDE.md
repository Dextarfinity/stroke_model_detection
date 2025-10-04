# 🚀 Railway Deployment Guide

## Quick Deployment Steps

### 1. Upload to GitHub
1. Create a new repository on GitHub (e.g., "stroke-detection-api")
2. Upload all files from this `railway_deployment` folder
3. Commit the changes

### 2. Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Sign up/login with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your repository
6. Railway automatically detects Dockerfile and deploys

### 3. Access Your API
- Your app URL: `https://stroke-detection-api-production.up.railway.app`
- Demo page: `https://your-app.railway.app/demo`
- API documentation: `https://your-app.railway.app/docs`

## Files Included

✅ `main.py` - FastAPI application
✅ `requirements.txt` - Python dependencies
✅ `Dockerfile` - Container configuration
✅ `Procfile` - Process definition
✅ `railway.json` - Railway settings
✅ `demo.html` - Interactive web demo
✅ `.gitignore` - Git ignore rules
✅ `README.md` - Documentation

## What Happens During Deployment

1. **Build Phase:**
   - Railway uses Dockerfile
   - Installs Python dependencies
   - Sets up environment

2. **Deploy Phase:**
   - Starts FastAPI server
   - Exposes on Railway's domain
   - Enables HTTPS automatically

3. **Runtime:**
   - Downloads YOLO model on first request
   - Serves API endpoints
   - Handles image uploads and predictions

## Environment Configuration

No manual environment variables needed! Railway automatically:
- Sets `PORT` environment variable
- Provides HTTPS certificates
- Configures networking

## Monitoring & Logs

Access Railway dashboard to:
- View deployment logs
- Monitor resource usage
- Check application status
- View error logs

## Troubleshooting

### Common Issues:

**Build Fails:**
- Check Railway build logs
- Verify all files are committed
- Ensure Dockerfile syntax is correct

**App Crashes:**
- Check application logs in Railway dashboard
- Verify Python dependencies are correct
- Check memory limits (Railway free tier)

**Slow Performance:**
- YOLO model loads on first request (30-60 seconds)
- Subsequent requests are faster
- Consider upgrading Railway plan for better performance

### Memory Optimization:
- Uses YOLO11n (smallest model) for efficiency
- Optimized for Railway's resource limits
- Automatic model caching

## Cost & Limits

**Free Tier:**
- $5 monthly credit
- Suitable for testing and small projects
- Automatic sleep after inactivity

**Paid Plans:**
- Start at $5/month
- Better performance and reliability
- No sleep mode

## Next Steps

1. **Custom Domain (Optional):**
   - Available on paid plans
   - Configure in Railway dashboard

2. **Environment Variables (If Needed):**
   - Add through Railway dashboard
   - No special variables required for basic setup

3. **Scaling:**
   - Railway handles automatic scaling
   - Monitor usage in dashboard

## Support Resources

- **Railway Documentation:** [docs.railway.app](https://docs.railway.app)
- **Railway Discord:** Community support
- **Railway Status:** [status.railway.app](https://status.railway.app)

---

**Your API will be live in 5-10 minutes after deployment! 🎉**