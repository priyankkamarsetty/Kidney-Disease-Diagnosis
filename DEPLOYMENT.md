# 🚀 Deployment Guide

## GitHub Setup

### 1. Create GitHub Repository
1. Go to [GitHub.com](https://github.com)
2. Click "New repository"
3. Name it: `kidney-disease-diagnosis`
4. Make it public
5. Don't initialize with README (we already have one)

### 2. Push to GitHub
```bash
# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/kidney-disease-diagnosis.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Local Development

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python3 app.py

# Access at: http://localhost:5001
```

### Testing the Model
```bash
# Test the model independently
python3 test_model.py
```

## Production Deployment

### Option 1: Heroku
1. Create `Procfile` (already exists)
2. Install Heroku CLI
3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Option 2: Python Anywhere
1. Upload files to PythonAnywhere
2. Set up virtual environment
3. Install requirements
4. Configure WSGI file

### Option 3: Local Server
```bash
# For production use
gunicorn app:app -b 0.0.0.0:5001
```

## Performance Optimization

### Model Caching
- Model is automatically saved to `kidney_model.pkl`
- Faster startup on subsequent runs
- No need to retrain every time

### Response Time
- Average prediction time: < 1 second
- Model loading: ~2-3 seconds (first time only)
- Cached model loading: < 1 second

## Troubleshooting

### Common Issues
1. **Port already in use**: Change port in `app.py`
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Model not loading**: Delete `kidney_model.pkl` to retrain

### Logs
- Check console output for detailed logs
- Model training progress is displayed
- Prediction results are logged

## Security Notes
- No authentication required for demo
- Add authentication for production use
- Consider HTTPS for sensitive data
- Validate all input parameters

## Monitoring
- Application logs in console
- Model accuracy metrics
- Response time monitoring
- Error tracking
