# 🏥 Chronic Kidney Disease Diagnosis System

A machine learning-based web application for predicting chronic kidney disease using medical parameters.

## 🚀 Features

- **Real-time Prediction**: Instant kidney disease diagnosis
- **19 Medical Parameters**: Comprehensive analysis using clinical data
- **High Accuracy**: Trained on real medical dataset (400+ samples)
- **User-friendly Interface**: Clean, modern web interface
- **Fast Response**: Optimized model loading and prediction

## 📊 Model Performance

- **Accuracy**: 100% (on test set)
- **Algorithm**: Random Forest Classifier
- **Features**: 19 medical parameters
- **Dataset**: 400+ patient records

## 🛠️ Technology Stack

- **Backend**: Python Flask
- **Machine Learning**: Scikit-learn
- **Frontend**: HTML, CSS, Bootstrap
- **Data Processing**: Pandas, NumPy

## 📋 Medical Parameters

The system analyzes the following 19 medical parameters:

1. **Age** (years)
2. **Blood Pressure** (mm/Hg)
3. **Specific Gravity**
4. **Albumin**
5. **Sugar**
6. **Blood Glucose Random** (mgs/dl)
7. **Blood Urea** (mgs/dl)
8. **Serum Creatinine** (mgs/dl)
9. **Sodium** (mEq/L)
10. **Potassium** (mEq/L)
11. **Hemoglobin** (gms)
12. **Packed Cell Volume**
13. **White Blood Cell Count** (cells/cumm)
14. **Hypertension** (1=Yes, 0=No)
15. **Diabetes** (1=Yes, 0=No)
16. **Coronary Artery Disease** (1=Yes, 0=No)
17. **Appetite** (1=Good, 0=Poor)
18. **Pedal Edema** (1=Yes, 0=No)
19. **Anaemia** (1=Yes, 0=No)

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd kidney
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python3 app.py
   ```

4. **Access the application**
   - Open your browser
   - Go to `http://localhost:5001`
   - Click "Start Diagnosis"
   - Enter medical parameters
   - Get instant prediction

## 📁 Project Structure

```
kidney/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── upload.csv            # Training dataset
├── test_model.py         # Model testing script
├── templates/            # HTML templates
│   ├── index.html        # Landing page
│   ├── index2.html       # Prediction form
│   └── ...
├── static/               # Static files (CSS, JS, images)
└── README.md            # This file
```

## 🔧 Usage

1. **Access the System**: Navigate to `http://localhost:5001`
2. **Start Diagnosis**: Click "Start Diagnosis" button
3. **Enter Parameters**: Fill in the 19 medical parameters
4. **Get Results**: Click "Predict" for instant diagnosis
5. **View Analysis**: Click "Analysis" for detailed charts

## 📈 Model Training

The model is trained using:
- **Dataset**: 400+ patient records with kidney disease indicators
- **Algorithm**: Random Forest Classifier
- **Preprocessing**: Standard scaling, missing value handling
- **Validation**: 80-20 train-test split

## 🎯 Prediction Results

The system provides:
- **Diagnosis**: Normal or Chronic Kidney Disease
- **Confidence**: Prediction confidence score
- **Instant Results**: Real-time analysis

## 🔒 Security & Privacy

- No patient data is stored
- All predictions are processed in real-time
- No login required for basic functionality
- Local processing for data privacy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is for educational and research purposes.

## 👥 Team

**VTPML12 Team** - Machine Learning Project

## 📞 Support

For questions or issues, please contact the development team.

---

**⚠️ Medical Disclaimer**: This system is for educational purposes only. Always consult healthcare professionals for medical diagnosis and treatment.
