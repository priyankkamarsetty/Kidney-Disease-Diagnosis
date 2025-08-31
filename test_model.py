#!/usr/bin/env python3
"""
Test script for the Kidney Disease Prediction Model
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def test_model():
    """Test the kidney disease prediction model"""
    
    print("🧪 Testing Kidney Disease Prediction Model")
    print("=" * 50)
    
    try:
        # Load the dataset
        print("📊 Loading dataset...")
        df = pd.read_csv('upload.csv')
        print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Clean the classification column - convert to binary
        print("🧹 Cleaning classification data...")
        df['classification'] = (df['classification'] > 0.5).astype(int)
        print(f"Target distribution after cleaning: {df['classification'].value_counts()}")
        
        # Check feature names
        feature_columns = [col for col in df.columns if col not in ['Id', 'classification']]
        print(f"📋 Features: {feature_columns}")
        
        # Prepare data
        X = df[feature_columns].values
        y = df['classification'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Split data (without stratification to avoid issues)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        print("🤖 Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully!")
        print(f"📈 Training accuracy: {accuracy:.4f}")
        print(f"📊 Test accuracy: {accuracy:.4f}")
        
        # Test with sample data
        print("\n🧪 Testing with sample data...")
        
        # Sample 1: Normal case (from dataset)
        sample1 = X_test[0]  # First test sample
        sample1_scaled = scaler.transform(sample1.reshape(1, -1))
        pred1 = model.predict(sample1_scaled)[0]
        prob1 = model.predict_proba(sample1_scaled)[0]
        
        print(f"Sample 1 - Prediction: {'Chronic Kidney Disease' if pred1 > 0.5 else 'Normal'}")
        print(f"Sample 1 - Confidence: {np.max(prob1):.3f}")
        
        # Sample 2: Another case
        sample2 = X_test[1]  # Second test sample
        sample2_scaled = scaler.transform(sample2.reshape(1, -1))
        pred2 = model.predict(sample2_scaled)[0]
        prob2 = model.predict_proba(sample2_scaled)[0]
        
        print(f"Sample 2 - Prediction: {'Chronic Kidney Disease' if pred2 > 0.5 else 'Normal'}")
        print(f"Sample 2 - Confidence: {np.max(prob2):.3f}")
        
        # Test with manual input
        print("\n🧪 Testing with manual input...")
        manual_input = np.array([50, 80, 1.02, 1, 0, 121, 36, 1.2, 135, 3.0, 15.4, 44, 7800, 1, 1, 0, 1, 0, 0])
        manual_scaled = scaler.transform(manual_input.reshape(1, -1))
        manual_pred = model.predict(manual_scaled)[0]
        manual_prob = model.predict_proba(manual_scaled)[0]
        
        print(f"Manual Input - Prediction: {'Chronic Kidney Disease' if manual_pred > 0.5 else 'Normal'}")
        print(f"Manual Input - Confidence: {np.max(manual_prob):.3f}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    test_model()
