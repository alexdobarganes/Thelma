#!/usr/bin/env python3
"""
Create Simple Model for Validation
==================================

Creates a simple scikit-learn LogisticRegression model as backup
for system validation when the production model is complex.
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_simple_model():
    """Create and save a simple validation model."""
    project_root = Path(__file__).parent.parent.parent
    
    print("🧠 Creating simple validation model...")
    
    # Load processed data
    data_path = project_root / 'data' / 'processed' / 'es_features_enhanced.csv'
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['target', 'timestamp']]
    X = df[feature_cols].fillna(0)
    y = df['target']
    
    # Sample for quick training
    X_sample = X.sample(n=10000, random_state=42)
    y_sample = y.loc[X_sample.index]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"✅ Model trained - Train: {train_score:.4f}, Test: {test_score:.4f}")
    
    # Save simple model
    simple_model_path = project_root / 'models' / 'production' / 'current' / 'simple_model.pkl'
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'performance': {
            'train_score': train_score,
            'test_score': test_score
        }
    }
    
    with open(simple_model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"💾 Simple model saved to: {simple_model_path}")
    
    return simple_model_path

if __name__ == "__main__":
    create_simple_model() 