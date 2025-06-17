#!/usr/bin/env python3
"""
CUDA Enhanced Model - EXACT 74 Features
Remove vol_2hour to get exactly 74 features matching original baseline
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

class EnhancedTradingNetwork(nn.Module):
    """Enhanced Trading Neural Network - EXACT Architecture"""
    
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1: 74 -> 256
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Layer 2: 256 -> 128  
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Layer 3: 128 -> 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Layer 4: 64 -> 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output: 32 -> 3
            nn.Linear(32, 3)
        )
    
    def forward(self, x):
        return self.network(x)

def prepare_data():
    """Load and prepare data with EXACTLY 74 features."""
    
    print("📊 Loading and preparing data...")
    
    # Load data
    df = pd.read_csv('data/processed/es_features_enhanced.csv', index_col=0, parse_dates=True)
    print(f"   Raw data shape: {df.shape}")
    
    # Get all non-target columns
    feature_columns = [col for col in df.columns if col not in ['target']]
    print(f"   All feature columns: {len(feature_columns)}")
    
    # CRITICAL: Remove vol_2hour to get exactly 74 features
    if 'vol_2hour' in feature_columns:
        feature_columns.remove('vol_2hour')
        print(f"   ✅ Removed vol_2hour")
    
    # Verify we have exactly 74 features
    if len(feature_columns) != 74:
        print(f"   ❌ ERROR: Expected 74 features, got {len(feature_columns)}")
        print(f"   Features: {sorted(feature_columns)}")
        raise ValueError(f"Feature count mismatch: {len(feature_columns)} != 74")
    
    print(f"   ✅ EXACT 74 features confirmed")
    
    # Prepare features and target
    X = df[feature_columns].fillna(0)
    y = df['target']
    
    # Remove any rows with infinite values
    mask = ~np.isinf(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    
    print(f"   Final data shape: {X.shape}")
    print(f"   Features: {X.shape[1]}")
    
    # Verify target distribution
    target_dist = y.value_counts(normalize=True).sort_index()
    print(f"   Target distribution: FLAT={target_dist[0]:.1%}, LONG={target_dist[1]:.1%}, SHORT={target_dist[2]:.1%}")
    
    return X, y, feature_columns

def train_model(X, y, device):
    """Train the enhanced model with exact specifications."""
    
    print(f"\n🚀 Training Enhanced Model on {device}...")
    
    # Split data with exact same random state
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"   Train: {X_train.shape[0]:,} samples")
    print(f"   Val: {X_val.shape[0]:,} samples")
    print(f"   Test: {X_test.shape[0]:,} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train.values).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.LongTensor(y_val.values).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.LongTensor(y_test.values).to(device)
    
    # Initialize model
    model = EnhancedTradingNetwork(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    batch_size = 1024
    num_epochs = 100
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # Calculate validation accuracy
            _, val_pred = torch.max(val_outputs, 1)
            val_acc = (val_pred == y_val_tensor).float().mean().item()
        
        scheduler.step(val_loss)
        avg_train_loss = train_loss / num_batches
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.1%}")
        
        if patience_counter >= 20:
            print(f"   Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Test predictions
        test_outputs = model(X_test_tensor)
        _, test_pred = torch.max(test_outputs, 1)
        
        # Calculate metrics
        test_acc = accuracy_score(y_test, test_pred.cpu().numpy())
        
        # Win rate calculation (exclude FLAT predictions)
        non_flat_mask = test_pred.cpu().numpy() != 0
        if non_flat_mask.sum() > 0:
            correct_predictions = (test_pred.cpu().numpy() == y_test.values)[non_flat_mask]
            win_rate = correct_predictions.mean()
        else:
            win_rate = 0.0
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Test Accuracy: {test_acc:.1%}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Non-FLAT predictions: {non_flat_mask.sum():,} ({non_flat_mask.mean():.1%})")
    
    return {
        'model': model,
        'scaler': scaler,
        'test_accuracy': test_acc,
        'win_rate': win_rate,
        'test_predictions': test_pred.cpu().numpy(),
        'test_actual': y_test.values,
        'feature_columns': X.columns.tolist()
    }

def main():
    """Main execution function."""
    
    print("🎯 ENHANCED MODEL - EXACT 74 FEATURES")
    print("=" * 50)
    
    # Setup CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Prepare data
    X, y, feature_columns = prepare_data()
    
    # Train model
    results = train_model(X, y, device)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/cuda_enhanced_exact_74_model_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model_state_dict': results['model'].state_dict(),
            'scaler': results['scaler'],
            'feature_columns': feature_columns,
            'device': str(device)
        }, f)
    
    # Save detailed results
    os.makedirs('reports/enhanced_exact_74_cuda', exist_ok=True)
    results_path = f'reports/enhanced_exact_74_cuda/cuda_enhanced_exact_74_results_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'model_type': 'Enhanced_Exact_74_Features',
            'device': str(device),
            'features_count': len(feature_columns),
            'removed_feature': 'vol_2hour',
            'test_accuracy': float(results['test_accuracy']),
            'win_rate': float(results['win_rate']),
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'feature_columns': feature_columns
        }, f, indent=2)
    
    print(f"\n💾 Results saved:")
    print(f"   Model: {model_path}")
    print(f"   Results: {results_path}")
    
    print(f"\n🎯 KEY METRICS:")
    print(f"   Features: {len(feature_columns)} (target: 74)")
    print(f"   Win Rate: {results['win_rate']:.1%} (target: 50.88%)")
    print(f"   Test Accuracy: {results['test_accuracy']:.1%}")
    
    if results['win_rate'] > 0.45:
        print(f"   ✅ Model performance acceptable")
    else:
        print(f"   ❌ Model performance below threshold")
    
    return results

if __name__ == "__main__":
    results = main() 