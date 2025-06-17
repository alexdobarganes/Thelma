#!/usr/bin/env python3
"""
Comprehensive Model Comparison for ES Futures Trading Strategy
Compare multiple ML models to find the optimal architecture for trading signals.

Models tested:
- LightGBM (gradient boosting)
- XGBoost (gradient boosting)
- CatBoost (gradient boosting)
- ExtraTrees (ensemble)
- LogisticRegression (linear baseline)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
import time
import json

# ML Libraries
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveModelComparison:
    """Compare multiple ML models for ES futures trading prediction."""
    
    def __init__(self, results_dir: str = "reports/comprehensive_comparison"):
        """Initialize comprehensive model comparison framework."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Comprehensive Model Comparison, results will be saved to {results_dir}")
    
    def load_features(self, file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load engineered features and target variable."""
        logger.info(f"Loading features from {file_path}")
        
        df = pd.read_csv(file_path, index_col=0)
        
        # Separate features and target
        target = df['target']
        features = df.drop(['target'], axis=1)
        
        # Remove any remaining NaN values
        mask = ~(features.isna().any(axis=1) | target.isna())
        features = features[mask]
        target = target[mask]
        
        logger.info(f"Loaded {len(features):,} samples with {len(features.columns)} features")
        logger.info(f"Target distribution: {target.value_counts().to_dict()}")
        
        return features, target
    
    def prepare_data_splits(self, features: pd.DataFrame, target: pd.Series, 
                           test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, Any]:
        """Prepare time-series aware data splits."""
        logger.info("Preparing time-series data splits...")
        
        # Sort by timestamp to maintain temporal order
        features = features.sort_index()
        target = target.sort_index()
        
        n_samples = len(features)
        test_start = int(n_samples * (1 - test_size))
        val_start = int(test_start * (1 - val_size))
        
        # Time-series splits (no shuffling)
        train_features = features.iloc[:val_start]
        train_target = target.iloc[:val_start]
        
        val_features = features.iloc[val_start:test_start]
        val_target = target.iloc[val_start:test_start]
        
        test_features = features.iloc[test_start:]
        test_target = target.iloc[test_start:]
        
        logger.info(f"Data splits - Train: {len(train_features):,}, Val: {len(val_features):,}, Test: {len(test_features):,}")
        
        return {
            'train': (train_features, train_target),
            'val': (val_features, val_target),
            'test': (test_features, test_target)
        }
    
    def train_lightgbm(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """Train LightGBM model."""
        logger.info("Training LightGBM model...")
        
        X_train, y_train = data_splits['train']
        X_val, y_val = data_splits['val']
        
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        start_time = time.time()
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        training_time = time.time() - start_time
        
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        val_pred_classes = np.argmax(val_pred, axis=1)
        
        return {
            'model': model,
            'model_type': 'lightgbm',
            'training_time': training_time,
            'val_f1_macro': f1_score(y_val, val_pred_classes, average='macro')
        }
    
    def train_xgboost(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """Train XGBoost model."""
        logger.info("Training XGBoost model...")
        
        X_train, y_train = data_splits['train']
        X_val, y_val = data_splits['val']
        
        start_time = time.time()
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            max_depth=6,
            learning_rate=0.05,
            n_estimators=1000,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='mlogloss',
            verbosity=0
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        training_time = time.time() - start_time
        
        val_pred = model.predict(X_val)
        
        return {
            'model': model,
            'model_type': 'xgboost',
            'training_time': training_time,
            'val_f1_macro': f1_score(y_val, val_pred, average='macro')
        }
    
    def train_catboost(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """Train CatBoost model."""
        logger.info("Training CatBoost model...")
        
        X_train, y_train = data_splits['train']
        X_val, y_val = data_splits['val']
        
        start_time = time.time()
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='MultiClass',
            eval_metric='MultiClass',
            early_stopping_rounds=50,
            random_seed=42,
            verbose=False
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        training_time = time.time() - start_time
        
        val_pred = model.predict(X_val)
        
        return {
            'model': model,
            'model_type': 'catboost',
            'training_time': training_time,
            'val_f1_macro': f1_score(y_val, val_pred, average='macro')
        }
    
    def train_extratrees(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """Train ExtraTrees model."""
        logger.info("Training ExtraTrees model...")
        
        X_train, y_train = data_splits['train']
        X_val, y_val = data_splits['val']
        
        start_time = time.time()
        model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        val_pred = model.predict(X_val)
        
        return {
            'model': model,
            'model_type': 'extratrees',
            'training_time': training_time,
            'val_f1_macro': f1_score(y_val, val_pred, average='macro')
        }
    
    def train_logistic(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """Train Logistic Regression model."""
        logger.info("Training Logistic Regression model...")
        
        X_train, y_train = data_splits['train']
        X_val, y_val = data_splits['val']
        
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        start_time = time.time()
        model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        val_pred = model.predict(X_val_scaled)
        
        return {
            'model': model,
            'scaler': scaler,
            'model_type': 'logistic',
            'training_time': training_time,
            'val_f1_macro': f1_score(y_val, val_pred, average='macro')
        }
    
    def evaluate_model(self, model_name: str, model_results: Dict[str, Any], 
                      data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive model evaluation including latency testing."""
        logger.info(f"Evaluating {model_name} model...")
        
        X_test, y_test = data_splits['test']
        model = model_results['model']
        model_type = model_results['model_type']
        
        # Handle scaling for logistic regression
        if model_type == 'logistic':
            scaler = model_results['scaler']
            X_test_scaled = scaler.transform(X_test)
            X_test_eval = X_test_scaled
        else:
            X_test_eval = X_test
        
        # Predictions and timing
        start_time = time.time()
        
        if model_type == 'lightgbm':
            test_pred_proba = model.predict(X_test_eval, num_iteration=model.best_iteration)
            test_pred = np.argmax(test_pred_proba, axis=1)
        else:
            test_pred = model.predict(X_test_eval)
            if hasattr(model, 'predict_proba'):
                test_pred_proba = model.predict_proba(X_test_eval)
        
        prediction_time = time.time() - start_time
        avg_latency = (prediction_time / len(X_test)) * 1000  # ms per prediction
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, test_pred)
        f1_macro = f1_score(y_test, test_pred, average='macro')
        f1_weighted = f1_score(y_test, test_pred, average='weighted')
        
        # Per-class F1 scores
        f1_per_class = f1_score(y_test, test_pred, average=None)
        
        evaluation = {
            'model_name': model_name,
            'model_type': model_type,
            'test_accuracy': accuracy,
            'test_f1_macro': f1_macro,
            'test_f1_weighted': f1_weighted,
            'test_f1_flat': f1_per_class[0],
            'test_f1_long': f1_per_class[1],
            'test_f1_short': f1_per_class[2],
            'avg_latency_ms': avg_latency,
            'total_prediction_time': prediction_time,
            'training_time': model_results['training_time'],
            'val_f1_macro': model_results['val_f1_macro']
        }
        
        logger.info(f"{model_name} evaluation complete:")
        logger.info(f"  Test F1 (macro): {f1_macro:.4f}")
        logger.info(f"  Test accuracy: {accuracy:.4f}")
        logger.info(f"  Avg latency: {avg_latency:.3f}ms")
        
        return evaluation
    
    def run_comprehensive_comparison(self, features_file: str) -> Dict[str, Any]:
        """Run comprehensive model comparison across multiple architectures."""
        logger.info("Starting comprehensive model comparison...")
        
        # Load data
        features, target = self.load_features(features_file)
        data_splits = self.prepare_data_splits(features, target)
        
        # Train all models
        models = {}
        models['lightgbm'] = self.train_lightgbm(data_splits)
        models['xgboost'] = self.train_xgboost(data_splits)
        models['catboost'] = self.train_catboost(data_splits)
        models['extratrees'] = self.train_extratrees(data_splits)
        models['logistic'] = self.train_logistic(data_splits)
        
        # Evaluate all models
        evaluations = {}
        for model_name, model_results in models.items():
            evaluations[model_name] = self.evaluate_model(model_name, model_results, data_splits)
        
        # Compare results
        comparison = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'total_samples': len(features),
                'num_features': len(features.columns),
                'target_distribution': target.value_counts().to_dict()
            },
            'models': evaluations
        }
        
        # Determine winner
        winner = self._determine_best_model(evaluations)
        comparison['winner'] = winner
        
        # Save results
        results_file = self.results_dir / "comprehensive_comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Comprehensive comparison complete. Results saved to {results_file}")
        logger.info(f"Winner: {winner['model']} (F1: {winner['f1_score']:.4f}, Latency: {winner['latency']:.3f}ms)")
        
        return comparison
    
    def _determine_best_model(self, evaluations: Dict[str, Dict]) -> Dict[str, Any]:
        """Determine best model based on multiple criteria."""
        
        # Rank models by F1 score (primary criterion)
        models_ranked = sorted(
            evaluations.items(),
            key=lambda x: x[1]['test_f1_macro'],
            reverse=True
        )
        
        best_model_name, best_eval = models_ranked[0]
        
        # Check if best model meets latency requirement
        meets_latency = best_eval['avg_latency_ms'] < 20
        
        if not meets_latency:
            # Find best model that meets latency requirement
            for model_name, eval_data in models_ranked:
                if eval_data['avg_latency_ms'] < 20:
                    best_model_name, best_eval = model_name, eval_data
                    break
        
        winner = {
            'model': best_model_name,
            'f1_score': best_eval['test_f1_macro'],
            'accuracy': best_eval['test_accuracy'],
            'latency': best_eval['avg_latency_ms'],
            'training_time': best_eval['training_time'],
            'reason': 'Best F1 score with acceptable latency' if meets_latency else 'Best F1 score among models meeting latency requirement'
        }
        
        return winner


def main():
    """Run comprehensive model comparison on ES features."""
    comparison = ComprehensiveModelComparison()
    
    try:
        results = comparison.run_comprehensive_comparison("data/processed/es_features.csv")
        
        print(f"\n🏆 COMPREHENSIVE MODEL COMPARISON RESULTS")
        print(f"=" * 60)
        
        # Sort models by F1 score for display
        models_sorted = sorted(
            results['models'].items(),
            key=lambda x: x[1]['test_f1_macro'],
            reverse=True
        )
        
        print(f"\n📊 MODEL RANKINGS (by F1 Score):")
        print(f"{'Rank':<4} {'Model':<12} {'F1 Score':<10} {'Accuracy':<10} {'Latency':<12} {'Train Time':<12}")
        print("-" * 70)
        
        for rank, (model_name, model_results) in enumerate(models_sorted, 1):
            print(f"{rank:<4} {model_name.upper():<12} "
                  f"{model_results['test_f1_macro']:.4f}   "
                  f"{model_results['test_accuracy']:.4f}   "
                  f"{model_results['avg_latency_ms']:.3f}ms      "
                  f"{model_results['training_time']:.2f}s")
        
        winner = results['winner']
        print(f"\n🥇 FINAL WINNER: {winner['model'].upper()}")
        print(f"   F1 Score: {winner['f1_score']:.4f}")
        print(f"   Accuracy: {winner['accuracy']:.4f}")
        print(f"   Latency: {winner['latency']:.3f}ms")
        print(f"   Training Time: {winner['training_time']:.2f}s")
        print(f"   Reason: {winner['reason']}")
        
        print(f"\n💾 Full results saved to: reports/comprehensive_comparison/comprehensive_comparison_results.json")
        
        return results
        
    except Exception as e:
        logger.error(f"Comprehensive comparison failed: {e}")
        raise


if __name__ == "__main__":
    main() 