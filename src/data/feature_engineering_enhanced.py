#!/usr/bin/env python3
"""
Enhanced Feature Engineering Pipeline for ES Futures ML Strategy
Implements multi-timeframe features and dynamic target engineering for improved performance.

Key Improvements:
1. Multi-timeframe features (1min, 5min, 15min, 1hour)
2. Dynamic target engineering based on volatility
3. Advanced market microstructure features
4. Volatility regime detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ESFeatureEngineerEnhanced:
    """Enhanced feature engineering pipeline with multi-timeframe and dynamic targets."""
    
    def __init__(self, 
                 lookback_window: int = 14,
                 normalization_window: int = 100):
        """Initialize enhanced feature engineering pipeline."""
        self.lookback_window = lookback_window
        self.normalization_window = normalization_window
        
        # Multi-timeframe definitions (in minutes)
        self.timeframes = {
            '1min': 1,
            '5min': 5, 
            '15min': 15,
            '1hour': 60
        }
        
        logger.info(f"🚀 Initialized Enhanced Feature Engineer with multi-timeframe support")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and prepare raw OHLCV data."""
        logger.info(f"Loading data from {file_path}")
        
        # Load with proper column names
        df = pd.read_csv(file_path, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'], skiprows=1)
        
        # Convert timestamp and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Ensure numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"📈 Loaded {len(df):,} records from {df.index.min()} to {df.index.max()}")
        return df
    
    def create_multitimeframe_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create multiple timeframe datasets."""
        logger.info("🕐 Creating multi-timeframe datasets...")
        
        mtf_data = {}
        
        # 1-minute data (original)
        mtf_data['1min'] = df.copy()
        
        # Resample to other timeframes
        for tf_name, minutes in self.timeframes.items():
            if tf_name != '1min':
                logger.info(f"  Creating {tf_name} timeframe...")
                
                # Resample OHLCV data
                resampled = df.resample(f'{minutes}min').agg({
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                mtf_data[tf_name] = resampled
                logger.info(f"    {tf_name}: {len(resampled):,} bars")
        
        return mtf_data
    
    def add_volatility_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime detection features."""
        logger.info("💹 Computing volatility regime features...")
        
        # Calculate various volatility measures
        df['returns'] = df['close'].pct_change()
        
        # Realized volatility (different windows)
        df['vol_5min'] = df['returns'].rolling(5).std() * np.sqrt(252 * 24 * 12)  # Annualized
        df['vol_30min'] = df['returns'].rolling(30).std() * np.sqrt(252 * 24 * 12)
        df['vol_2hour'] = df['returns'].rolling(120).std() * np.sqrt(252 * 24 * 12)
        df['vol_1day'] = df['returns'].rolling(1440).std() * np.sqrt(252 * 24 * 12)
        
        # Volatility ratios (current vs historical)
        df['vol_regime_short'] = df['vol_5min'] / df['vol_30min']
        df['vol_regime_medium'] = df['vol_30min'] / df['vol_2hour']
        df['vol_regime_long'] = df['vol_2hour'] / df['vol_1day']
        
        # Volatility percentiles (regime detection)
        df['vol_percentile_30d'] = df['vol_30min'].rolling(30*24*60).rank(pct=True)
        df['vol_percentile_7d'] = df['vol_30min'].rolling(7*24*60).rank(pct=True)
        
        # High/Low volatility regimes
        df['high_vol_regime'] = (df['vol_percentile_7d'] > 0.8).astype(int)
        df['low_vol_regime'] = (df['vol_percentile_7d'] < 0.2).astype(int)
        
        return df
    
    def add_multitimeframe_technical_features(self, mtf_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add technical features from multiple timeframes."""
        logger.info("📊 Computing multi-timeframe technical features...")
        
        # Start with 1-minute base data
        base_df = mtf_data['1min'].copy()
        
        # Add features from each timeframe
        for tf_name, tf_df in mtf_data.items():
            if tf_name == '1min':
                continue
                
            logger.info(f"  Processing {tf_name} features...")
            
            # Calculate technical indicators for this timeframe
            tf_features = self._calculate_technical_indicators(tf_df, tf_name)
            
            # Merge back to 1-minute data using forward fill
            base_df = base_df.join(tf_features, how='left').fillna(method='ffill')
        
        return base_df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Calculate technical indicators for a specific timeframe."""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features[f'{prefix}_returns'] = df['close'].pct_change()
        features[f'{prefix}_log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        features[f'{prefix}_ema_9'] = df['close'].ewm(span=9).mean()
        features[f'{prefix}_ema_21'] = df['close'].ewm(span=21).mean()
        features[f'{prefix}_ema_50'] = df['close'].ewm(span=50).mean()
        
        # Price relative to EMAs
        features[f'{prefix}_price_to_ema9'] = df['close'] / features[f'{prefix}_ema_9']
        features[f'{prefix}_price_to_ema21'] = df['close'] / features[f'{prefix}_ema_21']
        features[f'{prefix}_ema9_to_ema21'] = features[f'{prefix}_ema_9'] / features[f'{prefix}_ema_21']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features[f'{prefix}_rsi'] = 100 - (100 / (1 + rs))
        features[f'{prefix}_rsi_normalized'] = (features[f'{prefix}_rsi'] - 50) / 50
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features[f'{prefix}_atr'] = true_range.rolling(14).mean()
        features[f'{prefix}_atr_ratio'] = features[f'{prefix}_atr'] / df['close']
        
        # Bollinger Bands
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        features[f'{prefix}_bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        features[f'{prefix}_bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # Volume features
        features[f'{prefix}_volume_ma'] = df['volume'].rolling(20).mean()
        features[f'{prefix}_volume_ratio'] = df['volume'] / features[f'{prefix}_volume_ma']
        
        # Price momentum
        features[f'{prefix}_momentum_3'] = df['close'] / df['close'].shift(3) - 1
        features[f'{prefix}_momentum_10'] = df['close'] / df['close'].shift(10) - 1
        features[f'{prefix}_momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        return features
    
    def create_dynamic_target_variable(self, df: pd.DataFrame, 
                                     base_threshold: float = 0.0005,
                                     prediction_horizon: int = 5,
                                     volatility_adjustment: bool = True) -> pd.DataFrame:
        """Create dynamic target variable based on volatility regime."""
        logger.info(f"🎯 Creating dynamic target variable...")
        logger.info(f"   Base threshold: {base_threshold:.4f} ({base_threshold*100:.2f}%)")
        logger.info(f"   Prediction horizon: {prediction_horizon} minutes")
        logger.info(f"   Volatility adjustment: {volatility_adjustment}")
        
        # Calculate future returns
        future_price = df['close'].shift(-prediction_horizon)
        future_returns = (future_price - df['close']) / df['close']
        
        if volatility_adjustment:
            # Dynamic threshold based on realized volatility
            vol_window = 30  # 30-minute rolling volatility
            current_vol = df['returns'].rolling(vol_window).std()
            
            # Adjust threshold: higher vol = higher threshold needed
            vol_multiplier = np.clip(current_vol / current_vol.rolling(240).mean(), 0.5, 3.0)  # 4-hour average
            dynamic_threshold = base_threshold * vol_multiplier
            
            # Smooth the threshold to avoid jumps
            dynamic_threshold = dynamic_threshold.rolling(10).mean().fillna(base_threshold)
            
            logger.info(f"   Dynamic threshold range: {dynamic_threshold.min():.4f} - {dynamic_threshold.max():.4f}")
        else:
            dynamic_threshold = base_threshold
        
        # Create target with dynamic thresholds
        df['target'] = 0  # Default: FLAT
        
        if volatility_adjustment:
            # Use different threshold for each row
            long_condition = future_returns > dynamic_threshold
            short_condition = future_returns < -dynamic_threshold
        else:
            # Use static threshold
            long_condition = future_returns > base_threshold
            short_condition = future_returns < -base_threshold
        
        df.loc[long_condition, 'target'] = 1   # LONG
        df.loc[short_condition, 'target'] = 2  # SHORT
        
        # Store dynamic threshold for analysis
        if volatility_adjustment:
            df['target_threshold'] = dynamic_threshold
        
        # Target distribution analysis
        target_dist = df['target'].value_counts(normalize=True)
        logger.info(f"🎲 Target distribution:")
        logger.info(f"   FLAT (0): {target_dist[0]:.3f} ({target_dist[0]*100:.1f}%)")
        logger.info(f"   LONG (1): {target_dist[1]:.3f} ({target_dist[1]*100:.1f}%)")
        logger.info(f"   SHORT (2): {target_dist[2]:.3f} ({target_dist[2]*100:.1f}%)")
        
        return df
    
    def process_enhanced_pipeline(self, 
                                file_path: str,
                                base_threshold: float = 0.0003,  # Reduced from 0.0005
                                prediction_horizon: int = 5,
                                volatility_adjustment: bool = True,
                                output_path: str = None) -> pd.DataFrame:
        """Run enhanced feature engineering pipeline."""
        logger.info("🚀 Starting ENHANCED feature engineering pipeline...")
        
        # Load base data
        df = self.load_data(file_path)
        
        # Create multi-timeframe datasets
        mtf_data = self.create_multitimeframe_data(df)
        
        # Add volatility regime features to base 1-min data
        df = self.add_volatility_regime_features(mtf_data['1min'])
        
        # Add multi-timeframe technical features
        df = self.add_multitimeframe_technical_features(mtf_data)
        
        # Create dynamic target variable (LAST - after all features)
        df = self.create_dynamic_target_variable(
            df, 
            base_threshold=base_threshold,
            prediction_horizon=prediction_horizon,
            volatility_adjustment=volatility_adjustment
        )
        
        # Clean data
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        retention_rate = final_rows / initial_rows
        
        logger.info(f"✅ Enhanced pipeline complete!")
        logger.info(f"📊 Data: {initial_rows:,} → {final_rows:,} rows ({retention_rate:.2%} retained)")
        logger.info(f"🎯 Features: {len(df.columns)} total columns")
        
        # Save enhanced features
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path)
            logger.info(f"💾 Enhanced features saved to {output_path}")
        
        return df


def main():
    """Test the enhanced feature engineering pipeline."""
    
    # Initialize enhanced feature engineer
    engineer = ESFeatureEngineerEnhanced(
        lookback_window=14,
        normalization_window=100
    )
    
    # Process ES data with enhanced pipeline
    input_file = "data/es_1m/market_data.csv"
    output_file = "data/processed/es_features_enhanced.csv"
    
    try:
        features_df = engineer.process_enhanced_pipeline(
            file_path=input_file,
            base_threshold=0.0003,      # More sensitive threshold (3 bps)
            prediction_horizon=5,       # 5-minute prediction horizon  
            volatility_adjustment=True, # Dynamic volatility-based thresholds
            output_path=output_file
        )
        
        print(f"\n🎉 ENHANCED Feature Engineering Complete!")
        print(f"📁 Input: {input_file}")
        print(f"💾 Output: {output_file}")
        print(f"📊 Enhanced Features: {features_df.shape[1]} columns, {features_df.shape[0]:,} rows")
        print(f"📅 Date range: {features_df.index.min()} to {features_df.index.max()}")
        
        # Enhanced target analysis
        target_counts = features_df['target'].value_counts()
        print(f"\n🎯 Enhanced Target Distribution:")
        print(f"   • FLAT (0): {target_counts[0]:,} ({target_counts[0]/len(features_df):.1%})")
        print(f"   • LONG (1): {target_counts[1]:,} ({target_counts[1]/len(features_df):.1%})")  
        print(f"   • SHORT (2): {target_counts[2]:,} ({target_counts[2]/len(features_df):.1%})")
        
        print(f"\n🚀 EXPECTED IMPROVEMENTS:")
        print(f"   • Multi-timeframe features should improve trend detection")
        print(f"   • Dynamic targets should improve signal quality") 
        print(f"   • Target: F1 Score >0.44, Sharpe Ratio >0.10")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Enhanced pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main() 