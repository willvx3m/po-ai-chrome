import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.ensemble import IsolationForest
import logging

class EnhancedDataProcessor:
    """Enhanced data preprocessing with outlier detection and cleaning."""
    
    def __init__(self, outlier_method='modified_zscore', outlier_threshold=3.5, use_robust_scaler=True):
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.use_robust_scaler = use_robust_scaler
        
        if use_robust_scaler:
            self.scaler = RobustScaler(quantile_range=(25.0, 75.0))
        else:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        
        self.outlier_stats = {}
        
    def detect_outliers_iqr(self, data, threshold=1.5):
        """Detect outliers using Interquartile Range method."""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = (data < lower_bound) | (data > upper_bound)
        return outliers
    
    def detect_outliers_zscore(self, data, threshold=3.0):
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        outliers = z_scores > threshold
        return outliers
    
    def detect_outliers_modified_zscore(self, data, threshold=3.5):
        """Detect outliers using Modified Z-score (more robust)."""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            # If MAD is 0, use standard deviation as fallback
            mad = np.std(data)
        
        if mad == 0:
            # If still 0, no outliers
            return np.zeros(len(data), dtype=bool)
        
        modified_z_scores = 0.6745 * (data - median) / mad
        outliers = np.abs(modified_z_scores) > threshold
        return outliers
    
    def detect_outliers_isolation_forest(self, data, contamination=0.1):
        """Detect outliers using Isolation Forest (ML-based)."""
        if len(data) < 10:
            return np.zeros(len(data), dtype=bool)
        
        try:
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers = iso_forest.fit_predict(data.reshape(-1, 1)) == -1
            return outliers
        except:
            # Fallback to modified z-score if isolation forest fails
            return self.detect_outliers_modified_zscore(data)
    
    def smooth_price_spikes(self, df, price_cols=['open', 'high', 'low', 'close'], spike_threshold=0.05, window=3):
        """Smooth irregular price spikes using rolling median."""
        df_smooth = df.copy()
        spike_counts = {}
        
        for col in price_cols:
            if col not in df.columns:
                continue
                
            # Calculate percentage change
            pct_change = df[col].pct_change().abs()
            
            # Detect spikes
            spike_mask = pct_change > spike_threshold
            spike_count = spike_mask.sum()
            spike_counts[col] = spike_count
            
            if spike_count > 0:
                logging.info(f"Smoothing {spike_count} price spikes in {col} (>{spike_threshold*100:.1f}%)")
                
                # Calculate rolling median for smoothing
                rolling_median = df[col].rolling(window=window, center=True).median()
                
                # Replace spikes with rolling median
                df_smooth.loc[spike_mask, col] = rolling_median[spike_mask]
                
                # Fill any NaN values created by rolling median at edges
                df_smooth[col] = df_smooth[col].ffill().bfill()
        
        self.outlier_stats['price_spikes'] = spike_counts
        return df_smooth
    
    def handle_volume_outliers(self, df, volume_col='volume'):
        """Handle volume outliers with log transformation and capping."""
        if volume_col not in df.columns:
            # Create dummy volume if not present
            df[volume_col] = 1.0
            return df
        
        df_clean = df.copy()
        
        # Replace zero volume with small positive value
        df_clean[volume_col] = np.maximum(df_clean[volume_col], 1e-6)
        
        # Log transform to handle heavy-tailed distribution
        df_clean[volume_col] = np.log1p(df_clean[volume_col])
        
        # Cap extreme volume values at 99th percentile
        volume_99 = df_clean[volume_col].quantile(0.99)
        outlier_count = (df_clean[volume_col] > volume_99).sum()
        df_clean[volume_col] = np.minimum(df_clean[volume_col], volume_99)
        
        self.outlier_stats['volume_outliers'] = outlier_count
        logging.info(f"Capped {outlier_count} volume outliers at 99th percentile")
        
        return df_clean
    
    def fix_ohlc_relationships(self, df):
        """Fix impossible OHLC relationships."""
        df_fixed = df.copy()
        price_cols = ['open', 'high', 'low', 'close']
        
        # Check if all required columns exist
        missing_cols = [col for col in price_cols if col not in df.columns]
        if missing_cols:
            logging.warning(f"Missing OHLC columns: {missing_cols}")
            return df_fixed
        
        # Fix high/low relationships
        df_fixed['high'] = np.maximum(df_fixed['high'], df_fixed[['open', 'close']].max(axis=1))
        df_fixed['low'] = np.minimum(df_fixed['low'], df_fixed[['open', 'close']].min(axis=1))
        
        # Count fixes
        high_fixes = (df['high'] != df_fixed['high']).sum()
        low_fixes = (df['low'] != df_fixed['low']).sum()
        
        self.outlier_stats['ohlc_fixes'] = {'high_fixes': high_fixes, 'low_fixes': low_fixes}
        
        if high_fixes > 0 or low_fixes > 0:
            logging.info(f"Fixed OHLC relationships: {high_fixes} high fixes, {low_fixes} low fixes")
        
        return df_fixed
    
    def clean_ohlcv_data(self, df):
        """Comprehensive OHLCV data cleaning."""
        logging.info("Starting OHLCV data cleaning...")
        
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # 1. Remove rows with zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        existing_price_cols = [col for col in price_cols if col in df_clean.columns]
        
        for col in existing_price_cols:
            before_count = len(df_clean)
            df_clean = df_clean[df_clean[col] > 0]
            removed = before_count - len(df_clean)
            if removed > 0:
                logging.info(f"Removed {removed} rows with non-positive {col} values")
        
        # 2. Fix OHLC relationships
        df_clean = self.fix_ohlc_relationships(df_clean)
        
        # 3. Smooth price spikes
        df_clean = self.smooth_price_spikes(df_clean)
        
        # 4. Handle volume outliers
        df_clean = self.handle_volume_outliers(df_clean)
        
        final_rows = len(df_clean)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            logging.info(f"OHLCV cleaning completed: {removed_rows} rows removed ({initial_rows} → {final_rows})")
        else:
            logging.info(f"OHLCV cleaning completed: No rows removed")
        
        return df_clean
    
    def detect_and_handle_outliers(self, df, features):
        """Detect and handle outliers in technical indicators."""
        df_clean = df.copy()
        outlier_counts = {}
        
        for feature in features:
            if feature not in df.columns:
                continue
                
            data = df[feature].dropna()
            if len(data) == 0:
                continue
            
            # Choose outlier detection method
            if self.outlier_method == 'iqr':
                outliers_mask = self.detect_outliers_iqr(data, self.outlier_threshold)
            elif self.outlier_method == 'zscore':
                outliers_mask = self.detect_outliers_zscore(data, self.outlier_threshold)
            elif self.outlier_method == 'modified_zscore':
                outliers_mask = self.detect_outliers_modified_zscore(data, self.outlier_threshold)
            elif self.outlier_method == 'isolation_forest':
                outliers_mask = self.detect_outliers_isolation_forest(data, self.outlier_threshold)
            else:
                logging.warning(f"Unknown outlier method: {self.outlier_method}")
                continue
            
            outlier_count = outliers_mask.sum()
            outlier_counts[feature] = outlier_count
            
            if outlier_count > 0:
                logging.info(f"Found {outlier_count} outliers in {feature} ({outlier_count/len(data)*100:.1f}%)")
                
                # Replace outliers with rolling median
                rolling_median = df[feature].rolling(window=5, center=True).median()
                outlier_indices = data.index[outliers_mask]
                df_clean.loc[outlier_indices, feature] = rolling_median[outlier_indices]
                
                # Handle any remaining NaN values
                df_clean[feature] = df_clean[feature].ffill().bfill()
        
        self.outlier_stats['technical_indicators'] = outlier_counts
        return df_clean, outlier_counts
    
    def enhance_technical_indicators(self, df):
        """Enhance technical indicators with robust calculations."""
        df_enhanced = df.copy()
        
        # RSI capping (should be between 0 and 100)
        if 'RSI_14' in df.columns:
            df_enhanced['RSI_14'] = np.clip(df_enhanced['RSI_14'], 0, 100)
            
        # Stochastic capping
        if 'Stoch_K' in df.columns:
            df_enhanced['Stoch_K'] = np.clip(df_enhanced['Stoch_K'], 0, 100)
        if 'Stoch_D' in df.columns:
            df_enhanced['Stoch_D'] = np.clip(df_enhanced['Stoch_D'], 0, 100)
        
        # Handle infinite values in any technical indicator
        numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df_enhanced.columns:
                # Replace inf with NaN, then forward fill
                df_enhanced[col] = df_enhanced[col].replace([np.inf, -np.inf], np.nan)
        
        return df_enhanced
    
    def apply_robust_scaling(self, df, features):
        """Apply robust scaling that's less sensitive to outliers."""
        df_scaled = df.copy()
        
        # Ensure all features exist
        existing_features = [f for f in features if f in df.columns]
        
        if not existing_features:
            logging.warning("No features found for scaling")
            return df_scaled
        
        # Debug: Check input data quality before scaling
        logging.info("Debugging input data before scaling:")
        for feature in existing_features:
            data = df[feature]
            nan_count = data.isna().sum()
            inf_count = np.isinf(data).sum()
            finite_count = np.isfinite(data).sum()
            total_count = len(data)
            
            if nan_count > 0 or inf_count > 0:
                logging.warning(f"  {feature}: {nan_count} NaN, {inf_count} Inf, {finite_count}/{total_count} finite")
                
                # Clean problematic values before scaling
                if nan_count > 0:
                    logging.info(f"  Cleaning {nan_count} NaN values in {feature}")
                    df_scaled[feature] = df_scaled[feature].ffill().bfill()
                    
                if inf_count > 0:
                    logging.info(f"  Cleaning {inf_count} Inf values in {feature}")
                    df_scaled[feature] = df_scaled[feature].replace([np.inf, -np.inf], np.nan).ffill().bfill()
                    
                # Final check
                remaining_invalid = df_scaled[feature].isna().sum() + np.isinf(df_scaled[feature]).sum()
                if remaining_invalid > 0:
                    logging.error(f"  Still have {remaining_invalid} invalid values in {feature} after cleaning")
                    # Use median as last resort
                    median_val = df_scaled[feature].median()
                    if np.isfinite(median_val):
                        df_scaled[feature] = df_scaled[feature].fillna(median_val)
                        df_scaled[feature] = df_scaled[feature].replace([np.inf, -np.inf], median_val)
                        logging.info(f"  Replaced remaining invalid values with median: {median_val}")
                    else:
                        logging.error(f"  Cannot compute valid median for {feature}")
                        # Use zero as absolute last resort
                        df_scaled[feature] = df_scaled[feature].fillna(0).replace([np.inf, -np.inf], 0)
                        logging.warning(f"  Used zero as fallback for {feature}")
            else:
                logging.info(f"  {feature}: Clean (no NaN/Inf values)")
        
        # Verify all data is finite before scaling
        for feature in existing_features:
            if not np.all(np.isfinite(df_scaled[feature])):
                invalid_mask = ~np.isfinite(df_scaled[feature])
                invalid_count = invalid_mask.sum()
                logging.error(f"Feature {feature} still has {invalid_count} invalid values before scaling")
                # Force clean
                df_scaled.loc[invalid_mask, feature] = 0
                logging.warning(f"Force-cleaned {invalid_count} invalid values in {feature} with zeros")
        
        try:
            # Fit scaler on existing features
            scaled_values = self.scaler.fit_transform(df_scaled[existing_features])
            
            # Check for invalid values after scaling
            if np.any(np.isnan(scaled_values)) or np.any(np.isinf(scaled_values)):
                logging.error("Scaling produced NaN or Inf values!")
                nan_count = np.isnan(scaled_values).sum()
                inf_count = np.isinf(scaled_values).sum()
                logging.error(f"Post-scaling: {nan_count} NaN, {inf_count} Inf values")
                
                # Clean scaled values
                scaled_values = np.nan_to_num(scaled_values, nan=0.0, posinf=5.0, neginf=-5.0)
                logging.warning("Applied nan_to_num to clean scaled values")
            
            # Conservative clipping to prevent extreme values
            scaled_values = np.clip(scaled_values, -5, 5)
            
            # Final validation
            if np.any(np.isnan(scaled_values)) or np.any(np.isinf(scaled_values)):
                logging.error("CRITICAL: Still have invalid values after all cleaning attempts")
                # Last resort: replace with zeros
                scaled_values = np.nan_to_num(scaled_values, nan=0.0, posinf=0.0, neginf=0.0)
                logging.error("Applied emergency cleaning - replaced all invalid values with zeros")
            
            # Update dataframe
            df_scaled[existing_features] = scaled_values
            
            logging.info(f"Applied {'robust' if self.use_robust_scaler else 'standard'} scaling to {len(existing_features)} features")
            
            # Final data quality check
            final_check_passed = True
            for feature in existing_features:
                if not np.all(np.isfinite(df_scaled[feature])):
                    logging.error(f"FINAL CHECK FAILED: {feature} still has invalid values")
                    final_check_passed = False
            
            if final_check_passed:
                logging.info("Final data quality check: PASSED - All features have finite values")
            else:
                logging.error("Final data quality check: FAILED - Some features still have invalid values")
            
        except Exception as e:
            logging.error(f"Scaling failed: {e}")
            logging.error("Applying emergency fallback scaling")
            
            # Emergency fallback: simple min-max scaling with safety checks
            for feature in existing_features:
                data = df_scaled[feature]
                
                # Ensure finite values
                data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Simple min-max scaling
                data_min = data.min()
                data_max = data.max()
                
                if data_max != data_min and np.isfinite(data_min) and np.isfinite(data_max):
                    df_scaled[feature] = (data - data_min) / (data_max - data_min)
                else:
                    df_scaled[feature] = 0  # Constant data becomes zero
                
                # Clip to safe range
                df_scaled[feature] = np.clip(df_scaled[feature], -1, 1)
            
            logging.warning("Emergency fallback scaling completed")
        
        return df_scaled
    
    def process_data(self, df, features):
        """Complete enhanced data processing pipeline."""
        logging.info("="*60)
        logging.info("STARTING ENHANCED DATA PREPROCESSING")
        logging.info("="*60)
        
        # Reset outlier stats
        self.outlier_stats = {}
        
        # 1. Clean OHLCV data
        df_clean = self.clean_ohlcv_data(df)
        logging.info(f"Step 1 - OHLCV cleaning: {len(df)} → {len(df_clean)} rows")
        
        # 2. Enhance technical indicators
        df_enhanced = self.enhance_technical_indicators(df_clean)
        logging.info("Step 2 - Technical indicator enhancement completed")
        
        # 3. Detect and handle outliers in technical indicators
        df_no_outliers, outlier_counts = self.detect_and_handle_outliers(df_enhanced, features)
        total_outliers = sum(outlier_counts.values())
        logging.info(f"Step 3 - Outlier handling: {total_outliers} outliers processed across all features")
        
        # 4. Apply robust scaling
        df_final = self.apply_robust_scaling(df_no_outliers, features)
        logging.info("Step 4 - Robust scaling completed")
        
        # Log summary statistics
        self._log_processing_summary()
        
        logging.info("="*60)
        logging.info("ENHANCED PREPROCESSING COMPLETED")
        logging.info("="*60)
        
        return df_final, self.outlier_stats
    
    def _log_processing_summary(self):
        """Log summary of processing statistics."""
        logging.info("\nProcessing Summary:")
        logging.info("-" * 40)
        
        if 'price_spikes' in self.outlier_stats:
            spike_total = sum(self.outlier_stats['price_spikes'].values())
            logging.info(f"Price spikes smoothed: {spike_total}")
        
        if 'volume_outliers' in self.outlier_stats:
            logging.info(f"Volume outliers capped: {self.outlier_stats['volume_outliers']}")
        
        if 'ohlc_fixes' in self.outlier_stats:
            fixes = self.outlier_stats['ohlc_fixes']
            total_fixes = fixes['high_fixes'] + fixes['low_fixes']
            logging.info(f"OHLC relationship fixes: {total_fixes}")
        
        if 'technical_indicators' in self.outlier_stats:
            tech_outliers = sum(self.outlier_stats['technical_indicators'].values())
            logging.info(f"Technical indicator outliers: {tech_outliers}")
        
        logging.info(f"Scaling method: {'RobustScaler' if self.use_robust_scaler else 'MinMaxScaler'}")
        logging.info("-" * 40)
    
    def get_stats(self):
        """Get processing statistics."""
        return self.outlier_stats.copy()
    
    def save_scaler(self, path):
        """Save the fitted scaler."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scaler(self, path):
        """Load a fitted scaler."""
        import pickle
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f) 