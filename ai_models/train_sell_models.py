"""
End-to-end training pipeline for optimal-stopping sell signal prediction
Complete workflow from data extraction to model training and evaluation
"""
import os
import sys
import argparse
import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Any
import torch
import pytorch_lightning as pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_extraction import DEXDataExtractor
from models.baseline_models import SellDecisionEngine
from models.sequence_models import TCNSellPredictor, TransformerSellPredictor, create_sequence_data_loaders
from evaluation import WalkForwardEvaluator, run_backtest_comparison

class SellSignalTrainingPipeline:
    """Complete training pipeline for sell signal prediction models"""
    
    def __init__(self, 
                 start_date: str = "2024-01-01",
                 end_date: str = "2024-12-01",
                 validation_split_days: int = 30):
        
        self.start_date = start_date
        self.end_date = end_date  
        self.validation_split_days = validation_split_days
        
        # Initialize components
        self.data_extractor = DEXDataExtractor()
        self.trained_models = {}
        
        print(f"Initialized training pipeline for period: {start_date} to {end_date}")
        
    def extract_buy_signals(self) -> pl.DataFrame:
        """
        Extract or define your buy signals here
        Replace this with your actual buy signal logic
        """
        
        print("Extracting buy signals...")
        
        # Example buy signal query - replace with your actual logic
        buy_signals_query = f"""
        WITH volume_breakouts AS (
            SELECT 
                timestamp,
                token_a as token,
                pool_address as pool,
                dex,
                token_a_usdc_price as entry_price,
                abs(amount_b * token_b_usdc_price) as size_usdc,
                -- Simple volume breakout condition (replace with your logic)
                CASE 
                    WHEN abs(amount_b * token_b_usdc_price) > 10000 
                    AND token_a_usdc_price > 0
                    AND amount_a < 0  -- Buy transaction
                    THEN 1 
                    ELSE 0 
                END as is_buy_signal
            FROM solana_swaps_raw
            WHERE timestamp >= '{self.start_date}'
            AND timestamp <= '{self.end_date}'
            AND token_b IN allowed_quote_tokens()
            AND abs(amount_b * token_b_usdc_price) >= 1000  -- Min $1k trade size
        )
        SELECT 
            timestamp,
            token,
            pool,
            dex,
            entry_price,
            size_usdc
        FROM volume_breakouts 
        WHERE is_buy_signal = 1
        ORDER BY timestamp
        LIMIT 10000  -- Limit for testing - remove in production
        """
        
        buy_signals = self.data_extractor.client.query_df(buy_signals_query)
        buy_signals_pl = pl.from_pandas(buy_signals)
        
        print(f"Extracted {len(buy_signals_pl)} buy signals")
        return buy_signals_pl
        
    def prepare_training_data(self, buy_signals: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """Build complete training dataset with features and labels"""
        
        print("Building training dataset...")
        
        # Split into train/validation by time
        split_date = pd.to_datetime(self.end_date) - timedelta(days=self.validation_split_days)
        
        train_signals = buy_signals.filter(pl.col('timestamp') < split_date)
        val_signals = buy_signals.filter(pl.col('timestamp') >= split_date)
        
        print(f"Train signals: {len(train_signals)}, Validation signals: {len(val_signals)}")
        
        # Build training dataset (this is the heavy computation)
        print("Building training features and labels (this may take a while)...")
        train_data = self.data_extractor.build_training_dataset(
            train_signals, 
            self.start_date, 
            str(split_date.date())
        )
        
        val_data = self.data_extractor.build_training_dataset(
            val_signals,
            str(split_date.date()),
            self.end_date
        ) if len(val_signals) > 0 else None
        
        print(f"Training dataset: {len(train_data)} samples")
        if val_data is not None:
            print(f"Validation dataset: {len(val_data)} samples")
            
        return {
            'train': train_data,
            'validation': val_data,
            'train_signals': train_signals,
            'val_signals': val_signals
        }
        
    def train_baseline_models(self, train_data: pl.DataFrame, 
                            val_data: pl.DataFrame = None) -> SellDecisionEngine:
        """Train LightGBM/XGBoost baseline models"""
        
        print("\n" + "="*50)
        print("Training Baseline Models (LightGBM + XGBoost)")
        print("="*50)
        
        # Initialize and train
        sell_engine = SellDecisionEngine()
        results = sell_engine.train(train_data, val_data)
        
        # Save models
        model_path = sell_engine.save_models()
        print(f"Baseline models saved to: {model_path}")
        
        self.trained_models['baseline'] = sell_engine
        
        return sell_engine
        
    def train_sequence_models(self, train_data: pl.DataFrame, 
                            val_data: pl.DataFrame = None) -> Dict[str, Any]:
        """Train TCN and Transformer sequence models"""
        
        print("\n" + "="*50)
        print("Training Sequence Models (TCN + Transformer)")
        print("="*50)
        
        # Prepare sequence datasets
        train_loader, val_loader = create_sequence_data_loaders(
            train_data, 
            val_data, 
            sequence_length=MODEL_CONFIG.sequence_length,
            batch_size=MODEL_CONFIG.batch_size
        )
        
        # Get input dimension from first batch
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch[0].shape[-1]
        
        sequence_models = {}
        
        # Train TCN model
        print("\nTraining TCN model...")
        tcn_model = TCNSellPredictor(
            input_dim=input_dim,
            learning_rate=MODEL_CONFIG.learning_rate
        )
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=os.path.join(MODEL_DIR, 'tcn_checkpoints'),
                monitor='val_loss' if val_loader else 'train_loss',
                mode='min',
                save_top_k=3
            ),
            EarlyStopping(
                monitor='val_loss' if val_loader else 'train_loss',
                patience=10,
                mode='min'
            )
        ]
        
        # Setup trainer for your RTX Pro 4500 setup
        trainer = pytorch_lightning.Trainer(
            max_epochs=MODEL_CONFIG.max_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=MODEL_CONFIG.num_gpus if torch.cuda.is_available() else 1,
            precision=MODEL_CONFIG.precision,
            callbacks=callbacks,
            gradient_clip_val=MODEL_CONFIG.gradient_clip_val,
            log_every_n_steps=50,
            strategy='ddp' if MODEL_CONFIG.num_gpus > 1 else 'auto'
        )
        
        # Train TCN
        trainer.fit(tcn_model, train_loader, val_loader)
        sequence_models['tcn'] = tcn_model
        
        # Train Transformer model
        print("\nTraining Transformer model...")
        transformer_model = TransformerSellPredictor(
            input_dim=input_dim,
            d_model=MODEL_CONFIG.transformer_dim,
            nhead=MODEL_CONFIG.transformer_heads,
            num_layers=MODEL_CONFIG.transformer_layers,
            learning_rate=MODEL_CONFIG.learning_rate
        )
        
        # New trainer for transformer (separate checkpoints)
        transformer_callbacks = [
            ModelCheckpoint(
                dirpath=os.path.join(MODEL_DIR, 'transformer_checkpoints'),
                monitor='val_loss' if val_loader else 'train_loss',
                mode='min',
                save_top_k=3
            ),
            EarlyStopping(
                monitor='val_loss' if val_loader else 'train_loss',
                patience=15,  # More patience for transformer
                mode='min'
            )
        ]
        
        transformer_trainer = pytorch_lightning.Trainer(
            max_epochs=MODEL_CONFIG.max_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=MODEL_CONFIG.num_gpus if torch.cuda.is_available() else 1,
            precision=MODEL_CONFIG.precision,
            callbacks=transformer_callbacks,
            gradient_clip_val=MODEL_CONFIG.gradient_clip_val,
            log_every_n_steps=50,
            strategy='ddp' if MODEL_CONFIG.num_gpus > 1 else 'auto'
        )
        
        transformer_trainer.fit(transformer_model, train_loader, val_loader)
        sequence_models['transformer'] = transformer_model
        
        self.trained_models.update(sequence_models)
        
        return sequence_models
        
    def evaluate_all_models(self, datasets: Dict[str, pl.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Run comprehensive evaluation on all trained models"""
        
        print("\n" + "="*50)
        print("Model Evaluation")
        print("="*50)
        
        # Prepare evaluation data
        val_signals = datasets['val_signals']
        
        if len(val_signals) == 0:
            print("No validation data available for evaluation")
            return {}
            
        # Get price data for evaluation period
        split_date = pd.to_datetime(self.end_date) - timedelta(days=self.validation_split_days)
        
        # For evaluation, we need the same price/feature data used in training
        # This is a simplified version - in practice, you'd want to cache this
        print("Extracting price data for evaluation...")
        
        # Extract features for a sample of tokens from validation set
        sample_tokens = val_signals.select('token').unique().limit(10)  # Sample for speed
        
        # This is where you'd extract the full feature dataset for evaluation
        # For now, we'll use the validation training data as a proxy
        price_data = datasets.get('validation', datasets['train'])
        
        # Run evaluations
        evaluation_results = {}
        
        # Evaluate baseline model
        if 'baseline' in self.trained_models:
            print("\nEvaluating baseline model...")
            baseline_evaluator = WalkForwardEvaluator()
            baseline_metrics = baseline_evaluator.evaluate_model(
                self.trained_models['baseline'],
                val_signals,
                price_data,
                str(split_date.date()),
                self.end_date
            )
            evaluation_results['baseline'] = baseline_metrics
            
            # Generate report
            report = baseline_evaluator.generate_report(
                os.path.join(LOGS_DIR, 'baseline_evaluation_report.md')
            )
            
            # Create plots
            baseline_evaluator.plot_performance_analysis(
                os.path.join(LOGS_DIR, 'baseline_performance_plots.png')
            )
            
        # Note: Sequence model evaluation would need additional implementation
        # for proper inference on sequential data
        
        return evaluation_results
        
    def run_complete_pipeline(self, train_baseline: bool = True, 
                            train_sequence: bool = False) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        
        print("Starting Complete Sell Signal Training Pipeline")
        print("=" * 60)
        
        results = {}
        
        try:
            # Step 1: Extract buy signals
            buy_signals = self.extract_buy_signals()
            results['buy_signals_count'] = len(buy_signals)
            
            # Step 2: Prepare training data
            datasets = self.prepare_training_data(buy_signals)
            results['train_samples'] = len(datasets['train'])
            results['val_samples'] = len(datasets['validation']) if datasets['validation'] else 0
            
            # Step 3: Train models
            if train_baseline:
                baseline_model = self.train_baseline_models(
                    datasets['train'], 
                    datasets['validation']
                )
                results['baseline_trained'] = True
                
            if train_sequence and len(datasets['train']) > 1000:  # Need sufficient data
                sequence_models = self.train_sequence_models(
                    datasets['train'],
                    datasets['validation'] 
                )
                results['sequence_trained'] = True
            
            # Step 4: Evaluation
            evaluation_results = self.evaluate_all_models(datasets)
            results['evaluation'] = evaluation_results
            
            # Step 5: Summary
            self.print_final_summary(results)
            
            return results
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
            
    def print_final_summary(self, results: Dict[str, Any]):
        """Print final pipeline summary"""
        
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        
        print(f"📊 Data: {results.get('buy_signals_count', 0)} buy signals processed")
        print(f"🎯 Training: {results.get('train_samples', 0)} samples")
        print(f"✅ Validation: {results.get('val_samples', 0)} samples")
        
        if results.get('baseline_trained'):
            print("🤖 Baseline Models: ✅ Trained (LightGBM + XGBoost)")
        
        if results.get('sequence_trained'):
            print("🧠 Sequence Models: ✅ Trained (TCN + Transformer)")
            
        if 'evaluation' in results:
            print("\n📈 Evaluation Results:")
            for model_name, metrics in results['evaluation'].items():
                if metrics:
                    win_rate = metrics.get('win_rate', 0)
                    avg_return = metrics.get('avg_return', 0)
                    sharpe = metrics.get('sharpe_ratio', 0)
                    print(f"   {model_name}: Win Rate: {win_rate:.1%}, Avg Return: {avg_return:.2%}, Sharpe: {sharpe:.2f}")
                    
        print(f"\n📁 Models saved to: {MODEL_DIR}")
        print(f"📄 Logs saved to: {LOGS_DIR}")
        print("\nPipeline completed successfully! 🎉")


def main():
    """Main entry point for training pipeline"""
    
    parser = argparse.ArgumentParser(description='Train sell signal prediction models')
    parser.add_argument('--start-date', default='2024-01-01', 
                       help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-12-01',
                       help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--validation-days', type=int, default=30,
                       help='Days to reserve for validation')
    parser.add_argument('--baseline-only', action='store_true',
                       help='Train only baseline models (faster)')
    parser.add_argument('--sequence-only', action='store_true', 
                       help='Train only sequence models')
    parser.add_argument('--config-db-host', default='localhost',
                       help='ClickHouse database host')
    
    args = parser.parse_args()
    
    # Update database config if provided
    if args.config_db_host != 'localhost':
        DB_CONFIG.host = args.config_db_host
        
    # Initialize pipeline
    pipeline = SellSignalTrainingPipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        validation_split_days=args.validation_days
    )
    
    # Determine what to train
    train_baseline = not args.sequence_only
    train_sequence = not args.baseline_only
    
    # Run pipeline
    results = pipeline.run_complete_pipeline(
        train_baseline=train_baseline,
        train_sequence=train_sequence
    )
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    print("Sell Signal Model Training Pipeline")
    print("=" * 40)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  GPU not available, using CPU")
        
    # Run with example parameters
    results = main()
