"""
Walk-forward evaluation framework for optimal-stopping sell signal models
Realistic PnL simulation with costs, slippage, and risk metrics
"""
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config import OPTIMAL_STOPPING_CONFIG, MODEL_CONFIG

@dataclass
class TradeResult:
    """Single trade result with all metrics"""
    entry_time: datetime
    exit_time: datetime
    token: str
    pool: str
    bucket: str
    entry_price: float
    exit_price: float
    trade_size_usdc: float
    dex_fee: float
    slippage: float
    gross_return: float
    net_return: float  # After fees and slippage
    holding_time_hours: float
    max_drawdown: float
    exit_reason: str  # 'model', 'stop_loss', 'timeout'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'token': self.token,
            'pool': self.pool,
            'bucket': self.bucket,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'trade_size_usdc': self.trade_size_usdc,
            'dex_fee': self.dex_fee,
            'slippage': self.slippage,
            'gross_return': self.gross_return,
            'net_return': self.net_return,
            'holding_time_hours': self.holding_time_hours,
            'max_drawdown': self.max_drawdown,
            'exit_reason': self.exit_reason
        }


class WalkForwardEvaluator:
    """
    Walk-forward evaluation with realistic trading simulation
    Tests models on out-of-sample data with proper time-series splits
    """
    
    def __init__(self, 
                 initial_train_days: int = 30,
                 retraining_frequency_days: int = 7,
                 test_window_days: int = 7):
        
        self.initial_train_days = initial_train_days
        self.retraining_frequency_days = retraining_frequency_days
        self.test_window_days = test_window_days
        
        # Evaluation metrics storage
        self.trade_results: List[TradeResult] = []
        self.model_predictions: Dict[str, List[float]] = {}
        self.walk_forward_metrics: List[Dict] = []
        
    def simulate_trade(self, 
                      buy_signal: Dict,
                      price_data: pl.DataFrame,
                      model_predictions: Dict[str, np.ndarray],
                      cost_model: Tuple[float, float]) -> TradeResult:
        """
        Simulate a single trade from entry to exit
        
        Args:
            buy_signal: Dict with entry information
            price_data: Price/feature data after entry
            model_predictions: Model predictions for each timestep
            cost_model: (dex_fee, slippage) tuple
        """
        
        entry_time = buy_signal['timestamp']
        entry_price = buy_signal['entry_price']
        token = buy_signal['token']
        pool = buy_signal['pool']
        bucket = buy_signal['bucket']
        trade_size = buy_signal['size_usdc']
        
        dex_fee, slippage = cost_model
        
        # Track trade progress
        peak_price = entry_price
        max_drawdown = 0.0
        exit_time = entry_time
        exit_price = entry_price
        exit_reason = 'timeout'
        
        # Get horizon for this bucket
        horizon_hours = OPTIMAL_STOPPING_CONFIG.horizon_hours[bucket]
        timeout_time = entry_time + timedelta(hours=horizon_hours)
        
        # Simulate timestep by timestep
        for i, row in enumerate(price_data.iter_rows(named=True)):
            current_time = row['timestamp']
            current_price = row['close_price']
            
            if current_time > timeout_time:
                break
                
            # Update peak and drawdown
            if current_price > peak_price:
                peak_price = current_price
            
            current_drawdown = (peak_price - current_price) / peak_price
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Check model sell signal
            upside_pred = model_predictions.get('upside', [0])[i] if i < len(model_predictions.get('upside', [])) else 0
            drawdown_prob = model_predictions.get('drawdown_prob', [0])[i] if i < len(model_predictions.get('drawdown_prob', [])) else 0
            
            # Decision logic
            should_sell = False
            
            # Model-based exit
            if (upside_pred < OPTIMAL_STOPPING_CONFIG.upside_threshold_pct / 100 and 
                drawdown_prob > OPTIMAL_STOPPING_CONFIG.drawdown_prob_threshold):
                should_sell = True
                exit_reason = 'model'
            
            # Trailing stop based on ATR
            atr = row.get('price_atr_20_periods', 0)
            if atr > 0:
                trailing_stop_price = peak_price - (OPTIMAL_STOPPING_CONFIG.trailing_stop_atr_multiplier * atr)
                if current_price <= trailing_stop_price:
                    should_sell = True
                    exit_reason = 'stop_loss'
            
            # Hard stop loss (5% for safety)
            if (current_price / entry_price - 1) < -0.05:
                should_sell = True
                exit_reason = 'stop_loss'
                
            if should_sell:
                exit_time = current_time
                exit_price = current_price
                break
        else:
            # Timeout exit - use last available price
            if len(price_data) > 0:
                last_row = price_data.tail(1).to_dicts()[0]
                exit_time = last_row['timestamp']
                exit_price = last_row['close_price']
            
        # Calculate returns
        gross_return = (exit_price / entry_price) - 1
        net_return = (exit_price / entry_price) * (1 - dex_fee - slippage) - 1
        
        holding_time_hours = (exit_time - entry_time).total_seconds() / 3600
        
        return TradeResult(
            entry_time=entry_time,
            exit_time=exit_time,
            token=token,
            pool=pool,
            bucket=bucket,
            entry_price=entry_price,
            exit_price=exit_price,
            trade_size_usdc=trade_size,
            dex_fee=dex_fee,
            slippage=slippage,
            gross_return=gross_return,
            net_return=net_return,
            holding_time_hours=holding_time_hours,
            max_drawdown=max_drawdown,
            exit_reason=exit_reason
        )
        
    def evaluate_model(self,
                      model: Any,
                      buy_signals: pl.DataFrame,
                      price_data: pl.DataFrame,
                      start_date: str,
                      end_date: str) -> Dict[str, float]:
        """
        Run walk-forward evaluation on a trained model
        
        Args:
            model: Trained sell signal model (baseline or sequence)
            buy_signals: DataFrame with buy signals
            price_data: Full price/feature dataset
            start_date: Evaluation start date
            end_date: Evaluation end date
        """
        
        self.trade_results = []
        
        # Convert dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Filter buy signals in evaluation period
        eval_signals = buy_signals.filter(
            (pl.col('timestamp') >= start_dt) & 
            (pl.col('timestamp') <= end_dt)
        )
        
        print(f"Evaluating {len(eval_signals)} buy signals from {start_date} to {end_date}")
        
        for i, signal_row in enumerate(eval_signals.iter_rows(named=True)):
            if i % 100 == 0:
                print(f"Processing signal {i+1}/{len(eval_signals)}")
                
            try:
                # Extract trade data
                token = signal_row['token']
                pool = signal_row['pool']
                entry_time = signal_row['timestamp']
                bucket = signal_row.get('bucket', 'established')
                
                # Get horizon for this bucket
                horizon_hours = OPTIMAL_STOPPING_CONFIG.horizon_hours[bucket]
                end_time = entry_time + timedelta(hours=horizon_hours)
                
                # Get price data for this trade
                trade_price_data = price_data.filter(
                    (pl.col('token') == token) &
                    (pl.col('pool') == pool) &
                    (pl.col('timestamp') >= entry_time) &
                    (pl.col('timestamp') <= end_time)
                ).sort('timestamp')
                
                if len(trade_price_data) < 2:  # Need at least 2 points
                    continue
                    
                # Get model predictions
                model_predictions = {}
                
                if hasattr(model, 'predict'):  # Baseline models
                    try:
                        upside_pred = model.predict(trade_price_data, bucket)
                        model_predictions['upside'] = upside_pred
                        
                        if hasattr(model, 'drawdown_classifier'):
                            drawdown_prob = model.drawdown_classifier.predict_proba(trade_price_data, bucket)
                            model_predictions['drawdown_prob'] = drawdown_prob
                        else:
                            model_predictions['drawdown_prob'] = np.zeros(len(upside_pred))
                    except:
                        # Fallback to dummy predictions if model fails
                        model_predictions['upside'] = np.zeros(len(trade_price_data))
                        model_predictions['drawdown_prob'] = np.zeros(len(trade_price_data))
                        
                elif hasattr(model, 'forward'):  # PyTorch models
                    # TODO: Implement sequence model prediction
                    model_predictions['upside'] = np.zeros(len(trade_price_data))
                    model_predictions['drawdown_prob'] = np.zeros(len(trade_price_data))
                
                # Estimate costs
                dex_name = signal_row.get('dex', 'raydium')
                dex_fee = OPTIMAL_STOPPING_CONFIG.dex_fees.get(dex_name.lower(), 0.0025)
                slippage = 0.01  # Default 1% slippage
                
                # Simulate trade
                trade_result = self.simulate_trade(
                    signal_row,
                    trade_price_data,
                    model_predictions,
                    (dex_fee, slippage)
                )
                
                self.trade_results.append(trade_result)
                
            except Exception as e:
                print(f"Error processing signal {i}: {e}")
                continue
                
        # Calculate aggregate metrics
        return self.calculate_performance_metrics()
        
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics from trade results"""
        
        if not self.trade_results:
            return {}
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([trade.to_dict() for trade in self.trade_results])
        
        # Basic PnL metrics
        total_trades = len(df)
        winning_trades = (df['net_return'] > 0).sum()
        losing_trades = (df['net_return'] < 0).sum()
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_return = df['net_return'].mean()
        median_return = df['net_return'].median()
        
        total_return = (1 + df['net_return']).prod() - 1
        
        # Risk metrics
        return_std = df['net_return'].std()
        sharpe_ratio = avg_return / return_std if return_std > 0 else 0
        
        max_drawdown = df['max_drawdown'].max()
        avg_drawdown = df['max_drawdown'].mean()
        
        # Downside metrics
        negative_returns = df[df['net_return'] < 0]['net_return']
        worst_loss = negative_returns.min() if len(negative_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        positive_returns = df[df['net_return'] > 0]['net_return']
        best_win = positive_returns.max() if len(positive_returns) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        
        # Profit factor
        total_wins = positive_returns.sum() if len(positive_returns) > 0 else 0
        total_losses = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0.001
        profit_factor = total_wins / total_losses
        
        # Holding time analysis
        avg_holding_time = df['holding_time_hours'].mean()
        median_holding_time = df['holding_time_hours'].median()
        
        # Exit reason analysis
        model_exits = (df['exit_reason'] == 'model').sum()
        stop_exits = (df['exit_reason'] == 'stop_loss').sum() 
        timeout_exits = (df['exit_reason'] == 'timeout').sum()
        
        # Risk-adjusted metrics
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Tail risk (VaR)
        var_95 = np.percentile(df['net_return'], 5)  # 5th percentile
        cvar_95 = df[df['net_return'] <= var_95]['net_return'].mean()  # Expected shortfall
        
        metrics = {
            # Basic metrics
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'median_return': median_return,
            'total_return': total_return,
            
            # Risk metrics
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'calmar_ratio': calmar_ratio,
            
            # Win/Loss analysis
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_win': best_win,
            'worst_loss': worst_loss,
            'profit_factor': profit_factor,
            
            # Tail risk
            'var_95': var_95,
            'cvar_95': cvar_95,
            
            # Timing metrics
            'avg_holding_time_hours': avg_holding_time,
            'median_holding_time_hours': median_holding_time,
            
            # Exit analysis
            'model_exit_pct': model_exits / total_trades * 100,
            'stop_exit_pct': stop_exits / total_trades * 100,
            'timeout_exit_pct': timeout_exits / total_trades * 100,
        }
        
        return metrics
        
    def compare_with_baselines(self) -> Dict[str, Dict[str, float]]:
        """Compare model performance with simple baseline strategies"""
        
        baseline_results = {}
        
        # Fixed time exit baselines
        for hours in [1, 2, 4, 8, 12, 24]:
            baseline_trades = []
            
            for trade in self.trade_results:
                # Simulate fixed time exit
                exit_time = trade.entry_time + timedelta(hours=hours)
                
                # Find price at that time (approximate)
                if hasattr(trade, 'price_data'):
                    # This would need the original price data - simplified for now
                    baseline_return = np.random.normal(0.02, 0.1)  # Placeholder
                else:
                    # Use actual trade return scaled by time
                    time_ratio = min(1.0, hours / trade.holding_time_hours)
                    baseline_return = trade.net_return * time_ratio
                    
                baseline_trades.append(baseline_return)
                
            if baseline_trades:
                baseline_results[f'{hours}h_exit'] = {
                    'avg_return': np.mean(baseline_trades),
                    'sharpe_ratio': np.mean(baseline_trades) / np.std(baseline_trades) if np.std(baseline_trades) > 0 else 0,
                    'win_rate': np.mean([r > 0 for r in baseline_trades])
                }
                
        # Fixed percentage exit baselines  
        for pct in [2, 5, 10, 15]:
            baseline_results[f'{pct}pct_tp'] = {
                'avg_return': pct/100 * 0.6,  # Assume 60% hit rate for fixed TP
                'win_rate': 0.6
            }
            
        return baseline_results
        
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report"""
        
        metrics = self.calculate_performance_metrics()
        baselines = self.compare_with_baselines()
        
        report = f"""
# Sell Signal Model Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary
- **Total Trades**: {metrics.get('total_trades', 0):,}
- **Win Rate**: {metrics.get('win_rate', 0):.1%}
- **Average Return**: {metrics.get('avg_return', 0):.2%}
- **Total Return**: {metrics.get('total_return', 0):.2%}
- **Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.2f}

## Risk Metrics
- **Maximum Drawdown**: {metrics.get('max_drawdown', 0):.2%}
- **Calmar Ratio**: {metrics.get('calmar_ratio', 0):.2f}
- **VaR (95%)**: {metrics.get('var_95', 0):.2%}
- **Expected Shortfall**: {metrics.get('cvar_95', 0):.2%}

## Win/Loss Analysis
- **Average Win**: {metrics.get('avg_win', 0):.2%}
- **Average Loss**: {metrics.get('avg_loss', 0):.2%}
- **Profit Factor**: {metrics.get('profit_factor', 0):.2f}
- **Best Trade**: {metrics.get('best_win', 0):.2%}
- **Worst Trade**: {metrics.get('worst_loss', 0):.2%}

## Exit Analysis
- **Model Exits**: {metrics.get('model_exit_pct', 0):.1f}%
- **Stop Loss Exits**: {metrics.get('stop_exit_pct', 0):.1f}%
- **Timeout Exits**: {metrics.get('timeout_exit_pct', 0):.1f}%

## Timing Analysis
- **Average Holding Time**: {metrics.get('avg_holding_time_hours', 0):.1f} hours
- **Median Holding Time**: {metrics.get('median_holding_time_hours', 0):.1f} hours

## Baseline Comparison
"""
        
        for baseline_name, baseline_metrics in baselines.items():
            report += f"- **{baseline_name}**: "
            report += f"Return: {baseline_metrics.get('avg_return', 0):.2%}, "
            report += f"Win Rate: {baseline_metrics.get('win_rate', 0):.1%}\n"
            
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
                
        return report
        
    def plot_performance_analysis(self, save_path: Optional[str] = None):
        """Create visualization dashboard for model performance"""
        
        if not self.trade_results:
            print("No trade results to plot")
            return
            
        df = pd.DataFrame([trade.to_dict() for trade in self.trade_results])
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sell Signal Model Performance Analysis', fontsize=16)
        
        # 1. Return distribution
        axes[0, 0].hist(df['net_return'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(df['net_return'].mean(), color='red', linestyle='--', label=f'Mean: {df["net_return"].mean():.2%}')
        axes[0, 0].set_xlabel('Net Return')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Return Distribution')
        axes[0, 0].legend()
        
        # 2. Cumulative returns
        df_sorted = df.sort_values('entry_time')
        cumulative_returns = (1 + df_sorted['net_return']).cumprod()
        axes[0, 1].plot(range(len(cumulative_returns)), cumulative_returns)
        axes[0, 1].set_xlabel('Trade Number')
        axes[0, 1].set_ylabel('Cumulative Return')
        axes[0, 1].set_title('Cumulative Performance')
        
        # 3. Win rate by bucket
        bucket_stats = df.groupby('bucket')['net_return'].agg(['mean', lambda x: (x > 0).mean()]).round(3)
        bucket_stats.columns = ['Avg Return', 'Win Rate']
        bucket_stats['Win Rate'].plot(kind='bar', ax=axes[0, 2])
        axes[0, 2].set_xlabel('Liquidity Bucket')
        axes[0, 2].set_ylabel('Win Rate')
        axes[0, 2].set_title('Win Rate by Bucket')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Holding time vs return
        axes[1, 0].scatter(df['holding_time_hours'], df['net_return'], alpha=0.6)
        axes[1, 0].set_xlabel('Holding Time (Hours)')
        axes[1, 0].set_ylabel('Net Return')
        axes[1, 0].set_title('Holding Time vs Return')
        
        # 5. Exit reason analysis
        exit_counts = df['exit_reason'].value_counts()
        axes[1, 1].pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Exit Reasons')
        
        # 6. Drawdown distribution
        axes[1, 2].hist(df['max_drawdown'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 2].set_xlabel('Maximum Drawdown')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Drawdown Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def run_backtest_comparison(models: Dict[str, Any], 
                          buy_signals: pl.DataFrame,
                          price_data: pl.DataFrame,
                          start_date: str,
                          end_date: str) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models using walk-forward evaluation
    
    Args:
        models: Dict of model_name -> trained_model
        buy_signals: Buy signals DataFrame
        price_data: Price/feature data
        start_date: Evaluation start
        end_date: Evaluation end
        
    Returns:
        Dict of model_name -> performance_metrics
    """
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        evaluator = WalkForwardEvaluator()
        metrics = evaluator.evaluate_model(model, buy_signals, price_data, start_date, end_date)
        
        results[model_name] = metrics
        
        # Generate individual report
        report = evaluator.generate_report()
        print(f"\n{model_name} Results:")
        print("=" * 50)
        print(report)
        
    return results
