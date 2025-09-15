"""
Configuration file for AI sell signal prediction models
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class DatabaseConfig:
    """ClickHouse database configuration"""
    host: str = "localhost"
    port: int = 8123
    database: str = "default"
    username: str = "default" 
    password: str = ""

@dataclass
class ModelConfig:
    """Model training configuration optimized for RTX Pro 4500 setup"""
    # Hardware optimization
    device: str = "cuda"
    num_gpus: int = 2
    batch_size: int = 512  # Optimized for 32GB GPU memory
    max_workers: int = 16  # Utilize your high core count
    precision: str = "16-mixed"  # Mixed precision for memory efficiency
    
    # Time series parameters
    sequence_length: int = 168  # 7 days of hourly data
    prediction_horizons: List[int] = None  # [1, 4, 12, 24, 48]  # hours
    feature_window: int = 7  # days
    
    # Training parameters
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_val: float = 1.0
    
    # Model architecture
    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 6
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 4, 12, 24, 48]

@dataclass 
class FeatureConfig:
    """Feature engineering configuration"""
    # Technical indicators
    rsi_periods: List[int] = None  # [14, 21]
    sma_periods: List[int] = None  # [20, 50, 200]
    ema_periods: List[int] = None  # [12, 26]
    bollinger_period: int = 20
    
    # Volume indicators
    volume_sma_periods: List[int] = None  # [20, 50]
    vwap_periods: List[int] = None  # [20, 50]
    
    # Market microstructure
    order_book_levels: int = 5
    trade_flow_window: int = 60  # minutes
    
    # Trader behavior
    smart_money_threshold: float = 10000  # USD
    whale_threshold: float = 100000  # USD
    min_trades_for_pattern: int = 10
    
    def __post_init__(self):
        if self.rsi_periods is None:
            self.rsi_periods = [14, 21]
        if self.sma_periods is None:
            self.sma_periods = [20, 50, 200]  
        if self.ema_periods is None:
            self.ema_periods = [12, 26]
        if self.volume_sma_periods is None:
            self.volume_sma_periods = [20, 50]
        if self.vwap_periods is None:
            self.vwap_periods = [20, 50]

# Global configuration instances
DB_CONFIG = DatabaseConfig()
MODEL_CONFIG = ModelConfig()  
FEATURE_CONFIG = FeatureConfig()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories
for dir_path in [DATA_DIR, MODEL_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Optimal-stopping problem configuration
@dataclass
class OptimalStoppingConfig:
    """Configuration for optimal-stopping sell signal framework"""
    
    # Problem formulation
    approach: str = "upside_left"  # ["upside_left", "hindsight_optimal", "offline_rl"]
    
    # Horizons by token liquidity bucket
    horizon_hours: Dict[str, int] = None  # {"new_tokens": 2, "established": 12, "blue_chip": 24}
    
    # Cost model parameters
    dex_fees: Dict[str, float] = None  # {"raydium": 0.0025, "uniswap_v3": 0.0005}
    slippage_percentiles: Tuple[int, int] = (20, 50)  # Use 20th-50th percentile of recent swaps
    
    # Utility function parameters (for hindsight optimal approach)
    time_penalty_lambda: float = 0.01  # Penalty per hour held
    drawdown_penalty_mu: float = 0.1   # Penalty for max drawdown
    
    # Decision thresholds
    upside_threshold_pct: float = 0.8  # Sell if predicted upside < 0.8%
    drawdown_prob_threshold: float = 0.4  # Sell if P(>5% drawdown) > 40%
    trailing_stop_atr_multiplier: float = 2.5  # Trailing stop at 2.5x ATR
    
    # Token liquidity buckets
    liquidity_buckets: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.horizon_hours is None:
            self.horizon_hours = {
                "new_tokens": 2,      # < 7 days old, < $100k TVL
                "emerging": 6,        # 7-30 days, $100k-$1M TVL  
                "established": 12,    # > 30 days, $1M-$10M TVL
                "blue_chip": 24       # > $10M TVL
            }
            
        if self.dex_fees is None:
            self.dex_fees = {
                "raydium": 0.0025,
                "orca": 0.003,
                "uniswap_v2": 0.003,
                "uniswap_v3": 0.0005,  # Most common tier
                "sushiswap": 0.003,
                "aerodrome": 0.0005
            }
            
        if self.liquidity_buckets is None:
            self.liquidity_buckets = {
                "new_tokens": {"min_age_days": 0, "max_age_days": 7, "min_tvl": 0, "max_tvl": 100000},
                "emerging": {"min_age_days": 7, "max_age_days": 30, "min_tvl": 100000, "max_tvl": 1000000},
                "established": {"min_age_days": 30, "max_age_days": 365, "min_tvl": 1000000, "max_tvl": 10000000},
                "blue_chip": {"min_age_days": 365, "max_age_days": 9999, "min_tvl": 10000000, "max_tvl": float('inf')}
            }

# Global configuration instances - updated
DB_CONFIG = DatabaseConfig()
MODEL_CONFIG = ModelConfig()  
FEATURE_CONFIG = FeatureConfig()
OPTIMAL_STOPPING_CONFIG = OptimalStoppingConfig()
