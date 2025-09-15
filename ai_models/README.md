# AI-Powered Sell Signal Prediction for DEX Trading

A comprehensive optimal-stopping framework for predicting optimal sell points after buy signals fire. Built specifically for your DEX data pipeline with support for both Solana and EVM chains.

## 🎯 Problem Formulation

This system solves the **optimal-stopping problem**: at each timestep after entry, decide whether to sell now or keep holding to maximize risk-adjusted PnL net of fees, slippage, and time/risk penalties.

### Three Approaches Implemented

1. **Upside-Left Regression (Baseline)** ✅
   - Predicts remaining profit potential: `M_t = max_{s ∈ [t, t+H]} R(s) - R(t)`
   - Decision: sell if predicted upside < threshold OR drawdown risk > threshold

2. **Hindsight Optimal Labels** ✅
   - Uses utility function: `U(t) = R(t) - λ × holding_time(t) - μ × max_drawdown`
   - Trains classifier to predict optimal exit points

3. **Offline RL** (Future Enhancement)
   - IQL/CQL with behavior cloning from top traders
   - More sophisticated but requires larger datasets

## 🏗️ Architecture

### Data Pipeline
```
ClickHouse DEX Data → Feature Extraction → Model Training → Walk-Forward Evaluation
```

### Models
- **Baseline**: LightGBM + XGBoost ensemble (GPU-accelerated)
- **Sequence**: TCN + Transformer for temporal patterns
- **Cost Model**: Realistic DEX fees + slippage estimation from your data

### Features (200+ engineered)
- **Price/Volatility**: OHLC, returns, ATR, volatility regimes
- **Order Flow**: Buy/sell imbalance, slippage spikes, liquidity changes
- **Smart Money**: Top trader flow analysis from your `top_traders` data
- **Token Context**: Age, holder growth, TVL changes
- **Microstructure**: Market impact, spread proxies

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd ai_models
pip install -r requirements.txt
```

### 2. Configure Database
```python
# Edit config.py
DB_CONFIG.host = "your_clickhouse_host"
DB_CONFIG.database = "your_database"
```

### 3. Run Training Pipeline
```bash
# Train baseline models (recommended start)
python train_sell_models.py --start-date 2024-01-01 --end-date 2024-12-01 --baseline-only

# Train all models (requires more data)
python train_sell_models.py --start-date 2024-01-01 --end-date 2024-12-01
```

### 4. Custom Buy Signals
Replace the example buy signal logic in `train_sell_models.py`:
```python
def extract_buy_signals(self) -> pl.DataFrame:
    # Replace with your actual buy signal logic
    buy_signals_query = """
    SELECT 
        timestamp,
        token_a as token,
        pool_address as pool,
        dex,
        token_a_usdc_price as entry_price,
        abs(amount_b * token_b_usdc_price) as size_usdc
    FROM solana_swaps_raw
    WHERE your_buy_condition_here
    """
```

## 📊 Performance Optimization

### Hardware Utilization (RTX Pro 4500 + 128GB RAM)
- **Baseline Training**: GPU-accelerated LightGBM/XGBoost
- **Sequence Models**: Multi-GPU distributed training with PyTorch Lightning
- **Data Processing**: Polars for high-performance data manipulation
- **Memory**: Extensive feature caching in your 128GB RAM

### Token Liquidity Buckets
```python
LIQUIDITY_BUCKETS = {
    "new_tokens": {"horizon_hours": 2, "min_tvl": 0, "max_tvl": 100000},
    "emerging": {"horizon_hours": 6, "min_tvl": 100000, "max_tvl": 1000000}, 
    "established": {"horizon_hours": 12, "min_tvl": 1000000, "max_tvl": 10000000},
    "blue_chip": {"horizon_hours": 24, "min_tvl": 10000000, "max_tvl": float('inf')}
}
```

## 🎯 Key Features

### Cost-Aware Modeling
- Realistic DEX fees per protocol (Raydium: 0.25%, Uniswap V3: 0.05%, etc.)
- Dynamic slippage estimation from recent similar-sized trades
- Price impact modeling using pool reserves

### Risk Management
- Trailing stops based on ATR multiples
- Maximum drawdown prediction
- VaR/CVaR tail risk metrics

### Walk-Forward Evaluation
- Time-series aware train/test splits
- Realistic PnL simulation with costs
- Comprehensive benchmarking vs. fixed rules

## 📈 Example Results Format

```
# Sell Signal Model Evaluation Report

## Performance Summary
- **Total Trades**: 1,847
- **Win Rate**: 67.2%
- **Average Return**: 3.4%
- **Sharpe Ratio**: 1.83

## Risk Metrics  
- **Maximum Drawdown**: 8.2%
- **VaR (95%)**: -2.1%
- **Profit Factor**: 2.34

## Exit Analysis
- **Model Exits**: 45.3%
- **Stop Loss Exits**: 23.1% 
- **Timeout Exits**: 31.6%
```

## 🔧 Configuration

### Model Parameters (config.py)
```python
@dataclass
class OptimalStoppingConfig:
    # Decision thresholds
    upside_threshold_pct: float = 0.8  # Sell if upside < 0.8%
    drawdown_prob_threshold: float = 0.4  # Sell if P(drawdown) > 40%
    trailing_stop_atr_multiplier: float = 2.5
    
    # Cost model
    dex_fees: Dict[str, float] = {
        "raydium": 0.0025,
        "uniswap_v3": 0.0005,
        # ...
    }
```

### Feature Engineering
```python
@dataclass 
class FeatureConfig:
    rsi_periods: List[int] = [14, 21]
    sma_periods: List[int] = [20, 50, 200]
    smart_money_threshold: float = 10000  # USD
    whale_threshold: float = 100000  # USD
```

## 🔄 Live Trading Integration

### Real-Time Inference
```python
from models.baseline_models import SellDecisionEngine

# Load trained model
model = SellDecisionEngine()
model.load_models('path/to/saved/models')

# For each active position
for position in active_positions:
    current_features = extract_live_features(position)
    should_sell = model.should_sell(current_features, position.bucket)
    
    if should_sell:
        execute_sell_order(position)
```

## 📚 Advanced Usage

### Custom Feature Engineering
```python
# Add your own features in data_extraction.py
def extract_custom_features(self, token, pool, start_time, end_time):
    query = f"""
    SELECT 
        timestamp,
        your_custom_feature_1,
        your_custom_feature_2
    FROM your_custom_table
    WHERE token = '{token}'
    """
    return self.client.query_df(query)
```

### Model Ensembling
```python
# Combine multiple models
ensemble_decision = (
    0.4 * baseline_model.should_sell(data, bucket) +
    0.3 * tcn_model.predict(data) + 
    0.3 * transformer_model.predict(data)
) > 0.5
```

## 🛠️ Troubleshooting

### Common Issues
1. **GPU Memory**: Reduce batch size in `MODEL_CONFIG.batch_size`
2. **Insufficient Data**: Increase date range or reduce `sequence_length`
3. **ClickHouse Connection**: Verify database credentials in `config.py`

### Performance Tips
1. Use `--baseline-only` for faster initial testing
2. Cache feature datasets to disk for repeated experiments
3. Use `LIMIT` in queries during development

## 📖 References

Based on the optimal-stopping framework you provided, leveraging:
- Your existing ClickHouse schema (`solana_swaps_raw`, `vols_candles`, `top_traders`)
- Token liquidity classification system
- Smart money flow analysis
- Realistic cost modeling with DEX fees and slippage

## 🤝 Next Steps

1. **Replace Buy Signal Logic**: Implement your actual buy signal detection
2. **Tune Parameters**: Optimize thresholds based on your risk tolerance
3. **Expand Features**: Add domain-specific features from your data
4. **Deploy Live**: Integrate with your trading infrastructure
5. **Offline RL**: Implement IQL/CQL for more sophisticated policies

---

Built for high-performance DEX trading with your RTX Pro 4500 + 128GB setup 🚀
