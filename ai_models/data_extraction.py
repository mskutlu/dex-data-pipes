"""
ClickHouse data extraction for optimal-stopping sell signal prediction
Builds features from your existing DEX data schema
"""
import asyncio
import pandas as pd
import polars as pl
import clickhouse_connect
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from config import DB_CONFIG, OPTIMAL_STOPPING_CONFIG, FEATURE_CONFIG

class DEXDataExtractor:
    """Extract and preprocess features from ClickHouse DEX data"""
    
    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=DB_CONFIG.host,
            port=DB_CONFIG.port,
            database=DB_CONFIG.database,
            username=DB_CONFIG.username,
            password=DB_CONFIG.password
        )
        
    def get_token_liquidity_bucket(self, token: str, timestamp: datetime) -> str:
        """Classify token into liquidity bucket based on age and TVL"""
        query = f"""
        WITH 
            token_age AS (
                SELECT dateDiff('day', token_a_creation_date, '{timestamp}') as age_days
                FROM solana_swaps_raw 
                WHERE token_a = '{token}' 
                AND timestamp <= '{timestamp}'
                ORDER BY timestamp DESC 
                LIMIT 1
            ),
            recent_tvl AS (
                SELECT avgIf(pool_tvl_usdc, pool_tvl_usdc > 0) as avg_tvl
                FROM solana_swaps_raw
                WHERE token_a = '{token}'
                AND timestamp >= '{timestamp}' - INTERVAL 1 HOUR
                AND timestamp <= '{timestamp}'
            )
        SELECT 
            any(ta.age_days) as age_days,
            any(rt.avg_tvl) as avg_tvl_usdc
        FROM token_age ta
        CROSS JOIN recent_tvl rt
        """
        
        result = self.client.query(query)
        if not result.result_rows:
            return "new_tokens"
            
        age_days, avg_tvl = result.result_rows[0]
        
        # Classify based on configuration
        for bucket, criteria in OPTIMAL_STOPPING_CONFIG.liquidity_buckets.items():
            if (criteria["min_age_days"] <= age_days <= criteria["max_age_days"] and
                criteria["min_tvl"] <= (avg_tvl or 0) <= criteria["max_tvl"]):
                return bucket
                
        return "new_tokens"  # Default fallback
        
    def extract_solana_10s_features(self, token: str, pool: str, 
                                   start_time: datetime, end_time: datetime) -> pl.DataFrame:
        """Extract 10-second resolution features for Solana"""
        
        query = f"""
        WITH candles_raw AS (
            SELECT
                timestamp,
                finalizeAggregation(close_token_a_usdc) as close_price,
                finalizeAggregation(open_token_a_usdc) as open_price,
                finalizeAggregation(high_token_a_usdc) as high_price,
                finalizeAggregation(low_token_a_usdc) as low_price,
                finalizeAggregation(volume_usdc) as volume_usdc,
                finalizeAggregation(swap_count) as swap_count,
                finalizeAggregation(avg_slippage_pct) as avg_slippage_pct,
                finalizeAggregation(max_pool_tvl_usdc) as max_pool_tvl_usdc
            FROM solana_dex_swaps_10s_candles
            WHERE token_a = '{token}'
            AND pool_address = '{pool}'
            AND timestamp >= '{start_time}'
            AND timestamp <= '{end_time}'
            ORDER BY timestamp
        ),
        enhanced_features AS (
            SELECT *,
                -- Price features
                log(close_price / lag(close_price, 1) OVER (ORDER BY timestamp)) as log_return_10s,
                log(close_price / lag(close_price, 18) OVER (ORDER BY timestamp)) as log_return_3min,
                log(close_price / lag(close_price, 36) OVER (ORDER BY timestamp)) as log_return_6min,
                log(close_price / lag(close_price, 180) OVER (ORDER BY timestamp)) as log_return_30min,
                
                -- Volatility features  
                stddev(log(close_price / lag(close_price, 1) OVER (ORDER BY timestamp))) 
                    OVER (ORDER BY timestamp ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) as volatility_3min,
                stddev(log(close_price / lag(close_price, 1) OVER (ORDER BY timestamp)))
                    OVER (ORDER BY timestamp ROWS BETWEEN 35 PRECEDING AND CURRENT ROW) as volatility_6min,
                
                -- ATR-like range
                avg(high_price - low_price) 
                    OVER (ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as atr_20_periods,
                
                -- Volume features
                avg(volume_usdc) OVER (ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as volume_sma_20,
                volume_usdc / nullIf(avg(volume_usdc) OVER (ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) as volume_ratio,
                
                -- Liquidity features
                max_pool_tvl_usdc / lag(max_pool_tvl_usdc, 1) OVER (ORDER BY timestamp) - 1 as tvl_change_pct,
                
                -- Slippage trends
                avg(avg_slippage_pct) OVER (ORDER BY timestamp ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) as slippage_sma_12,
                avg_slippage_pct > avg(avg_slippage_pct) OVER (ORDER BY timestamp ROWS BETWEEN 35 PRECEDING AND CURRENT ROW) as slippage_spike
                
            FROM candles_raw
        )
        SELECT * FROM enhanced_features
        WHERE timestamp >= '{start_time}' + INTERVAL 6 MINUTE  -- Skip initial periods with nulls
        """
        
        return pl.from_pandas(self.client.query_df(query))
        
    def extract_smart_money_features(self, token: str, pool: str, 
                                   start_time: datetime, end_time: datetime) -> pl.DataFrame:
        """Extract smart money flow features using top_traders data"""
        
        query = f"""
        WITH smart_traders AS (
            -- Get top traders in the period (top 10% by PnL)
            SELECT wallet
            FROM solana_swaps_new.top_traders(
                start_date='{start_time}' - INTERVAL 7 DAY,
                end_date='{end_time}',
                allowed_quote_tokens=array(untuple(allowed_quote_tokens()))
            )
            WHERE pnl_percent > 0 
            AND token_win_ratio > 0.6
            ORDER BY pnl_percent DESC
            LIMIT 100
        ),
        smart_money_trades AS (
            SELECT 
                toStartOfInterval(timestamp, INTERVAL 1 MINUTE) as minute,
                sum(if(amount_a < 0, abs(amount_a * token_a_usdc_price), 0)) as smart_buy_volume,
                sum(if(amount_a > 0, abs(amount_a * token_a_usdc_price), 0)) as smart_sell_volume,
                count(if(amount_a < 0, 1, null)) as smart_buy_count,
                count(if(amount_a > 0, 1, null)) as smart_sell_count
            FROM solana_swaps_raw s
            INNER JOIN smart_traders st ON s.account = st.wallet
            WHERE s.token_a = '{token}'
            AND s.pool_address = '{pool}'  
            AND s.timestamp >= '{start_time}'
            AND s.timestamp <= '{end_time}'
            GROUP BY minute
            ORDER BY minute
        )
        SELECT 
            minute as timestamp,
            smart_buy_volume,
            smart_sell_volume,
            smart_buy_count, 
            smart_sell_count,
            smart_sell_volume / nullIf(smart_buy_volume + smart_sell_volume, 0) as smart_sell_ratio,
            -- Rolling features
            sum(smart_sell_volume) OVER (ORDER BY minute ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as smart_sell_volume_10min,
            sum(smart_buy_volume) OVER (ORDER BY minute ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as smart_buy_volume_10min
        FROM smart_money_trades
        """
        
        return pl.from_pandas(self.client.query_df(query))
        
    def extract_order_flow_features(self, token: str, pool: str,
                                  start_time: datetime, end_time: datetime) -> pl.DataFrame:
        """Extract order flow and microstructure features"""
        
        query = f"""
        WITH minute_flows AS (
            SELECT 
                toStartOfMinute(timestamp) as minute,
                sum(if(amount_a < 0, abs(amount_b * token_b_usdc_price), 0)) as buy_volume_usdc,
                sum(if(amount_a > 0, abs(amount_b * token_b_usdc_price), 0)) as sell_volume_usdc,
                count(if(amount_a < 0, 1, null)) as buy_count,
                count(if(amount_a > 0, 1, null)) as sell_count,
                avg(abs(slippage_pct)) as avg_slippage,
                quantile(0.8)(abs(slippage_pct)) as p80_slippage,
                avg(pool_tvl_usdc) as avg_tvl
            FROM solana_swaps_raw
            WHERE token_a = '{token}'
            AND pool_address = '{pool}'
            AND timestamp >= '{start_time}'
            AND timestamp <= '{end_time}'
            GROUP BY minute
            ORDER BY minute
        )
        SELECT 
            minute as timestamp,
            buy_volume_usdc,
            sell_volume_usdc,
            buy_count,
            sell_count,
            -- Order flow imbalance
            (buy_volume_usdc - sell_volume_usdc) / nullIf(buy_volume_usdc + sell_volume_usdc, 0) as volume_imbalance,
            (buy_count - sell_count) / nullIf(buy_count + sell_count, 0) as count_imbalance,
            
            -- Slippage and market impact
            avg_slippage,
            p80_slippage,
            avg_slippage > lag(avg_slippage, 5) OVER (ORDER BY minute) as slippage_increasing,
            
            -- Liquidity changes
            avg_tvl,
            avg_tvl / lag(avg_tvl, 1) OVER (ORDER BY minute) - 1 as tvl_change_1min,
            
            -- Rolling aggregates
            avg(volume_imbalance) OVER (ORDER BY minute ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as imbalance_5min_avg,
            stddev(volume_imbalance) OVER (ORDER BY minute ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as imbalance_10min_std
            
        FROM minute_flows
        """
        
        return pl.from_pandas(self.client.query_df(query))
        
    def calculate_cost_model(self, token: str, pool: str, trade_size_usdc: float,
                           timestamp: datetime, dex_name: str) -> Tuple[float, float]:
        """Calculate realistic trading costs (fees + slippage)"""
        
        # Get DEX fee
        dex_fee = OPTIMAL_STOPPING_CONFIG.dex_fees.get(dex_name.lower(), 0.003)
        
        # Estimate slippage from recent similar-sized trades
        p_low, p_high = OPTIMAL_STOPPING_CONFIG.slippage_percentiles
        
        slippage_query = f"""
        SELECT 
            quantile({p_low/100})(abs(slippage_pct)) as p{p_low}_slippage,
            quantile({p_high/100})(abs(slippage_pct)) as p{p_high}_slippage
        FROM solana_swaps_raw
        WHERE token_a = '{token}'
        AND pool_address = '{pool}'
        AND timestamp >= '{timestamp}' - INTERVAL 1 HOUR
        AND timestamp <= '{timestamp}'
        AND abs(amount_b * token_b_usdc_price) BETWEEN {trade_size_usdc * 0.5} AND {trade_size_usdc * 2.0}
        """
        
        result = self.client.query(slippage_query)
        if result.result_rows:
            slippage_low, slippage_high = result.result_rows[0]
            avg_slippage = (slippage_low + slippage_high) / 2 if slippage_low and slippage_high else 0.01
        else:
            avg_slippage = 0.01  # Default 1% slippage
            
        return dex_fee, avg_slippage / 100  # Convert slippage to decimal
        
    def build_training_dataset(self, buy_signals: pl.DataFrame, 
                             start_date: str, end_date: str) -> pl.DataFrame:
        """
        Build complete training dataset with features and labels
        
        buy_signals should have columns: timestamp, token, pool, dex, entry_price, size_usdc
        """
        
        training_data = []
        
        for row in buy_signals.iter_rows(named=True):
            token = row['token']
            pool = row['pool'] 
            entry_time = row['timestamp']
            entry_price = row['entry_price']
            trade_size = row['size_usdc']
            dex = row['dex']
            
            # Determine liquidity bucket and horizon
            bucket = self.get_token_liquidity_bucket(token, entry_time)
            horizon_hours = OPTIMAL_STOPPING_CONFIG.horizon_hours[bucket]
            
            # Set time windows
            pre_window = timedelta(minutes=30)  # 30min before for features
            post_window = timedelta(hours=horizon_hours)
            
            feature_start = entry_time - pre_window
            feature_end = entry_time + post_window
            
            try:
                # Extract all feature sets
                price_features = self.extract_solana_10s_features(token, pool, feature_start, feature_end)
                smart_money = self.extract_smart_money_features(token, pool, feature_start, feature_end) 
                order_flow = self.extract_order_flow_features(token, pool, feature_start, feature_end)
                
                # Get cost model
                dex_fee, slippage = self.calculate_cost_model(token, pool, trade_size, entry_time, dex)
                
                # Calculate upside-left labels for each timestep after entry
                entry_idx = price_features.filter(pl.col('timestamp') >= entry_time).select(pl.first()).collect()
                if len(entry_idx) == 0:
                    continue
                    
                post_entry_data = price_features.filter(pl.col('timestamp') >= entry_time)
                
                # For each timestep, calculate max future return (upside-left)
                for i, price_row in enumerate(post_entry_data.iter_rows(named=True)):
                    current_time = price_row['timestamp']
                    current_price = price_row['close_price']
                    
                    # Calculate net return at current time
                    net_return_current = (current_price / entry_price) * (1 - dex_fee - slippage) - 1
                    
                    # Find max future return within remaining horizon
                    remaining_data = post_entry_data.slice(i, None)
                    if len(remaining_data) == 0:
                        continue
                        
                    future_prices = remaining_data.select('close_price').to_series()
                    max_future_price = future_prices.max()
                    max_future_return = (max_future_price / entry_price) * (1 - dex_fee - slippage) - 1
                    
                    # Upside-left = max_future_return - current_return
                    upside_left = max_future_return - net_return_current
                    
                    # Build feature vector for this timestep
                    features = {
                        'timestamp': current_time,
                        'token': token,
                        'pool': pool,
                        'bucket': bucket,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'net_return_current': net_return_current,
                        'upside_left': upside_left,
                        'dex_fee': dex_fee,
                        'slippage': slippage,
                        'trade_size_usdc': trade_size,
                        'minutes_since_entry': (current_time - entry_time).total_seconds() / 60
                    }
                    
                    # Add price features
                    features.update({f'price_{k}': v for k, v in price_row.items() if k != 'timestamp'})
                    
                    # Add smart money features (join by timestamp)
                    sm_row = smart_money.filter(pl.col('timestamp') == current_time)
                    if len(sm_row) > 0:
                        sm_dict = sm_row.to_dicts()[0]
                        features.update({f'smart_{k}': v for k, v in sm_dict.items() if k != 'timestamp'})
                    
                    # Add order flow features  
                    of_row = order_flow.filter(pl.col('timestamp') == current_time)
                    if len(of_row) > 0:
                        of_dict = of_row.to_dicts()[0]  
                        features.update({f'flow_{k}': v for k, v in of_dict.items() if k != 'timestamp'})
                    
                    training_data.append(features)
                    
            except Exception as e:
                print(f"Error processing {token} at {entry_time}: {e}")
                continue
                
        return pl.DataFrame(training_data)

# Example usage query for buy signals - you'll need to adapt this to your buy signal logic
BUY_SIGNALS_QUERY = """
-- Example: Simple volume breakout buy signals
-- Replace this with your actual buy signal logic
WITH volume_breakouts AS (
    SELECT 
        timestamp,
        token_a as token,
        pool_address as pool,
        dex,
        token_a_usdc_price as entry_price,
        abs(amount_b * token_b_usdc_price) as size_usdc
    FROM solana_swaps_raw s1
    WHERE timestamp >= '{start_date}'
    AND timestamp <= '{end_date}' 
    AND token_b IN allowed_quote_tokens()
    -- Add your buy signal conditions here
    AND abs(amount_b * token_b_usdc_price) > 1000  -- Min $1k trades
)
SELECT * FROM volume_breakouts
ORDER BY timestamp
"""
