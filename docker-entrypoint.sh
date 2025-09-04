#!/bin/bash
set -e

# Function to handle graceful shutdown
cleanup() {
    echo "Received shutdown signal, stopping processes..."
    kill -TERM "$child" 2>/dev/null
    wait "$child"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Set default environment variables
export NODE_ENV=${NODE_ENV:-production}
export CLICKHOUSE_HOST=${CLICKHOUSE_HOST:-ch}
export CLICKHOUSE_PORT=${CLICKHOUSE_PORT:-8123}
export CLICKHOUSE_USERNAME=${CLICKHOUSE_USERNAME:-default}
export CLICKHOUSE_PASSWORD=${CLICKHOUSE_PASSWORD:-Sakirs1234**}
export CLICKHOUSE_DB=${CLICKHOUSE_DB:-default}

# Wait for ClickHouse to be ready
echo "Waiting for ClickHouse to be ready..."
until nc -z "$CLICKHOUSE_HOST" "$CLICKHOUSE_PORT" 2>/dev/null || curl -f "http://$CLICKHOUSE_HOST:$CLICKHOUSE_PORT/ping" >/dev/null 2>&1; do
    echo "ClickHouse is unavailable - sleeping"
    sleep 2
done
echo "ClickHouse is ready!"

# Determine which script to run based on NETWORK environment variable
case "$NETWORK" in
    "ethereum")
        echo "Starting Ethereum DEX swaps processor..."
        echo "Block range: ${BLOCK_FROM:-22290737} to ${BLOCK_TO:-latest}"
        yarn ts-node pipes/evm/swaps/cli.ts &
        ;;
    "base")
        echo "Starting Base DEX swaps processor..."
        echo "Block range: ${BLOCK_FROM:-29102278} to ${BLOCK_TO:-latest}"
        yarn ts-node pipes/evm/swaps/cli.ts &
        ;;
    "solana")
        echo "Starting Solana DEX swaps processor..."
        echo "Block range: ${BLOCK_FROM:-345246630} to ${BLOCK_TO:-latest}"
        echo "Cache dump path: ${CACHE_DUMP_PATH:-./cache/solana-cache.jsonl}"
        yarn ts-node pipes/solana/swaps/cli.ts &
        ;;
    *)
        echo "Error: NETWORK environment variable must be set to 'ethereum', 'base', or 'solana'"
        exit 1
        ;;
esac

# Store the PID of the background process
child=$!

# Wait for the process to complete or receive a signal
wait "$child"
