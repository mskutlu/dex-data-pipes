#!/bin/bash

# Unified deployment management script for all networks

set -e

show_help() {
    echo "DEX Data Pipes Unified Deployment Management"
    echo ""
    echo "Usage: $0 [COMMAND] [NETWORK]"
    echo ""
    echo "Commands:"
    echo "  build         Build Docker image"
    echo "  start         Start processors"
    echo "  stop          Stop processors" 
    echo "  restart       Restart processors"
    echo "  logs          Show logs (Ctrl+C to exit)"
    echo "  status        Show container status"
    echo "  clean         Stop and remove containers"
    echo "  help          Show this help message"
    echo ""
    echo "Networks (optional):"
    echo "  ethereum      Ethereum network only"
    echo "  base          Base network only"
    echo "  solana        Solana network only"
    echo "  all           All networks (default)"
    echo ""
    echo "Examples:"
    echo "  $0 start all              # Start all networks"
    echo "  $0 start ethereum         # Start only Ethereum"
    echo "  $0 logs base              # Show logs for Base only"
    echo "  $0 stop                   # Stop all networks"
}

NETWORK=${2:-all}

case "$1" in
    build)
        echo "Building Docker image..."
        docker-compose -f docker-compose.deploy.yml --env-file .env.deploy --profile all build --no-cache
        ;;
    start)
        echo "Starting $NETWORK processor(s)..."
        docker-compose -f docker-compose.deploy.yml --env-file .env.deploy --profile $NETWORK up -d
        echo "$NETWORK processor(s) started. Use './deploy.sh logs $NETWORK' to view logs."
        ;;
    stop)
        echo "Stopping $NETWORK processor(s)..."
        docker-compose -f docker-compose.deploy.yml --env-file .env.deploy --profile $NETWORK down
        ;;
    restart)
        echo "Restarting $NETWORK processor(s)..."
        docker-compose -f docker-compose.deploy.yml --env-file .env.deploy --profile $NETWORK restart
        ;;
    logs)
        echo "Showing logs for $NETWORK processor(s) (Ctrl+C to exit)..."
        docker-compose -f docker-compose.deploy.yml --env-file .env.deploy --profile $NETWORK logs -f
        ;;
    status)
        echo "Processor status:"
        docker-compose -f docker-compose.deploy.yml --env-file .env.deploy ps
        ;;
    clean)
        echo "Stopping and cleaning up $NETWORK processor(s)..."
        docker-compose -f docker-compose.deploy.yml --env-file .env.deploy --profile $NETWORK down -v
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
