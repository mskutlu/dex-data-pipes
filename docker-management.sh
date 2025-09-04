#!/bin/bash

# Docker management script for dex-data-pipes

set -e

show_help() {
    echo "Docker management script for dex-data-pipes"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build         Build all Docker images"
    echo "  start         Start all services"
    echo "  stop          Stop all services"
    echo "  restart       Restart all services"
    echo "  logs          Show logs for all services"
    echo "  logs-eth      Show logs for Ethereum processor"
    echo "  logs-base     Show logs for Base processor"
    echo "  logs-solana   Show logs for Solana processor"
    echo "  logs-ch       Show logs for ClickHouse"
    echo "  status        Show status of all services"
    echo "  clean         Stop and remove all containers and volumes"
    echo "  help          Show this help message"
}

case "$1" in
    build)
        echo "Building Docker images..."
        docker-compose build --no-cache
        ;;
    start)
        echo "Starting all services..."
        docker-compose up -d
        echo "Services started. Use './docker-management.sh logs' to view logs."
        ;;
    stop)
        echo "Stopping all services..."
        docker-compose down
        ;;
    restart)
        echo "Restarting all services..."
        docker-compose restart
        ;;
    logs)
        echo "Showing logs for all services (Ctrl+C to exit)..."
        docker-compose logs -f
        ;;
    logs-eth)
        echo "Showing logs for Ethereum processor (Ctrl+C to exit)..."
        docker-compose logs -f ethereum-swaps
        ;;
    logs-base)
        echo "Showing logs for Base processor (Ctrl+C to exit)..."
        docker-compose logs -f base-swaps
        ;;
    logs-solana)
        echo "Showing logs for Solana processor (Ctrl+C to exit)..."
        docker-compose logs -f solana-swaps
        ;;
    logs-ch)
        echo "Showing logs for ClickHouse (Ctrl+C to exit)..."
        docker-compose logs -f ch
        ;;
    status)
        echo "Service status:"
        docker-compose ps
        ;;
    clean)
        echo "Stopping and cleaning up all containers and volumes..."
        docker-compose down -v
        docker system prune -f
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
