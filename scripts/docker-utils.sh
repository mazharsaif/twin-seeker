#!/bin/bash

# Docker utilities for Twin-Seeker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Build the Docker image
build_image() {
    print_status "Building Twin-Seeker Docker image..."
    docker build -t twin-seeker:latest .
    print_success "Docker image built successfully!"
}

# Start services with Docker Compose
start_services() {
    print_status "Starting Twin-Seeker services..."
    docker-compose up -d
    print_success "Services started successfully!"
    print_status "API available at: http://localhost:8000"
    print_status "Health check: http://localhost:8000/api/v1/face/health"
}

# Stop services
stop_services() {
    print_status "Stopping Twin-Seeker services..."
    docker-compose down
    print_success "Services stopped successfully!"
}

# View logs
view_logs() {
    print_status "Viewing logs..."
    docker-compose logs -f twin-seeker-api
}

# Check service health
check_health() {
    print_status "Checking service health..."
    
    # Check if container is running
    if docker-compose ps | grep -q "twin-seeker-api.*Up"; then
        print_success "Container is running"
    else
        print_error "Container is not running"
        return 1
    fi
    
    # Check API health
    if curl -f http://localhost:8000/api/v1/face/health > /dev/null 2>&1; then
        print_success "API health check passed"
    else
        print_error "API health check failed"
        return 1
    fi
}

# Clean up Docker resources
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    print_success "Cleanup completed!"
}

# Show usage
show_usage() {
    echo "Twin-Seeker Docker Utilities"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     Build the Docker image"
    echo "  start     Start all services"
    echo "  stop      Stop all services"
    echo "  restart   Restart all services"
    echo "  logs      View service logs"
    echo "  health    Check service health"
    echo "  cleanup   Clean up Docker resources"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 start"
    echo "  $0 health"
}

# Main script logic
main() {
    check_docker
    
    case "${1:-help}" in
        build)
            build_image
            ;;
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            sleep 2
            start_services
            ;;
        logs)
            view_logs
            ;;
        health)
            check_health
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 