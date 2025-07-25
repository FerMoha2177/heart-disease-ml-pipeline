#!/bin/bash

# Heart Disease ML API - Local Development Runner
# Fixed to work from project root directory

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
PROJECT_NAME="heart-disease-ml"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"  # Use full path
ENV_FILE="$PROJECT_ROOT/.env"

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

# Function to check if Docker is running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Docker is ready"
}

# Function to check if docker-compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose -f $COMPOSE_FILE"
    elif docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose -f $COMPOSE_FILE"
    else
        print_error "Docker Compose is not available"
        exit 1
    fi
    
    print_success "Using: docker compose with file $COMPOSE_FILE"
}

# Function to create environment file if it doesn't exist
setup_env() {
    print_status "Setting up environment..."
    
    # Change to project root for .env file
    cd "$PROJECT_ROOT"
    
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_warning "Created .env from .env.example. Please update MongoDB connection if needed."
        else
            cat > .env << EOF
# MongoDB Configuration (Docker MongoDB)
MONGODB_CONNECTION_STRING=mongodb://admin:admin123@localhost:27017/healthcare?authSource=admin
MONGODB_DATABASE_NAME=healthcare

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Model Configuration
MODEL_PATH=models/heart_disease_classifier.joblib
MODEL_VERSION=1.0.0

# Logging
LOG_LEVEL=INFO

# Environment
ENVIRONMENT=development
EOF
            print_success "Created default .env file for Docker development"
        fi
    else
        print_success "Environment file already exists"
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    directories=("models" "data/raw" "data/processed" "logs" "notebooks")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        fi
    done
    
    print_success "Directories ready"
}

# Function to show service status
show_status() {
    print_status "Checking service status..."
    
    echo
    echo "=== CONTAINER STATUS ==="
    $COMPOSE_CMD ps
    
    echo
    echo "=== SERVICE HEALTH ==="
    echo "API Health: http://localhost:8000/api/v1/health"
    echo "API Docs: http://localhost:8000/docs"
    echo "MongoDB: mongodb://admin:admin123@localhost:27017/healthcare"
    echo "Mongo Express: http://localhost:8081 (admin/admin123)"
    echo "Jupyter Lab: http://localhost:8888"
    echo
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    # Wait for services to be ready
    sleep 10
    
    # Run health checks
    print_status "Checking API health..."
    if curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        print_success "API is healthy"
    else
        print_warning "API health check failed"
    fi
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."
    $COMPOSE_CMD down
    print_success "Services stopped"
}

# Function to clean up everything
cleanup() {
    print_status "Cleaning up..."
    $COMPOSE_CMD down -v --remove-orphans
    docker system prune -f
    print_success "Cleanup completed"
}

# Function to show logs
show_logs() {
    local service=${1:-""}
    if [ -n "$service" ]; then
        print_status "Showing logs for $service..."
        $COMPOSE_CMD logs -f "$service"
    else
        print_status "Showing all logs..."
        $COMPOSE_CMD logs -f
    fi
}

# Function to show help
show_help() {
    echo "Heart Disease ML API - Local Development Runner"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  start     Start all services (default)"
    echo "  stop      Stop all services"
    echo "  restart   Restart all services"
    echo "  status    Show service status"
    echo "  logs      Show logs (optional: specify service name)"
    echo "  test      Run health checks"
    echo "  clean     Clean up containers and volumes"
    echo "  help      Show this help message"
    echo
    echo "Examples:"
    echo "  $0              # Start all services"
    echo "  $0 start        # Start all services"
    echo "  $0 logs api     # Show API logs"
    echo "  $0 logs         # Show all logs"
    echo "  $0 stop         # Stop all services"
    echo
    echo "Services will be available at:"
    echo "  - API: http://localhost:8000"
    echo "  - API Docs: http://localhost:8000/docs"
    echo "  - MongoDB: localhost:27017"
    echo "  - Mongo Express: http://localhost:8081"
    echo "  - Jupyter: http://localhost:8888"
}

# Main execution
main() {
    local command=${1:-"start"}
    
    echo "=== Heart Disease ML API - Local Development ==="
    echo "Working from: $PROJECT_ROOT"
    echo "Docker Compose file: $COMPOSE_FILE"
    echo
    
    case $command in
        "start")
            check_docker
            check_docker_compose
            setup_env
            create_directories
            # Removed create_mongo_init - not needed!
            
            print_status "Starting services..."
            cd "$PROJECT_ROOT"  # Ensure we're in project root
            $COMPOSE_CMD up -d --build
            
            print_success "Services started successfully!"
            show_status
            
            print_status "Running initial health checks..."
            run_tests
            
            echo
            print_success "Development environment is ready!"
            echo
            echo "Next steps:"
            echo "1. Visit http://localhost:8000/docs for API documentation"
            echo "2. Use Postman to test the endpoints"
            echo "3. Access Jupyter at http://localhost:8888 for ML development"
            echo "4. Monitor logs with: $0 logs"
            echo
            ;;
        "stop")
            check_docker_compose
            stop_services
            ;;
        "restart")
            check_docker_compose
            print_status "Restarting services..."
            cd "$PROJECT_ROOT"
            $COMPOSE_CMD restart
            print_success "Services restarted"
            show_status
            ;;
        "status")
            check_docker_compose
            show_status
            ;;
        "logs")
            check_docker_compose
            show_logs "$2"
            ;;
        "test")
            run_tests
            ;;
        "clean")
            check_docker_compose
            cleanup
            ;;
        "help")
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            echo
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"