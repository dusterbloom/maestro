#!/bin/bash
# test-genai-migration.sh - Script to test the GenAI Processors migration

set -e

echo "🚀 Testing GenAI Processors Migration"
echo "====================================="

# Function to check if a service is healthy
check_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    echo "🔍 Checking $service_name at $url..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null; then
            echo "✅ $service_name is healthy"
            return 0
        fi
        
        echo "⏳ Attempt $attempt/$max_attempts: $service_name not ready..."
        sleep 2
        ((attempt++))
    done
    
    echo "❌ $service_name failed to become healthy"
    return 1
}

# Function to run tests
run_test() {
    local test_name=$1
    local compose_file=$2
    
    echo ""
    echo "🧪 Running test: $test_name"
    echo "Using compose file: $compose_file"
    echo "----------------------------------------"
    
    # Start services
    echo "🔄 Starting services..."
    docker-compose -f "$compose_file" up -d
    
    # Wait for services to be healthy
    sleep 10
    
    # Check orchestrator health
    if check_service "GenAI Orchestrator" "http://localhost:8000/health"; then
        echo "✅ GenAI Orchestrator is running"
        
        # Get metrics
        echo "📊 Getting metrics..."
        curl -s "http://localhost:8000/metrics" | jq . || echo "Metrics endpoint response received"
        
        # Check frontend
        if check_service "Frontend UI" "http://localhost:3001"; then
            echo "✅ Frontend UI is accessible"
        else
            echo "⚠️  Frontend UI check failed"
        fi
        
    else
        echo "❌ GenAI Orchestrator health check failed"
        echo "📋 Container logs:"
        docker-compose -f "$compose_file" logs orchestrator-genai
        docker-compose -f "$compose_file" down
        return 1
    fi
    
    # Keep running for manual testing
    echo ""
    echo "🎉 Test environment is running!"
    echo "📱 Frontend URL: http://localhost:3001"
    echo "🔗 Orchestrator API: http://localhost:8000"
    echo "📊 Health check: http://localhost:8000/health"
    echo "📈 Metrics: http://localhost:8000/metrics"
    echo ""
    echo "Press Enter to continue to next test or Ctrl+C to stop..."
    read -r
    
    # Stop services
    echo "🛑 Stopping test environment..."
    docker-compose -f "$compose_file" down
}

# Test 1: GenAI Processors only
run_test "GenAI Processors Only" "docker-compose.genai-only.yml"

# Test 2: Side-by-side comparison (if user wants it)
echo ""
echo "🤔 Do you want to run the side-by-side comparison test? (y/n)"
read -r answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    run_test "Side-by-Side Comparison" "docker-compose.genai-test.yml"
else
    echo "✅ Skipping side-by-side test"
fi

echo ""
echo "🎉 All tests completed!"
echo "✨ GenAI Processors migration testing finished"
