#!/bin/bash

# Automated script to test tensor parallelism with 3 real nodes
echo "Starting automated tensor parallelism test..."

# Install necessary dependencies if not present
if [ ! -d "node_modules/puppeteer" ]; then
  echo "Installing required dependencies..."
  npm install --save-dev puppeteer uuid socket.io-client
fi

# Make sure servers are not already running
echo "Stopping any existing servers..."
lsof -ti:3000 | xargs kill -9 2>/dev/null
lsof -ti:8080 | xargs kill -9 2>/dev/null
lsof -ti:8081 | xargs kill -9 2>/dev/null

# Start the Vite development server which also starts the WebSocket server
echo "Starting development servers..."
npm run dev &
DEV_PID=$!

# Wait for servers to start
echo "Waiting for servers to initialize (this may take 30 seconds)..."
sleep 30

# Start the test
echo "Executing tensor parallelism test with 3 nodes..."
node tests/tensor-parallel-node-test.js

TEST_EXIT_CODE=$?

# Clean up - kill the dev server
echo "Cleaning up..."
kill $DEV_PID 2>/dev/null
sleep 2

# Make sure any remaining processes on these ports are killed
lsof -ti:3000 | xargs kill -9 2>/dev/null
lsof -ti:8080 | xargs kill -9 2>/dev/null
lsof -ti:8081 | xargs kill -9 2>/dev/null

# Display result
if [ $TEST_EXIT_CODE -eq 0 ]; then
  echo "✅ TEST PASSED: Tensor parallelism working correctly with real nodes!"
else
  echo "❌ TEST FAILED: Tensor parallelism verification failed."
fi

exit $TEST_EXIT_CODE 