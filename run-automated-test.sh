#!/bin/bash

# Script to run the automated tensor parallelism test
echo "Starting automated tensor parallelism test..."

# Start the server in the background
echo "Starting server..."
node server.js &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Run the automated test
echo "Running automated test..."
node tests/automated-tensor-test.js
TEST_EXIT_CODE=$?

# Clean up
echo "Cleaning up..."
kill $SERVER_PID

# Check test result
if [ $TEST_EXIT_CODE -eq 0 ]; then
  echo "✅ TEST PASSED: Tensor parallelism is working correctly!"
else
  echo "❌ TEST FAILED: Tensor parallelism verification failed."
fi

exit $TEST_EXIT_CODE 