#!/bin/bash

# Full end-to-end test for tensor parallelism
echo "Starting end-to-end tensor parallelism test..."

# Start the server in the background
echo "Starting server..."
node server.js &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Start the test observer in the background
echo "Starting test observer..."
node tests/tensor-parallelism-test.js &
TEST_PID=$!

# Open 3 browser windows
echo "Opening browser instances..."
open -a "Google Chrome" http://localhost:3000
sleep 2
open -a "Google Chrome" http://localhost:3000
sleep 2
open -a "Google Chrome" http://localhost:3000

echo "Test environment ready!"
echo "1. Load the model on all 3 nodes"
echo "2. On one node, enter the prompt: 'write me a short story'"
echo "3. Observe the test output for verification"

# Wait for test to complete
wait $TEST_PID
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