#!/bin/bash

# Test script for tensor parallelism with 3 nodes
echo "Starting tensor parallelism test with 3 nodes..."

# Open 3 browser windows to different ports
# We'll use the same application but different browser windows
open -a "Google Chrome" http://localhost:3000
sleep 2
open -a "Google Chrome" http://localhost:3000
sleep 2
open -a "Google Chrome" http://localhost:3000

echo "All browser instances launched. Please connect all nodes and test tensor parallelism." 