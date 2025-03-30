/**
 * Simple test to verify batch distribution
 */
import { io } from 'socket.io-client';

// Connect to the server
const socket = io('http://localhost:8080');

// Test nodes
const leaderNode = 'test_leader';
const follower1 = 'test_follower1';
const follower2 = 'test_follower2';

// Register a test node
socket.on('connect', () => {
  console.log('Connected to server, socket ID:', socket.id);
  
  // Register as leader node
  socket.emit('register_node', {
    id: leaderNode,
    type: 'test',
    capabilities: ['tensor_parallel'],
    status: 'online'
  });
  
  console.log('Registered as', leaderNode);
  
  // Simulate other nodes already being connected
  socket.emit('message', {
    type: 'register_test_nodes'
  });
  
  // Wait a moment then distribute batch tasks
  setTimeout(() => {
    console.log('Starting batch distribution test...');
    
    // First batch assignment - should go to follower1
    console.log('Sending batch 1 to', follower1);
    socket.emit('operation', {
      from: leaderNode,
      to: follower1,
      taskId: 'task_1',
      operation: 'process_layers',
      data: {
        layers: ['layer_1', 'layer_2'],
        batchNumber: 1,
        params: { test: true }
      }
    });
    
    // Second batch assignment - should go to follower2
    console.log('Sending batch 2 to', follower2);
    socket.emit('operation', {
      from: leaderNode,
      to: follower2,
      taskId: 'task_2',
      operation: 'process_layers',
      data: {
        layers: ['layer_3', 'layer_4'],
        batchNumber: 2,
        params: { test: true }
      }
    });
    
    // Third batch - process locally
    console.log('Processing batch 3 locally');
    
    // Keep the test running for a bit to see logs
    setTimeout(() => {
      console.log('Test complete, disconnecting...');
      socket.disconnect();
    }, 5000);
    
  }, 1000);
});

// Handle messages from the server
socket.on('message', (msg) => {
  console.log('Received message:', msg);
});

// Handle operation results
socket.on('operation_result', (data) => {
  console.log('Received operation result:', data);
  console.log(`Follower ${data.from} processed batch ${data.result.batchNumber}`);
});

// Handle errors
socket.on('connect_error', (err) => {
  console.error('Connection error:', err);
});

socket.on('disconnect', () => {
  console.log('Disconnected from server');
});

// Set up logging for various events
const events = [
  'node_registered', 'node_list', 'node_status_update', 
  'node_disconnected', 'tensor_model_registered', 'tensor_signal'
];

for (const event of events) {
  socket.on(event, (data) => {
    console.log(`Event ${event}:`, data);
  });
}

console.log('Test script running...'); 