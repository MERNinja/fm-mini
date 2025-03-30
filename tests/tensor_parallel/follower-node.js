/**
 * Follower node for batch distribution test
 */
import { io } from 'socket.io-client';

// Check for required args
if (process.argv.length < 3) {
  console.error('Usage: node follower-node.js <node_id>');
  process.exit(1);
}

// Get node ID from command line
const nodeId = process.argv[2];
console.log('Starting follower node with ID:', nodeId);

// Connect to the server
const socket = io('http://localhost:8080');

// Register on connection
socket.on('connect', () => {
  console.log('Connected to server, socket ID:', socket.id);
  
  // Register with server
  socket.emit('register_node', {
    id: nodeId,
    type: 'test',
    capabilities: ['tensor_parallel'],
    status: 'online'
  });
  
  console.log('Registered as', nodeId);
});

// Handle tensor operations
socket.on('operation', (data) => {
  if (data.to === nodeId) {
    console.log(`[${nodeId}] Received operation:`, data.operation);
    console.log(`[${nodeId}] Operation data:`, data.data);
    
    // Handle process_layers operation
    if (data.operation === 'process_layers') {
      const batchNumber = data.data.batchNumber || 1;
      const layers = data.data.layers || [];
      
      console.log(`[${nodeId}] Processing batch ${batchNumber} with ${layers.length} layers`);
      
      // Simulate processing
      setTimeout(() => {
        console.log(`[${nodeId}] Completed processing batch ${batchNumber}`);
        
        // Send result back
        socket.emit('operation_result', {
          from: nodeId,
          to: data.from,
          taskId: data.taskId,
          operation: data.operation,
          result: {
            success: true,
            processingTime: 500,
            layers,
            batchNumber,
            partialResult: `Result from ${nodeId} for batch ${batchNumber}`
          }
        });
      }, 1000);
    }
  }
});

// Handle messages
socket.on('message', (msg) => {
  console.log(`[${nodeId}] Received message:`, msg);
});

// Handle errors
socket.on('connect_error', (err) => {
  console.error(`[${nodeId}] Connection error:`, err);
});

socket.on('disconnect', () => {
  console.log(`[${nodeId}] Disconnected from server`);
  process.exit(0);
});

// Set up logging for various events
const events = [
  'node_registered', 'node_list', 'node_status_update',
  'node_disconnected', 'tensor_model_registered', 'tensor_signal'
];

for (const event of events) {
  socket.on(event, (data) => {
    console.log(`[${nodeId}] Event ${event}:`, data);
  });
}

// Keep the process running
process.on('SIGINT', () => {
  console.log(`[${nodeId}] Shutting down...`);
  socket.disconnect();
  process.exit(0);
}); 