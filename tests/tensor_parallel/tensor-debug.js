/**
 * Tensor Parallelism Debug Test
 * This test shows in detail how multiple nodes work together to process a prompt
 */
import { io } from 'socket.io-client';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

// Get __dirname in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Constants
const SERVER_URL = 'http://localhost:8080';
const TEST_PROMPT = 'Write a story about tensor parallelism.';

/**
 * Simple node class that clearly shows tensor operations
 */
class TensorNode {
  constructor(id, isLeader = false) {
    this.id = id;
    this.isLeader = isLeader;
    this.socket = null;
    this.connectedNodes = [];
    this.activeOperations = new Map();
    this.completedOperations = [];
    
    // For tensor parallelism
    this.assignedLayers = [];
    this.layerResults = new Map();
  }
  
  async connect() {
    console.log(`[${this.id}] Connecting to ${SERVER_URL}...`);
    
    return new Promise((resolve, reject) => {
      try {
        this.socket = io(SERVER_URL);
        
        this.socket.on('connect', () => {
          console.log(`[${this.id}] Connected with socket ID: ${this.socket.id}`);
          
          // Register with server
          this.socket.emit('register_node', {
            id: this.id,
            type: this.isLeader ? 'leader' : 'worker',
            capabilities: ['tensor_parallel', 'debug'],
            isLeader: this.isLeader
          });
          
          resolve();
        });
        
        this.socket.on('connect_error', (error) => {
          console.error(`[${this.id}] Connection error:`, error);
          reject(error);
        });
        
        // Track nodes
        this.socket.on('node_registered', (node) => {
          console.log(`[${this.id}] New node registered: ${node.id}`);
          if (node.id !== this.id) {
            this.connectedNodes.push(node);
          }
        });
        
        this.socket.on('node_disconnected', (nodeId) => {
          console.log(`[${this.id}] Node disconnected: ${nodeId}`);
          this.connectedNodes = this.connectedNodes.filter(node => node.id !== nodeId);
        });
        
        // Handle tensor operation requests
        this.socket.on('tensor_operation_request', (data) => {
          if (data.to === this.id) {
            this.processOperation(data);
          }
        });
        
        // Handle tensor operation results
        this.socket.on('tensor_operation_result', (data) => {
          if (data.to === this.id) {
            this.handleOperationResult(data);
          }
        });
      } catch (error) {
        console.error(`[${this.id}] Error:`, error);
        reject(error);
      }
    });
  }
  
  /**
   * Process an incoming tensor operation request
   */
  async processOperation(data) {
    const { from, operation, layers, timestamp, operationId } = data;
    
    console.log(`[${this.id}] ⚙️ RECEIVED OPERATION from ${from}: ${operation}`);
    console.log(`[${this.id}] ⚙️ Processing layers: ${layers ? layers.join(', ') : 'none'}`);
    
    if (operation === 'process_layers') {
      // Store the assigned layers
      this.assignedLayers = layers || [];
      
      // Simulate processing time (proportional to number of layers)
      const processingTime = 500 + (layers.length * 200);
      console.log(`[${this.id}] ⚙️ Processing will take ${processingTime}ms`);
      
      // Track this operation
      this.activeOperations.set(operationId || timestamp, {
        from,
        operation,
        layers,
        startTime: Date.now()
      });
      
      // Simulate actual processing
      await new Promise(resolve => setTimeout(resolve, processingTime));
      
      // Send result back
      console.log(`[${this.id}] ✅ COMPLETED OPERATION ${operation} for layers: ${layers ? layers.join(', ') : 'none'}`);
      
      this.socket.emit('tensor_operation_result', {
        from: this.id,
        to: from,
        operation: 'layer_result',
        layers: layers,
        operationId: operationId || timestamp,
        result: {
          processedLayers: layers,
          processingTime,
          nodeId: this.id
        },
        timestamp: new Date().toISOString()
      });
      
      // Move from active to completed
      const op = this.activeOperations.get(operationId || timestamp);
      if (op) {
        this.activeOperations.delete(operationId || timestamp);
        this.completedOperations.push({
          ...op,
          completionTime: Date.now(),
          duration: Date.now() - op.startTime
        });
      }
    }
  }
  
  /**
   * Handle the result of a tensor operation
   */
  handleOperationResult(data) {
    const { from, operation, layers, result, operationId } = data;
    
    console.log(`[${this.id}] 📥 RECEIVED RESULT from ${from}: ${operation}`);
    
    if (operation === 'layer_result') {
      // Store the result
      this.layerResults.set(from, {
        layers,
        result,
        receivedAt: Date.now()
      });
      
      console.log(`[${this.id}] 📊 Result stats: processed ${result.processedLayers.length} layers in ${result.processingTime}ms`);
      
      // Check if all results are in
      this.checkAllResultsReceived();
    }
  }
  
  /**
   * Check if all expected results have been received
   */
  checkAllResultsReceived() {
    // Only the leader needs to check for completion
    if (!this.isLeader) return;
    
    const expectedNodes = this.connectedNodes.length;
    const receivedResults = this.layerResults.size;
    
    console.log(`[${this.id}] 📊 Results received: ${receivedResults}/${expectedNodes}`);
    
    if (receivedResults >= expectedNodes) {
      console.log(`[${this.id}] 🎉 ALL RESULTS RECEIVED - Processing complete!`);
      console.log(`[${this.id}] 📊 Final result summary:`);
      
      let totalLayers = 0;
      for (const [nodeId, result] of this.layerResults.entries()) {
        console.log(`[${this.id}] - Node ${nodeId}: processed ${result.layers.length} layers in ${result.result.processingTime}ms`);
        totalLayers += result.layers.length;
      }
      
      console.log(`[${this.id}] 📊 Total layers processed by other nodes: ${totalLayers}`);
      console.log(`[${this.id}] 📊 Layers processed by leader: ${this.assignedLayers.length}`);
      console.log(`[${this.id}] 📊 Total layers: ${totalLayers + this.assignedLayers.length}`);
      
      // Generate mock response
      console.log(`[${this.id}] 🤖 Final response generated using tensor parallelism across ${this.layerResults.size + 1} nodes`);
    }
  }
  
  /**
   * If leader, distribute a prompt across connected nodes
   */
  async processPrompt(prompt) {
    if (!this.isLeader) {
      console.log(`[${this.id}] Cannot process prompt: not a leader node`);
      return;
    }
    
    console.log(`\n[${this.id}] 🚀 PROCESSING PROMPT: "${prompt}"`);
    
    if (this.connectedNodes.length === 0) {
      console.log(`[${this.id}] No other nodes available, processing locally`);
      return;
    }
    
    console.log(`[${this.id}] 🔄 Using tensor parallelism with ${this.connectedNodes.length} other nodes`);
    
    // Reset operation tracking
    this.layerResults.clear();
    this.activeOperations.clear();
    this.completedOperations = [];
    
    // Generate layer assignments
    const allLayers = this.generateLayers(12); // 12 transformer layers
    const assignments = this.partitionLayers(allLayers, this.connectedNodes.length + 1);
    
    console.log(`[${this.id}] 📋 Layer assignments:`);
    
    // Assign our own layers
    this.assignedLayers = assignments[0];
    console.log(`[${this.id}] - ${this.id} (self): ${this.assignedLayers.join(', ')}`);
    
    // Assign and send layers to other nodes
    for (let i = 0; i < this.connectedNodes.length; i++) {
      const nodeId = this.connectedNodes[i].id;
      const nodeLayers = assignments[i + 1];
      console.log(`[${this.id}] - ${nodeId}: ${nodeLayers.join(', ')}`);
      
      const operationId = `op_${Date.now()}_${i}`;
      
      // Send tensor operation request to the node
      this.socket.emit('tensor_operation_request', {
        from: this.id,
        to: nodeId,
        operation: 'process_layers',
        layers: nodeLayers,
        operationId,
        timestamp: new Date().toISOString()
      });
      
      console.log(`[${this.id}] 📤 SENT OPERATION to ${nodeId}: process_layers (${nodeLayers.length} layers)`);
    }
    
    // Process our own layers
    console.log(`[${this.id}] ⚙️ PROCESSING OWN LAYERS: ${this.assignedLayers.join(', ')}`);
    
    // Simulate our own processing time
    const processingTime = 500 + (this.assignedLayers.length * 200);
    await new Promise(resolve => setTimeout(resolve, processingTime));
    
    console.log(`[${this.id}] ✅ COMPLETED processing own layers in ${processingTime}ms`);
    
    // Check if we already have all results (in case they came in during our processing)
    this.checkAllResultsReceived();
  }
  
  /**
   * Generate mock transformer layers
   */
  generateLayers(count) {
    const layers = [];
    for (let i = 0; i < count; i++) {
      layers.push(`transformer_layer_${i}`);
    }
    return layers;
  }
  
  /**
   * Partition layers among nodes
   */
  partitionLayers(layers, nodeCount) {
    const result = Array(nodeCount).fill().map(() => []);
    
    for (let i = 0; i < layers.length; i++) {
      const nodeIndex = i % nodeCount;
      result[nodeIndex].push(layers[i]);
    }
    
    return result;
  }
  
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      console.log(`[${this.id}] Disconnected`);
    }
  }
}

/**
 * Run the debug test
 */
async function runDebugTest() {
  console.log('=== Tensor Parallelism Debug Test ===');
  
  const nodes = [];
  
  try {
    // Create and connect leader node
    const leaderNode = new TensorNode('tensor_leader', true);
    await leaderNode.connect();
    nodes.push(leaderNode);
    
    // Create and connect worker nodes
    for (let i = 1; i <= 2; i++) {
      const workerNode = new TensorNode(`tensor_worker_${i}`, false);
      await workerNode.connect();
      nodes.push(workerNode);
      
      // Allow time for node registration to propagate
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    console.log('\nAll nodes connected. Starting test...');
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Process a test prompt
    await leaderNode.processPrompt(TEST_PROMPT);
    
    // Allow time for all communications to complete
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    console.log('\n=== Test completed ===');
    
  } catch (error) {
    console.error('Test error:', error);
  } finally {
    // Disconnect all nodes
    for (const node of nodes) {
      node.disconnect();
    }
  }
}

// Run the test if called directly
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  runDebugTest()
    .then(() => process.exit(0))
    .catch(err => {
      console.error('Unhandled error:', err);
      process.exit(1);
    });
} 