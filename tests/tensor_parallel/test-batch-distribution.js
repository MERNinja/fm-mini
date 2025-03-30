/**
 * Test script for batch distribution across nodes in tensor parallelism
 */
import { io } from 'socket.io-client';

// Constants
const SERVER_URL = 'http://localhost:8080';
const NUM_NODES = 3;
const TEST_TIMEOUT = 10000; // 10 seconds instead of 60
const DELAY_BETWEEN_NODES = 1000; // 1 second

/**
 * Node client with batch processing capabilities
 */
class BatchTestNode {
  constructor(id, isLeader = false) {
    this.id = id;
    this.isLeader = isLeader;
    this.socket = null;
    this.connectedPeers = new Set();
    this.receivedBatches = new Set();
    this.layerResults = new Map();
  }
  
  async connect() {
    return new Promise((resolve, reject) => {
      console.log(`[${this.id}] Connecting to ${SERVER_URL}...`);
      
      this.socket = io(SERVER_URL);
      
      this.socket.on('connect', () => {
        console.log(`[${this.id}] Connected with socket ID: ${this.socket.id}`);
        
        // Register with server
        this.socket.emit('register_node', {
          id: this.id,
          type: 'test',
          capabilities: ['tensor_parallel', 'mock_llm'],
          status: 'online'
        });
        
        // Setup event handlers
        this.setupEventHandlers();
        
        resolve();
      });
      
      this.socket.on('connect_error', (error) => {
        console.error(`[${this.id}] Connection error:`, error);
        reject(error);
      });
    });
  }
  
  setupEventHandlers() {
    // Node registration
    this.socket.on('node_registered', (node) => {
      console.log(`[${this.id}] Node registered: ${node.id}`);
      if (node.id !== this.id) {
        this.connectedPeers.add(node.id);
      }
    });
    
    // Handle tensor operations
    this.socket.on('operation', (data) => {
      if (data.to === this.id) {
        console.log(`[${this.id}] 📩 Received operation: ${data.operation} from ${data.from} with batch ${data.data?.batchNumber || 'unknown'}`);
        this.handleOperation(data);
      }
    });
    
    // Handle operation results
    this.socket.on('operation_result', (data) => {
      if (data.to === this.id) {
        console.log(`[${this.id}] 📊 Received result from ${data.from} with batch ${data.result?.batchNumber || 'unknown'}`);
        
        // Store the result
        this.layerResults.set(data.from, data.result);
        
        // Check if we have all results
        if (this.isLeader && this.layerResults.size === this.connectedPeers.size) {
          console.log(`[${this.id}] 🎉 All results received! (${this.layerResults.size} of ${this.connectedPeers.size})`);
          this.summarizeResults();
        } else if (this.isLeader) {
          console.log(`[${this.id}] Waiting for more results... (${this.layerResults.size} of ${this.connectedPeers.size + 1} total)`);
        }
      }
    });
    
    // Debug for all socket events
    this.socket.onAny((event, ...args) => {
      if (event !== 'node_registered' && !event.startsWith('node_')) {
        console.log(`[${this.id}] 🔔 Socket event: ${event}`);
      }
    });
    
    // Disconnect
    this.socket.on('disconnect', (reason) => {
      console.log(`[${this.id}] Disconnected: ${reason}`);
    });
  }
  
  handleOperation(data) {
    const { from, taskId, operation, data: operationData } = data;
    
    if (operation === 'process_layers') {
      const batchNumber = operationData.batchNumber || 1;
      const layers = operationData.layers || [];
      
      this.receivedBatches.add(batchNumber);
      
      console.log(`[${this.id}] ⚙️ Processing batch ${batchNumber} with ${layers.length} layers`);
      
      // Simulate processing - reduce to 1 second to see results faster
      setTimeout(() => {
        console.log(`[${this.id}] ✅ Completed processing batch ${batchNumber}`);
        
        // Send result back
        this.socket.emit('operation_result', {
          from: this.id,
          to: from,
          taskId,
          operation,
          result: {
            success: true,
            processingTime: 500,
            layers,
            batchNumber,
            partialResult: `Result from ${this.id} for batch ${batchNumber}`
          }
        });
      }, 1000);
    }
  }
  
  async startBatchDistribution() {
    if (!this.isLeader) {
      console.log(`[${this.id}] Not a leader node, waiting for operations...`);
      return;
    }
    
    // Wait for peers to connect
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    console.log(`[${this.id}] Starting batch distribution test with ${this.connectedPeers.size} connected peers`);
    
    // Record start time for metrics
    const startTime = Date.now();
    
    // Reset tracking
    this.layerResults.clear();
    
    // Generate mock layers
    const layers = this.generateLayers(24);
    
    // Distribute layers among nodes (including self)
    const allNodes = [this.id, ...Array.from(this.connectedPeers)];
    const layerAssignments = this.partitionLayers(layers, allNodes.length);
    
    console.log(`[${this.id}] 📋 Layer assignments:`);
    for (let i = 0; i < allNodes.length; i++) {
      console.log(`[${this.id}] - ${allNodes[i]}: ${layerAssignments[i].length} layers`);
    }
    
    // Send operations to peers and process own layers
    const taskId = `task_${Date.now()}`;
    
    // Assign batches - one batch per node
    const totalBatches = Math.min(allNodes.length, 3); // Max 3 batches
    
    for (let i = 0; i < allNodes.length; i++) {
      const nodeId = allNodes[i];
      if (nodeId === this.id) {
        // Process own batch locally
        const batchNumber = 1; // Leader always gets batch 1
        console.log(`[${this.id}] Processing batch ${batchNumber} locally`);
        
        // Simulate local processing - reduced to 1 second
        setTimeout(() => {
          const result = {
            success: true,
            processingTime: 500,
            layers: layerAssignments[i],
            batchNumber,
            partialResult: `Result from local node for batch ${batchNumber}`
          };
          
          this.layerResults.set(this.id, result);
          this.receivedBatches.add(batchNumber);
          
          // Check if we have all results
          if (this.layerResults.size === allNodes.length) {
            console.log(`[${this.id}] 🎉 All results received!`);
            this.summarizeResults();
          }
        }, 1000);
      } else {
        // Only send if we have batches to assign
        if (i < totalBatches) {
          const batchNumber = i + 1; // Batches are 1-indexed
          console.log(`[${this.id}] Sending batch ${batchNumber} to ${nodeId}`);
          
          // Send operation to remote node
          this.socket.emit('operation', {
            from: this.id,
            to: nodeId,
            taskId,
            operation: 'process_layers',
            data: {
              layers: layerAssignments[i],
              batchNumber: batchNumber,
              params: { test: true }
            }
          });
        }
      }
    }
  }
  
  summarizeResults() {
    console.log("\n===== BATCH DISTRIBUTION TEST RESULTS =====");
    
    // Check if each node processed only its assigned batch
    for (const [nodeId, result] of this.layerResults.entries()) {
      console.log(`Node ${nodeId}: Processed batch ${result.batchNumber}`);
    }
    
    // Success criteria: Each node processed exactly one batch
    const processedBatches = new Set();
    let duplicateBatches = false;
    
    for (const [_, result] of this.layerResults.entries()) {
      if (processedBatches.has(result.batchNumber)) {
        duplicateBatches = true;
      }
      processedBatches.add(result.batchNumber);
    }
    
    if (duplicateBatches) {
      console.log("\n❌ TEST FAILED: Same batch processed by multiple nodes");
    } else if (processedBatches.size !== this.layerResults.size) {
      console.log("\n❌ TEST FAILED: Not all batches were processed");
    } else {
      console.log("\n✅ TEST PASSED: Each node processed exactly one batch");
    }
    
    console.log("\nProcessed batches:", Array.from(processedBatches));
    console.log("===========================================\n");
  }
  
  generateLayers(count) {
    const layers = [];
    for (let i = 0; i < count; i++) {
      layers.push(`transformer_layer_${i}`);
    }
    return layers;
  }
  
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
    }
  }
}

/**
 * Run the test
 */
async function runTest() {
  const nodes = [];
  let testCompleted = false;
  
  try {
    console.log("=== Starting Batch Distribution Test ===");
    
    // Create and connect leader node
    const leader = new BatchTestNode('leader_node', true);
    await leader.connect();
    nodes.push(leader);
    
    // Create and connect follower nodes
    for (let i = 0; i < NUM_NODES - 1; i++) {
      const id = `follower_node_${i+1}`;
      console.log(`Creating follower ${id}...`);
      
      const node = new BatchTestNode(id);
      await node.connect();
      nodes.push(node);
      
      // Wait between client connections
      await new Promise(resolve => setTimeout(resolve, DELAY_BETWEEN_NODES));
    }
    
    console.log("\nAll nodes connected successfully");
    console.log(`Total nodes: ${nodes.length}`);
    
    // Wait a bit for all nodes to register
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Add result listener to know when test is complete
    const originalSummarize = leader.summarizeResults;
    leader.summarizeResults = function() {
      originalSummarize.call(this);
      testCompleted = true;
    };
    
    // Start the batch distribution test
    await leader.startBatchDistribution();
    
    // Wait for test to complete with timeout
    const maxWaitTime = 15000; // 15 seconds max
    const startWait = Date.now();
    
    while (!testCompleted && (Date.now() - startWait < maxWaitTime)) {
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    // Final wait to make sure all console output is visible
    await new Promise(resolve => setTimeout(resolve, 2000));
    
  } catch (error) {
    console.error("Test error:", error);
  } finally {
    // Clean up
    console.log("\nDisconnecting nodes...");
    for (const node of nodes) {
      node.disconnect();
    }
    console.log("All nodes disconnected");
  }
}

// Run the test
runTest(); 