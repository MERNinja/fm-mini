/**
 * Simple test script for tensor parallelism infrastructure
 * This is a minimal test that doesn't actually load models
 */
import { io } from 'socket.io-client';
import path from 'path';
import fs from 'fs';

// Basic mock client for testing
class SimpleMockClient {
  constructor(config = {}) {
    this.config = {
      id: `node_${Math.random().toString(36).substring(2, 9)}`,
      serverUrl: 'http://localhost:8099', // Will be auto-updated by run-tests.js
      ...config
    };
    
    this.socket = null;
    this.status = 'idle';
    this.connectedNodes = [];
  }
  
  async connect() {
    return new Promise((resolve, reject) => {
      try {
        console.log(`[${this.config.id}] Connecting to ${this.config.serverUrl}...`);
        this.socket = io(this.config.serverUrl);
        
        this.socket.on('connect', () => {
          console.log(`[${this.config.id}] Connected with socket ID: ${this.socket.id}`);
          
          // Register as a node
          this.socket.emit('register_node', {
            id: this.config.id,
            type: 'mock',
            capabilities: ['tensor_parallel', 'mock'],
          });
          
          this.status = 'connected';
          resolve(true);
        });
        
        this.socket.on('connect_error', (error) => {
          console.error(`[${this.config.id}] Connection error:`, error);
          reject(error);
        });
        
        // Keep track of connected nodes
        this.socket.on('node_registered', (node) => {
          console.log(`[${this.config.id}] New node registered:`, node.id);
          this.connectedNodes.push(node);
        });
        
        this.socket.on('node_disconnected', (nodeId) => {
          console.log(`[${this.config.id}] Node disconnected:`, nodeId);
          this.connectedNodes = this.connectedNodes.filter(node => node.id !== nodeId);
        });
        
        this.socket.on('disconnect', () => {
          console.log(`[${this.config.id}] Disconnected from server`);
        });
        
      } catch (error) {
        console.error(`[${this.config.id}] Error connecting:`, error);
        reject(error);
      }
    });
  }
  
  getNodes() {
    return new Promise((resolve) => {
      // Listen for node_list event response
      const listener = (nodeList) => {
        this.socket.off('node_list', listener); // Remove listener after receiving
        resolve(nodeList || []);
      };
      
      // Listen for the node_list event
      this.socket.on('node_list', listener);
      
      // Emit get_nodes without callback
      this.socket.emit('get_nodes');
      
      // Timeout if response doesn't arrive
      setTimeout(() => {
        this.socket.off('node_list', listener); // Remove listener
        console.warn(`[${this.config.id}] get_nodes response not received, returning empty array`);
        resolve([]);
      }, 3000);
    });
  }
  
  broadcast(message) {
    this.socket.emit('message', {
      type: message.type || 'text',
      from: this.config.id,
      content: message.content,
      timestamp: new Date().toISOString()
    });
  }
  
  disconnect() {
    if (this.socket) {
      this.socket.emit('unregister_node', { id: this.config.id });
      this.socket.disconnect();
      this.socket = null;
      console.log(`[${this.config.id}] Disconnected from server`);
    }
  }
}

// Main test function
async function runTest() {
  console.log('=== Starting Simple Test ===');
  const clients = [];
  
  try {
    // Create and connect 3 clients
    for (let i = 1; i <= 3; i++) {
      const client = new SimpleMockClient({ id: `simple_node_${i}` });
      await client.connect();
      clients.push(client);
      
      // Small delay between connections
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    console.log('\nAll clients connected successfully');
    
    // Have the first client get the list of nodes
    const firstClient = clients[0];
    const nodes = await firstClient.getNodes();
    
    console.log('\nConnected nodes:', nodes.length);
    nodes.forEach((node, index) => {
      console.log(`Node ${index + 1}: ${node.id} (${node.type})`);
    });
    
    // Send a broadcast message from the first client
    firstClient.broadcast({
      type: 'test',
      content: 'Hello from the first client!'
    });
    
    console.log('\nBroadcast message sent');
    
    // Wait a bit to allow messages to be received
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    console.log('\n=== Test completed successfully ===');
    return true;
    
  } catch (error) {
    console.error('Test error:', error);
    return false;
  } finally {
    // Disconnect all clients
    for (const client of clients) {
      if (client.socket) {
        client.disconnect();
      }
    }
    console.log('All clients disconnected');
  }
}

// If this script is run directly, execute the test
if (import.meta.url === `file://${process.argv[1]}`) {
  runTest()
    .then(success => {
      process.exit(success ? 0 : 1);
    })
    .catch(error => {
      console.error('Unhandled error:', error);
      process.exit(1);
    });
}

export { runTest, SimpleMockClient }; 