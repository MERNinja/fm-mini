#!/usr/bin/env node
/**
 * Direct test script for troubleshooting
 * Runs directly without spawning separate processes
 */
import { io } from 'socket.io-client';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';

// Get __dirname in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '../..');

// Port for the test server
const PORT = 8123;

// Start a simple server
function startServer() {
  const app = express();
  const server = createServer(app);
  const io = new Server(server, {
    cors: {
      origin: '*',
      methods: ['GET', 'POST']
    }
  });
  
  // Store connected nodes
  const nodes = {};
  
  io.on('connection', (socket) => {
    console.log('New client connected:', socket.id);
    
    // Register a new node
    socket.on('register_node', (nodeData) => {
      console.log('Node registered:', nodeData);
      
      // Add connection timestamp
      const nodeInfo = {
        ...nodeData,
        socketId: socket.id,
        connectedAt: new Date().toISOString()
      };
      
      // Store node info
      nodes[nodeData.id] = nodeInfo;
      
      // Notify all clients about the new node
      io.emit('node_registered', nodeInfo);
    });
    
    // Get all nodes
    socket.on('get_nodes', (callback) => {
      callback(Object.values(nodes));
    });
    
    // Handle messages between nodes
    socket.on('message', (message) => {
      console.log('Message received:', message);
      socket.broadcast.emit('message', {
        ...message,
        socketId: message.socketId || socket.id
      });
    });
    
    // Handle disconnection
    socket.on('disconnect', () => {
      console.log('Client disconnected:', socket.id);
      
      // Find and remove the disconnected node
      const nodeId = Object.keys(nodes).find(id => nodes[id].socketId === socket.id);
      
      if (nodeId) {
        console.log('Node disconnected:', nodeId);
        delete nodes[nodeId];
        
        // Notify all clients about the disconnection
        io.emit('node_disconnected', nodeId);
      }
    });
  });
  
  // Start the server
  server.listen(PORT, () => {
    console.log(`Test server listening on port ${PORT}`);
  });
  
  return server;
}

// Basic client for testing
class TestClient {
  constructor(id) {
    this.id = id;
    this.socket = null;
    this.connectedNodes = [];
  }
  
  async connect() {
    return new Promise((resolve, reject) => {
      try {
        console.log(`[${this.id}] Connecting to server...`);
        this.socket = io(`http://localhost:${PORT}`);
        
        this.socket.on('connect', () => {
          console.log(`[${this.id}] Connected to server with socket ID: ${this.socket.id}`);
          
          // Register with the server
          this.socket.emit('register_node', {
            id: this.id,
            type: 'test',
            capabilities: ['test']
          });
          
          resolve();
        });
        
        this.socket.on('connect_error', (error) => {
          console.error(`[${this.id}] Connection error:`, error);
          reject(error);
        });
        
        // Handle node registration and disconnection
        this.socket.on('node_registered', (node) => {
          if (node.id !== this.id) {
            console.log(`[${this.id}] New node registered: ${node.id}`);
            this.connectedNodes.push(node);
          }
        });
        
        this.socket.on('node_disconnected', (nodeId) => {
          console.log(`[${this.id}] Node disconnected: ${nodeId}`);
          this.connectedNodes = this.connectedNodes.filter(node => node.id !== nodeId);
        });
        
        // Handle messages
        this.socket.on('message', (message) => {
          console.log(`[${this.id}] Received message:`, message);
        });
      } catch (error) {
        console.error(`[${this.id}] Error:`, error);
        reject(error);
      }
    });
  }
  
  async getNodes() {
    return new Promise((resolve) => {
      this.socket.emit('get_nodes', (nodes) => {
        resolve(nodes || []);
      });
    });
  }
  
  sendMessage(message) {
    this.socket.emit('message', {
      from: this.id,
      content: message,
      timestamp: new Date().toISOString()
    });
  }
  
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      console.log(`[${this.id}] Disconnected from server`);
    }
  }
}

// Main test function
async function runTest() {
  let server;
  const clients = [];
  
  try {
    // Start the server
    console.log('Starting test server...');
    server = startServer();
    
    // Wait for server to start
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Create and connect clients
    console.log('Creating test clients...');
    
    for (let i = 1; i <= 3; i++) {
      const client = new TestClient(`client_${i}`);
      await client.connect();
      clients.push(client);
      
      // Small delay between connections
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    // Verify all clients are connected
    const nodes = await clients[0].getNodes();
    console.log(`Connected nodes: ${nodes.length}`);
    
    // Send a test message
    console.log('Sending test message...');
    clients[0].sendMessage('Hello from client 1!');
    
    // Wait for message processing
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    console.log('Test completed successfully');
    return true;
  } catch (error) {
    console.error('Test error:', error);
    return false;
  } finally {
    // Clean up
    for (const client of clients) {
      client.disconnect();
    }
    
    if (server) {
      server.close(() => {
        console.log('Server closed');
      });
    }
  }
}

// Run the test
if (import.meta.url === `file://${process.argv[1]}`) {
  runTest()
    .then(success => {
      // Allow time for cleanup before exiting
      setTimeout(() => {
        process.exit(success ? 0 : 1);
      }, 1000);
    })
    .catch(error => {
      console.error('Unhandled error:', error);
      process.exit(1);
    });
} 