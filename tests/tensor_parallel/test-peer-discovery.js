/**
 * Custom test script for peer discovery issues
 * Tests why "Detected 0 peer nodes for tensor parallelism" occurs
 */
import { io } from 'socket.io-client';

// Constants
const SERVER_URL = 'http://localhost:8080';
const NUM_NODES = 3;
const TEST_TIMEOUT = 30000; // 30 seconds
const DELAY_BETWEEN_NODES = 1000; // 1 second

/**
 * Simple client for testing tensor parallelism
 */
class SimpleClient {
  constructor(id) {
    this.id = id;
    this.socket = null;
    this.connectedPeers = new Set();
    this.discoveredNodes = new Set();
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
      this.discoveredNodes.add(node.id);
    });
    
    // Disconnect
    this.socket.on('disconnect', (reason) => {
      console.log(`[${this.id}] Disconnected: ${reason}`);
    });
  }
  
  async findPeers() {
    return new Promise((resolve) => {
      this.socket.emit('get_nodes', (nodes) => {
        console.log(`[${this.id}] Received node list:`, nodes?.map(n => n.id) || []);
        
        if (!nodes || !Array.isArray(nodes)) {
          console.log(`[${this.id}] No nodes received or invalid data`);
          resolve([]);
          return;
        }
        
        // Filter out self
        const peers = nodes.filter(n => n.id !== this.id);
        console.log(`[${this.id}] Peers available: ${peers.length}`);
        
        // Connect to peers
        for (const peer of peers) {
          this.connectedPeers.add(peer.id);
          console.log(`[${this.id}] Connected to peer: ${peer.id}`);
          
          // Log activity
          this.socket.emit('node_activity', {
            nodeId: this.id,
            socketId: this.socket.id,
            action: 'peer_connected',
            prompt: `Connected to peer: ${peer.id}`,
            timestamp: new Date().toISOString()
          });
        }
        
        resolve(Array.from(this.connectedPeers));
      });
    });
  }
  
  checkParallelInference() {
    const peers = Array.from(this.connectedPeers);
    
    console.log(`[${this.id}] Detected ${peers.length} peer nodes for tensor parallelism: ${peers.join(', ')}`);
    
    // Log to server
    this.socket.emit('node_activity', {
      nodeId: this.id,
      socketId: this.socket.id,
      action: 'parallel_discovery',
      prompt: `Detected ${peers.length} peer nodes for tensor parallelism${peers.length > 0 ? ': ' + peers.join(', ') : ''}`,
      timestamp: new Date().toISOString()
    });
    
    // Log node counts
    console.log(`[${this.id}] Discovered nodes: ${this.discoveredNodes.size} vs Connected peers: ${this.connectedPeers.size}`);
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
  const clients = [];
  
  try {
    console.log("=== Starting Peer Discovery Test ===");
    
    // Create and connect clients
    for (let i = 0; i < NUM_NODES; i++) {
      const id = `test_node_${i+1}`;
      console.log(`Creating client ${id}...`);
      
      const client = new SimpleClient(id);
      await client.connect();
      clients.push(client);
      
      // Wait between client connections
      if (i < NUM_NODES - 1) {
        await new Promise(resolve => setTimeout(resolve, DELAY_BETWEEN_NODES));
      }
    }
    
    console.log("\nAll clients connected successfully");
    console.log(`Total clients: ${clients.length}`);
    
    // Let the first client discover and connect to peers
    console.log("\nFinding peers...");
    const leader = clients[0];
    const connectedPeers = await leader.findPeers();
    
    console.log(`\nFound ${connectedPeers.length} peers`);
    
    // Wait a moment for connections to stabilize
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Check if peers are detected for parallel inference
    console.log("\nChecking parallel inference...");
    leader.checkParallelInference();
    
    // Check if we see 0 peers in parallel discovery despite having connections
    if (leader.discoveredNodes.size > 0 && leader.connectedPeers.size === 0) {
      console.log("\n⚠️ ISSUE FOUND: Nodes are discovered but not added to connected peers!");
    } else if (leader.connectedPeers.size === 0) {
      console.log("\n⚠️ ISSUE FOUND: No peers connected at all!");
    } else {
      console.log("\n✅ Peer connections working correctly");
    }
    
  } catch (error) {
    console.error("Test error:", error);
  } finally {
    // Clean up
    console.log("\nDisconnecting clients...");
    for (const client of clients) {
      client.disconnect();
    }
    console.log("All clients disconnected");
  }
}

// Run the test
runTest(); 