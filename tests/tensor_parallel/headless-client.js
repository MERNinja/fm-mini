/**
 * Headless client for testing tensor parallelism
 * Simulates a tensor node with WebLLM capabilities
 */
import { io } from 'socket.io-client';
import path from 'path';
import fs from 'fs';
import { TensorParallelLLM } from '../../app/utils/webLlmAdapter.js';
import { StrategyType } from '../../app/utils/parallelStrategy.js';
import { createModelManager } from './model-manager.js';

/**
 * HeadlessClient - A client that can run as a tensor node without UI
 */
export class HeadlessClient {
  constructor(config = {}) {
    this.config = {
      id: `node_${Math.random().toString(36).substring(2, 9)}`,
      serverUrl: 'http://localhost:8080',
      modelId: 'tinyllama-1.1b-chat-v1.0',
      modelDir: './models', // Directory to store downloaded models
      ...config
    };
    
    this.socket = null;
    this.llmAdapter = null;
    this.llmEngine = null;
    this.modelInfo = {
      name: this.config.modelId,
      status: 'idle',
      loaded: false
    };
    
    this.connectedNodes = [];
    this.connectedPeers = [];
    this.isParallelEnabled = false;
    this.strategyType = null;
    
    this.chatHistory = [];
    this.modelManager = null;
  }
  
  /**
   * Connect to the signaling server
   */
  async connect() {
    return new Promise((resolve, reject) => {
      try {
        console.log(`[${this.config.id}] Connecting to ${this.config.serverUrl}...`);
        this.socket = io(this.config.serverUrl);
        
        this.socket.on('connect', () => {
          console.log(`[${this.config.id}] Connected with socket ID: ${this.socket.id}`);
          
          // Register node with the server
          this.socket.emit('register_node', {
            id: this.config.id,
            type: this.config.type || 'headless',
            capabilities: ['tensor_parallel'],
            status: 'online'
          });
          
          // Setup message handlers
          this.setupMessageHandlers();
          
          // Setup tensor operation handlers
          this.setupTensorOperationHandlers();
          
          this.status = 'connected';
          resolve(true);
        });
        
        this.socket.on('connect_error', (error) => {
          console.error(`[${this.config.id}] Connection error:`, error);
          reject(error);
        });
        
        // Handle incoming messages
        this.socket.on('message', (message) => {
          console.log(`[${this.config.id}] Received message:`, message);
          if (message.type === 'chat' && message.prompt) {
            this.handleChatRequest(message);
          }
        });
        
        // Handle tensor operation requests
        this.socket.on('tensor_operation_request', (data) => {
          if (data.to === this.config.id) {
            this.handleTensorOperationRequest(data);
          }
        });
        
        // Handle tensor operation results
        this.socket.on('tensor_operation_result', (data) => {
          if (data.to === this.config.id) {
            console.log(`[${this.config.id}] Received tensor operation result from ${data.from}`);
          }
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
        console.error(`[${this.config.id}] Error connecting to server:`, error);
        reject(error);
      }
    });
  }
  
  /**
   * Set up message handlers for socket events
   */
  setupMessageHandlers() {
    if (!this.socket) {
      return;
    }
    
    // Handle node registration and disconnection
    this.socket.on('node_registered', (node) => {
      console.log(`[${this.config.id}] New node registered: ${node.id}`);
    });
    
    this.socket.on('node_disconnected', (nodeId) => {
      console.log(`[${this.config.id}] Node disconnected: ${nodeId}`);
    });
    
    // Handle messages from other clients
    this.socket.on('message', (message) => {
      console.log(`[${this.config.id}] Received message from ${message.from || 'unknown'}`);
      
      // Handle different message types
      if (message.type === 'chat_request' && message.to === this.config.id) {
        this.handleChatRequest(message);
      }
    });
    
    // Handle errors
    this.socket.on('error', (error) => {
      console.error(`[${this.config.id}] Socket error:`, error);
    });
    
    this.socket.on('connect_error', (error) => {
      console.error(`[${this.config.id}] Connection error:`, error);
    });
    
    this.socket.on('disconnect', (reason) => {
      console.log(`[${this.config.id}] Disconnected: ${reason}`);
      this.status = 'disconnected';
    });
  }
  
  /**
   * Load the LLM model
   */
  async loadModel() {
    try {
      console.log(`[${this.config.id}] Loading model: ${this.config.modelId}`);
      this.modelInfo.status = 'loading';
      
      // Initialize model manager
      this.modelManager = await createModelManager({
        modelDir: this.config.modelDir,
        defaultModel: this.config.modelId
      });
      
      // Special handling for mock-test-model
      if (this.config.modelId === 'mock-test-model') {
        console.log(`[${this.config.id}] Using mock model for testing`);
        
        // Create a mock engine for testing
        this.llmEngine = this.createMockEngine();
        
        // Update model status
        this.modelInfo.loaded = true;
        this.modelInfo.status = 'ready';
        
        console.log(`[${this.config.id}] Mock model loaded successfully`);
        
        // Broadcast model availability
        this.socket.emit('status_update', {
          id: this.config.id,
          status: 'ready',
          modelInfo: this.modelInfo
        });
        
        return true;
      }
      
      // Standard model loading
      // Download model if not available
      if (!this.modelManager.isModelDownloaded(this.config.modelId)) {
        console.log(`[${this.config.id}] Model not found locally, downloading...`);
        await this.modelManager.downloadModel(this.config.modelId);
      }
      
      // Get model path
      const modelPath = this.modelManager.getModelPath(this.config.modelId);
      
      // Setup model loading options
      const options = {
        modelPath: modelPath,
        // Use local system cache
        cacheDir: this.config.modelDir
      };

      // Register progress callback
      this.llmAdapter.onLoadingProgress((report) => {
        console.log(`[${this.config.id}] Loading progress: ${report.progress.toFixed(2)}, ${report.text}`);
      });
      
      // Load the model
      this.llmEngine = await this.llmAdapter.loadModel(this.config.modelId, options);
      
      // Update model status
      this.modelInfo.loaded = true;
      this.modelInfo.status = 'ready';
      
      console.log(`[${this.config.id}] Model loaded successfully`);
      
      // Broadcast model availability
      this.socket.emit('status_update', {
        id: this.config.id,
        status: 'ready',
        modelInfo: this.modelInfo
      });
      
      return true;
    } catch (error) {
      console.error(`[${this.config.id}] Error loading model:`, error);
      this.modelInfo.status = 'error';
      this.modelInfo.error = error.message;
      
      // Broadcast error status
      this.socket.emit('status_update', {
        id: this.config.id,
        status: 'error',
        error: error.message
      });
      
      return false;
    }
  }
  
  /**
   * Create a mock engine for testing
   * @returns {Object} A mock engine object
   */
  createMockEngine() {
    return {
      // Add tensor parallelism specific methods
      tensorParallel: {
        // Enable tensor parallelism
        enable: async (nodes) => {
          console.log(`[${this.config.id}] Mock enabling tensor parallelism with ${nodes.length} nodes`);
          this.isParallelEnabled = true;
          
          // Store node IDs in connectedPeers for reference
          this.connectedPeers = nodes.map(node => node.id);
          
          // Display connected nodes
          console.log(`[${this.config.id}] Connected to peers: ${this.connectedPeers.join(', ')}`);
          
          // Send a notification to the peers 
          for (const peerId of this.connectedPeers) {
            this.socket.emit('tensor_operation_request', {
              from: this.config.id,
              to: peerId,
              type: 'connection',
              operation: 'handshake',
              timestamp: new Date().toISOString()
            });
          }
          
          return true;
        },
        
        // Disable tensor parallelism
        disable: () => {
          console.log(`[${this.config.id}] Mock disabling tensor parallelism`);
          
          // Notify peers about tensor shutdown
          for (const peerId of this.connectedPeers) {
            this.socket.emit('tensor_operation_request', {
              from: this.config.id,
              to: peerId,
              type: 'connection',
              operation: 'shutdown',
              timestamp: new Date().toISOString()
            });
          }
          
          this.isParallelEnabled = false;
          this.connectedPeers = [];
          return true;
        },
        
        // Get status info
        getStatus: () => ({
          enabled: this.isParallelEnabled,
          connectedPeers: this.connectedPeers,
          partitioning: this.generateMockPartitioning(),
          strategy: this.strategyType || 'mock_strategy'
        }),
        
        // Set parallelism strategy
        setStrategy: (strategyType) => {
          console.log(`[${this.config.id}] Mock setting strategy to ${strategyType}`);
          this.strategyType = strategyType;
          return true;
        },
        
        // Get available strategy types
        getAvailableStrategies: () => ['layer_parallel', 'tensor_parallel'],
        
        // Get performance metrics for the current strategy
        getPerformanceMetrics: () => ({
          transitions: this.connectedPeers.length + 1,
          dataPerTransition: 1024,
          totalDataTransferred: 1024 * (this.connectedPeers.length + 1),
          totalDataBytes: 4096 * (this.connectedPeers.length + 1),
          totalDataMB: (4096 * (this.connectedPeers.length + 1)) / (1024 * 1024)
        })
      },
      
      // Mock chat completions
      chat: {
        completions: {
          create: async (params) => {
            console.log(`[${this.config.id}] Received prompt: ${params.messages[0].content.substring(0, 50)}...`);
            
            // Check if tensor parallelism is enabled
            if (this.isParallelEnabled && this.connectedPeers.length > 0) {
              console.log(`[${this.config.id}] Using tensor parallelism with ${this.connectedPeers.length} peers`);
              
              // Partition the work - in a real system, this would involve analyzing the layers
              // and distributing tensor operations
              const partitioning = this.generateMockPartitioning();
              console.log(`[${this.config.id}] Model partitioning: `, JSON.stringify(partitioning));
              
              // Immediately send tensor operation requests to peers
              await this.distributeTensorOperations(params.messages[0].content);
              
              // Simulate processing with tensor parallelism
              console.log(`[${this.config.id}] Coordinating distributed inference...`);
              await new Promise(resolve => setTimeout(resolve, 1500));
              
              // Return mock completion with tensor parallelism information
              return {
                choices: [
                  {
                    message: {
                      role: 'assistant',
                      content: `This response was generated using tensor parallelism across ${this.connectedPeers.length + 1} nodes: ${[this.config.id, ...this.connectedPeers].join(', ')}. Each node processed different parts of the model.`
                    },
                    finish_reason: 'stop'
                  }
                ],
                usage: {
                  prompt_tokens: params.messages[0].content.length / 4,
                  completion_tokens: 50,
                  total_tokens: params.messages[0].content.length / 4 + 50
                }
              };
            } else {
              // Without tensor parallelism, process locally
              console.log(`[${this.config.id}] Processing prompt locally (no tensor parallelism)`);
              
              // Simulate local processing
              await new Promise(resolve => setTimeout(resolve, 3000));
              
              // Return mock completion
              return {
                choices: [
                  {
                    message: {
                      role: 'assistant',
                      content: `This is a mock response from ${this.config.id} using a single node (no tensor parallelism).`
                    },
                    finish_reason: 'stop'
                  }
                ],
                usage: {
                  prompt_tokens: params.messages[0].content.length / 4,
                  completion_tokens: 20,
                  total_tokens: params.messages[0].content.length / 4 + 20
                }
              };
            }
          }
        }
      }
    };
  }
  
  /**
   * Generate a mock partitioning for tensor parallelism demonstration
   */
  generateMockPartitioning() {
    const partitioning = {};
    
    // Add self for some layers
    partitioning[`input_embedding`] = this.config.id;
    partitioning[`output_layer`] = this.config.id;
    
    // Distribute transformer blocks among peers
    if (this.connectedPeers.length > 0) {
      const numLayers = 12; // Mock 12 transformer layers
      let nodeIndex = 0;
      const allNodes = [this.config.id, ...this.connectedPeers];
      
      for (let i = 0; i < numLayers; i++) {
        // Distribute layers in round-robin fashion
        const assignedNode = allNodes[nodeIndex % allNodes.length];
        partitioning[`transformer_block_${i}`] = assignedNode;
        nodeIndex++;
      }
    }
    
    return partitioning;
  }
  
  /**
   * Distribute tensor operations to peers
   * @param {string} prompt The original prompt
   */
  async distributeTensorOperations(prompt) {
    // For each peer, send a tensor operation request based on their assigned layers
    const partitioning = this.generateMockPartitioning();
    
    const peerOperations = new Map();
    
    // Identify operations for each peer
    for (const [layer, nodeId] of Object.entries(partitioning)) {
      if (nodeId !== this.config.id) {
        if (!peerOperations.has(nodeId)) {
          peerOperations.set(nodeId, []);
        }
        peerOperations.get(nodeId).push(layer);
      }
    }
    
    // Send operations to each peer
    for (const [peerId, layers] of peerOperations.entries()) {
      console.log(`[${this.config.id}] Sending tensor operations for layers [${layers.join(', ')}] to peer ${peerId}`);
      
      this.socket.emit('tensor_operation_request', {
        from: this.config.id,
        to: peerId,
        type: 'inference',
        operation: 'process_layers',
        layers: layers,
        timestamp: new Date().toISOString()
      });
      
      // We're not waiting for a real response in this mock implementation
    }
    
    return true;
  }
  
  /**
   * Enable tensor parallelism with other nodes
   */
  async enableTensorParallel() {
    if (!this.llmEngine || !this.modelInfo.loaded) {
      throw new Error('Model not loaded yet');
    }
    
    // Get available nodes
    return new Promise((resolve, reject) => {
      this.socket.emit('get_nodes', null, (nodes) => {
        try {
          // Filter out self from nodes list
          const otherNodes = nodes.filter(node => node.id !== this.config.id);
          
          if (otherNodes.length === 0) {
            console.log(`[${this.config.id}] No other nodes available for tensor parallelism`);
            resolve(false);
            return;
          }
          
          console.log(`[${this.config.id}] Enabling tensor parallelism with ${otherNodes.length} other nodes`);
          
          // Enable tensor parallelism
          this.llmEngine.tensorParallel.enable(otherNodes)
            .then(success => {
              this.isParallelEnabled = success;
              
              if (success) {
                console.log(`[${this.config.id}] Tensor parallelism enabled`);
                
                // Log the partitioning strategy
                const status = this.llmEngine.tensorParallel.getStatus();
                console.log(`[${this.config.id}] Partitioning:`, status.partitioning);
              } else {
                console.log(`[${this.config.id}] Failed to enable tensor parallelism`);
              }
              
              resolve(success);
            })
            .catch(error => {
              console.error(`[${this.config.id}] Error enabling tensor parallelism:`, error);
              reject(error);
            });
        } catch (error) {
          console.error(`[${this.config.id}] Error in enableTensorParallel:`, error);
          reject(error);
        }
      });
    });
  }
  
  /**
   * Set tensor parallelism strategy
   * @param {string} strategyType Type of strategy from StrategyType
   */
  setParallelStrategy(strategyType) {
    if (!this.llmEngine || !this.modelInfo.loaded) {
      throw new Error('Model not loaded yet');
    }
    
    return this.llmEngine.tensorParallel.setStrategy(strategyType);
  }
  
  /**
   * Send a chat message to the LLM
   * @param {string} prompt The prompt to send
   */
  async sendChat(prompt) {
    if (!this.llmEngine || !this.modelInfo.loaded) {
      throw new Error('Model not loaded yet');
    }
    
    try {
      console.log(`[${this.config.id}] Processing prompt: ${prompt}`);
      
      // Add to chat history
      this.chatHistory.push({
        role: 'user',
        content: prompt,
        timestamp: new Date().toISOString()
      });
      
      // Log activity
      this.socket.emit('node_activity', {
        id: this.config.id,
        type: 'chat_request',
        prompt,
        timestamp: new Date().toISOString()
      });
      
      // Process with LLM
      const startTime = Date.now();
      const result = await this.llmEngine.chat.completions.create({
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.7,
        max_tokens: 800
      });
      const endTime = Date.now();
      
      // Add response to chat history
      this.chatHistory.push({
        role: 'assistant',
        content: result.choices[0].message.content,
        timestamp: new Date().toISOString()
      });
      
      // Log completion stats
      const stats = {
        id: this.config.id,
        type: 'chat_response',
        prompt,
        tokens: {
          input: result.usage.prompt_tokens,
          output: result.usage.completion_tokens,
          total: result.usage.total_tokens
        },
        time: {
          total_ms: endTime - startTime,
          tokens_per_second: result.usage.completion_tokens / ((endTime - startTime) / 1000)
        },
        parallelism: {
          enabled: this.isParallelEnabled,
          nodes: this.isParallelEnabled ? this.llmEngine.tensorParallel.getStatus().connectedPeers.length : 0
        },
        timestamp: new Date().toISOString()
      };
      
      console.log(`[${this.config.id}] Chat response stats:`, stats);
      this.socket.emit('node_activity', stats);
      
      return {
        response: result.choices[0].message.content,
        stats
      };
    } catch (error) {
      console.error(`[${this.config.id}] Error processing chat:`, error);
      
      // Log error
      this.socket.emit('node_activity', {
        id: this.config.id,
        type: 'error',
        error: error.message,
        timestamp: new Date().toISOString()
      });
      
      throw error;
    }
  }
  
  /**
   * Handle incoming chat requests from other clients
   */
  async handleChatRequest(message) {
    try {
      const { prompt, from } = message;
      
      console.log(`[${this.config.id}] Received chat request from ${from}: ${prompt}`);
      
      const result = await this.sendChat(prompt);
      
      // Send response back to requester
      this.socket.emit('message', {
        type: 'chat_response',
        from: this.config.id,
        to: from,
        prompt,
        response: result.response,
        stats: result.stats
      });
      
    } catch (error) {
      console.error(`[${this.config.id}] Error handling chat request:`, error);
      
      // Send error back to requester
      this.socket.emit('message', {
        type: 'error',
        from: this.config.id,
        to: message.from,
        error: error.message
      });
    }
  }
  
  /**
   * Broadcast a message to all other nodes
   */
  broadcastMessage(message) {
    this.socket.emit('message', {
      ...message,
      from: this.config.id,
      timestamp: new Date().toISOString()
    });
  }
  
  /**
   * Disconnect from the server
   */
  disconnect() {
    if (this.socket) {
      // Unregister the node
      this.socket.emit('unregister_node', { id: this.config.id });
      
      // Disconnect socket
      this.socket.disconnect();
      this.socket = null;
      
      console.log(`[${this.config.id}] Disconnected from server`);
    }
  }
  
  /**
   * Handle tensor operation requests from other nodes
   * @param {Object} data The operation request data
   */
  async handleTensorOperationRequest(data) {
    const { from, type, operation, layers } = data;
    
    console.log(`[${this.config.id}] Received tensor operation request from ${from}:`);
    console.log(`[${this.config.id}] Type: ${type}, Operation: ${operation}`);
    
    if (type === 'connection') {
      if (operation === 'handshake') {
        console.log(`[${this.config.id}] Handshake received from ${from} - ready for tensor operations`);
      } else if (operation === 'shutdown') {
        console.log(`[${this.config.id}] Tensor operations shutdown received from ${from}`);
      }
    } else if (type === 'inference') {
      if (operation === 'process_layers') {
        console.log(`[${this.config.id}] Processing layers [${layers.join(', ')}] for node ${from}`);
        
        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Send back result
        this.socket.emit('tensor_operation_result', {
          from: this.config.id,
          to: from,
          type: 'inference',
          operation: 'layers_processed',
          layers: layers,
          timestamp: new Date().toISOString()
        });
      }
    }
  }
  
  /**
   * Listen for socket events related to tensor operations
   */
  setupTensorOperationHandlers() {
    if (!this.socket) {
      return;
    }
    
    // Listen for tensor_operation events (process_layers etc.)
    this.socket.on('tensor_operation', (data) => {
      if (data.to !== this.config.id) return;
      
      const { from, taskId, operation, data: operationData } = data;
      
      console.log(`⚙️ RECEIVED OPERATION from ${from}: ${operation}`);
      
      // Process the operation
      if (operation === 'process_layers') {
        const { layers, prompt, modelId, messageHistory } = operationData;
        console.log(`⚙️ Processing layers: ${layers.join(', ')}`);
        console.log(`⚙️ Processing prompt: "${prompt}"`);
        
        // Log that we are processing a part of this prompt - other connected clients will see this
        this.socket.emit('message', {
          from: this.config.id,
          to: from, // This will ensure the message gets broadcast
          type: 'tensor_processing',
          socketId: this.socket.id,
          text: `Node ${this.config.id} is processing ${layers.length} layers for prompt: "${prompt.substring(0, 30)}${prompt.length > 30 ? '...' : ''}"`
        });
        
        // Also log to node activity
        this.socket.emit('node_activity', {
          nodeId: this.config.id,
          socketId: this.socket.id,
          action: 'processing_prompt',
          prompt: `Processing ${layers.length} layers for prompt: "${prompt.substring(0, 30)}${prompt.length > 30 ? '...' : ''}"`,
          timestamp: new Date().toISOString()
        });
        
        // Simulate processing time
        const processingTime = 1300; // milliseconds
        console.log(`⚙️ Processing will take ${processingTime}ms`);
        
        // Simulate async processing
        setTimeout(() => {
          console.log(`✅ COMPLETED OPERATION ${operation} for layers: ${layers.join(', ')}`);
          
          // Log completion to all clients
          this.socket.emit('message', {
            from: this.config.id,
            to: from,
            type: 'tensor_completed',
            socketId: this.socket.id,
            text: `Node ${this.config.id} completed processing ${layers.length} layers in ${processingTime}ms`
          });
          
          // Log that we completed our part
          this.socket.emit('node_activity', {
            nodeId: this.config.id,
            socketId: this.socket.id,
            action: 'completed_layers',
            prompt: `Completed processing ${layers.length} layers in ${processingTime}ms for prompt: "${prompt.substring(0, 30)}${prompt.length > 30 ? '...' : ''}"`,
            timestamp: new Date().toISOString()
          });
          
          // Send the result back
          this.socket.emit('tensor_operation_result', {
            from: this.config.id,
            to: from,
            taskId,
            operation,
            result: {
              processingTime,
              layers,
              partialResult: `[Partial result from ${this.config.id} - ${layers.length} layers]`
            }
          });
        }, processingTime);
      }
    });
  }
} 