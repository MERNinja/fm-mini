/**
 * TensorParallel - Utility for tensor-level model parallelism across WebRTC peers
 */

// Tensor operations helper functions
export class TensorOps {
  /**
   * Splits a tensor across its first dimension
   * @param {Float32Array} tensor The tensor data
   * @param {Array<number>} shape The tensor shape
   * @param {number} numPartitions Number of partitions to split into
   * @returns {Array<{data: Float32Array, shape: Array<number>}>} Array of tensor partitions
   */
  static splitTensor(tensor, shape, numPartitions) {
    if (!tensor || !shape || shape.length === 0) {
      throw new Error('Invalid tensor or shape');
    }

    const totalElements = tensor.length;
    const firstDimSize = shape[0];
    
    // Calculate partition size (round up to ensure all elements are covered)
    const partitionSize = Math.ceil(firstDimSize / numPartitions);
    const result = [];

    for (let i = 0; i < numPartitions; i++) {
      const start = i * partitionSize;
      const end = Math.min(start + partitionSize, firstDimSize);
      
      if (start >= firstDimSize) break; // No more elements to process
      
      // Calculate elements in this partition
      const newShape = [...shape];
      newShape[0] = end - start;
      
      // Calculate starting and ending indices in the flattened array
      const elementsPerFirstDim = totalElements / firstDimSize;
      const startIdx = Math.floor(start * elementsPerFirstDim);
      const endIdx = Math.floor(end * elementsPerFirstDim);
      
      // Create partition
      const partitionData = tensor.slice(startIdx, endIdx);
      result.push({
        data: partitionData,
        shape: newShape
      });
    }

    return result;
  }

  /**
   * Merges tensor partitions back into a single tensor
   * @param {Array<{data: Float32Array, shape: Array<number>}>} partitions Array of tensor partitions
   * @param {Array<number>} originalShape Original tensor shape
   * @returns {{data: Float32Array, shape: Array<number>}} Merged tensor
   */
  static mergeTensorPartitions(partitions, originalShape) {
    if (!partitions || partitions.length === 0 || !originalShape) {
      throw new Error('Invalid partitions or original shape');
    }

    // Calculate total elements in the merged tensor
    const totalElements = partitions.reduce((sum, part) => sum + part.data.length, 0);
    const result = new Float32Array(totalElements);
    
    let offset = 0;
    for (const partition of partitions) {
      result.set(partition.data, offset);
      offset += partition.data.length;
    }

    return {
      data: result,
      shape: originalShape
    };
  }

  /**
   * Applies an operation to a tensor
   * @param {Float32Array} tensor The tensor data
   * @param {string} operation Operation to apply ('matmul', 'add', etc.)
   * @param {Float32Array} operand Second operand for binary operations
   * @param {Object} params Additional parameters for the operation
   * @returns {Float32Array} Result tensor
   */
  static applyOperation(tensor, operation, operand = null, params = {}) {
    // Simple placeholder implementation
    // In a real implementation, we would dispatch to proper tensor operations
    
    switch (operation) {
      case 'relu':
        return TensorOps.relu(tensor);
      case 'add':
        return TensorOps.elementWiseAdd(tensor, operand);
      // Add more operations as needed
      default:
        throw new Error(`Unsupported operation: ${operation}`);
    }
  }

  /**
   * Simple ReLU activation
   */
  static relu(tensor) {
    const result = new Float32Array(tensor.length);
    for (let i = 0; i < tensor.length; i++) {
      result[i] = Math.max(0, tensor[i]);
    }
    return result;
  }

  /**
   * Element-wise addition
   */
  static elementWiseAdd(tensor1, tensor2) {
    if (tensor1.length !== tensor2.length) {
      throw new Error('Tensors must have the same size for element-wise addition');
    }
    const result = new Float32Array(tensor1.length);
    for (let i = 0; i < tensor1.length; i++) {
      result[i] = tensor1[i] + tensor2[i];
    }
    return result;
  }
}

// Connection management for tensor parallelism
export class TensorParallelManager {
  constructor() {
    this.peers = new Map(); // Map of peer ID to RTCPeerConnection
    this.dataChannels = new Map(); // Map of peer ID to RTCDataChannel
    this.registeredModels = new Map(); // Map of model ID to model metadata
    this.activeTasks = new Map(); // Map of task ID to task metadata
    this.taskCallbacks = new Map(); // Map of task ID to result callback
    this.operationCallbacks = new Map(); // Map of (peerId+operation) to callback
    this.iceServers = [
      { urls: 'stun:stun.l.google.com:19302' },
      { urls: 'stun:stun1.l.google.com:19302' }
    ];
    this.selfId = 'node_' + Math.random().toString(36).substring(2, 9);
    this.socket = null; // Socket.io instance
    this.connectedPeers = new Set(); // Set of connected peer IDs - EXPLICITLY initialize as a Set
  }

  /**
   * Initialize the tensor parallel manager with socket.io connection
   * @param {Object} socket Optional socket.io instance to use
   */
  async init(socket = null) {
    return new Promise((resolve) => {
      try {
        console.log('Initializing tensor parallel manager');
        
        // Use provided socket or create new one
        if (socket) {
          this.socket = socket;
          console.log(`Using provided socket with ID: ${this.socket.id}`);
        } else {
          // Create the socket.io connection
          this.socket = io();
        }
        
        // Register socket events
        this.socket.on('connect', () => {
          console.log(`Socket connected with ID: ${this.socket.id}`);
          
          // Register this node with the signaling server
          this.socket.emit('register_node', {
            id: this.selfId,
            model: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
            ip: 'localhost:3000',
            status: 'online'
          });
          
          // Let the server know we're ready for tensor parallelism
          this.socket.emit('node_activity', {
            nodeId: this.selfId,
            action: 'tensor_parallel_initialized',
            prompt: `Node ${this.selfId} is initialized and ready for tensor parallelism. Requesting node list...`,
            timestamp: new Date().toISOString()
          });
          
          // Request the list of available nodes
          this.socket.emit('get_nodes', (nodes) => {
            // CRITICAL: Call the peer discovery handler
            this.handlePeerDiscovery(nodes);
            
            // Debug log the connected peers
            console.log('[DEBUG] Connected peers after initialization:', 
              Array.from(this.connectedPeers).map(id => `${id}`).join(', ') || 'none');
              
            resolve(true);
          });
        });
        
        // Handle disconnection
        this.socket.on('disconnect', (reason) => {
          console.log(`Socket disconnected: ${reason}`);
        });
        
        // Handle reconnection
        this.socket.on('reconnect', () => {
          console.log('Socket reconnected');
        });
        
        // Handle node registration events (notifies when new nodes appear)
        this.socket.on('node_registered', (node) => {
          console.log('New node registered:', node);
          
          if (node.id !== this.selfId) {
            console.log(`New peer node discovered: ${node.id}`);
            
            // IMPORTANT: Always add new peers directly
            this.addDirectPeer(node.id);
            
            // Log the discovery
            this.socket.emit('node_activity', {
              nodeId: this.selfId,
              socketId: this.socket?.id,
              action: 'peer_discovered',
              prompt: `New peer node discovered: ${node.id}`,
              timestamp: new Date().toISOString()
            });
          }
        });
        
        // Handle incoming operations
        this.socket.on('operation', (data) => {
          this.handleOperation(data);
        });
      } catch (error) {
        console.error('Error initializing tensor parallel manager:', error);
        resolve(false);
      }
    });
  }

  /**
   * Set the node ID for this instance
   * @param {string} nodeId The node ID
   */
  setNodeId(nodeId) {
    this.selfId = nodeId;
    return this;
  }

  /**
   * Register a model for tensor parallelism
   * @param {string} modelId The model ID
   * @param {Object} modelInfo Model metadata
   */
  registerModel(modelId, modelInfo) {
    this.registeredModels.set(modelId, {
      ...modelInfo,
      registeredAt: new Date(),
      status: 'available'
    });
    
    // Notify the server about the model registration
    if (this.socket) {
      this.socket.emit('register_tensor_model', {
        nodeId: this.selfId,
        modelId,
        modelInfo
      });
    }
    
    return this;
  }

  /**
   * Create a peer connection to another node
   * @param {string} peerId The peer node ID
   */
  async createPeerConnection(peerId) {
    if (this.peers.has(peerId)) return this.peers.get(peerId);
    
    const peerConnection = new RTCPeerConnection({ 
      iceServers: this.iceServers 
    });
    
    // Create data channel for binary tensor transfers
    const dataChannel = peerConnection.createDataChannel('tensors', {
      ordered: true,
      maxRetransmits: 3,
      // Use binary mode for tensor data
      binaryType: 'arraybuffer'
    });
    
    dataChannel.onopen = () => {
      console.log(`Data channel to peer ${peerId} opened`);
    };
    
    dataChannel.onclose = () => {
      console.log(`Data channel to peer ${peerId} closed`);
    };
    
    dataChannel.onmessage = (event) => {
      this.handleDataChannelMessage(peerId, event.data);
    };
    
    this.dataChannels.set(peerId, dataChannel);
    
    // Handle ICE candidates
    peerConnection.onicecandidate = (event) => {
      if (event.candidate) {
        this.socket.emit('tensor_signal', {
          from: this.selfId,
          to: peerId,
          type: 'candidate',
          candidate: event.candidate
        });
      }
    };
    
    // Handle incoming data channels
    peerConnection.ondatachannel = (event) => {
      const receivedChannel = event.channel;
      receivedChannel.binaryType = 'arraybuffer';
      
      receivedChannel.onmessage = (msgEvent) => {
        this.handleDataChannelMessage(peerId, msgEvent.data);
      };
      
      this.dataChannels.set(peerId, receivedChannel);
    };
    
    this.peers.set(peerId, peerConnection);
    return peerConnection;
  }

  /**
   * Initiate connection to a peer
   * @param {string} peerId The peer node ID
   */
  async connectToPeer(peerId) {
    try {
      console.log(`Attempting to connect to peer: ${peerId} via WebRTC or socket fallback`);
      
      // Don't try to connect to self
      if (peerId === this.selfId) {
        console.log(`Skipping connection to self (${peerId})`);
        return false;
      }
      
      // Create a WebRTC peer connection
      const peerConnection = await this.createPeerConnection(peerId);
      
      // Always log to socket to see connection attempts in UI
      if (this.socket) {
        this.socket.emit('node_activity', {
          nodeId: this.selfId,
          socketId: this.socket?.id,
          action: 'connection_attempt',
          prompt: `Attempting connection to peer ${peerId}`,
          timestamp: new Date().toISOString()
        });
      }
      
      // Create an offer with a timeout in case signaling gets stuck
      const offerPromise = new Promise(async (resolve) => {
        try {
          // Create an offer
          const offer = await peerConnection.createOffer();
          await peerConnection.setLocalDescription(offer);
          
          // Send the offer to the peer through the signaling server
          this.socket.emit('tensor_signal', {
            from: this.selfId,
            to: peerId,
            type: 'offer',
            sdp: peerConnection.localDescription
          });
          
          resolve(true);
        } catch (error) {
          console.warn(`Error creating offer for ${peerId}:`, error);
          resolve(false);
        }
      });
      
      // Add a timeout for the connection attempt
      const timeoutPromise = new Promise(resolve => {
        setTimeout(() => {
          console.log(`Connection attempt to ${peerId} timed out after 3 seconds, using socket fallback.`);
          resolve(false);
        }, 3000);
      });
      
      // Race the offer against the timeout
      const success = await Promise.race([offerPromise, timeoutPromise]);
      
      // Even if WebRTC fails, we can still use socket.io as fallback
      if (!success) {
        console.log(`Using socket.io fallback for peer ${peerId}.`);
        
        // Log fallback in the UI
        if (this.socket) {
          this.socket.emit('node_activity', {
            nodeId: this.selfId,
            socketId: this.socket?.id,
            action: 'connection_fallback',
            prompt: `Using socket.io fallback for peer ${peerId}`,
            timestamp: new Date().toISOString()
          });
        }
      } else {
        console.log(`Successfully initiated WebRTC connection to peer ${peerId}.`);
        
        // Log success in the UI
        if (this.socket) {
          this.socket.emit('node_activity', {
            nodeId: this.selfId,
            socketId: this.socket?.id,
            action: 'connection_success',
            prompt: `Successfully initiated connection to peer ${peerId}`,
            timestamp: new Date().toISOString()
          });
        }
      }
      
      return true; // Return true regardless because we'll use socket.io fallback
    } catch (error) {
      console.error(`Error in connectToPeer for ${peerId}:`, error);
      
      // Log error in the UI
      if (this.socket) {
        this.socket.emit('node_activity', {
          nodeId: this.selfId,
          socketId: this.socket?.id,
          action: 'connection_error',
          prompt: `Error connecting to peer ${peerId}: ${error.message}`,
          timestamp: new Date().toISOString()
        });
      }
      
      // Still return true because we'll use socket fallback
      return true;
    }
  }

  /**
   * Handle incoming data channel messages
   * @param {string} peerId The peer node ID
   * @param {ArrayBuffer} data The message data
   */
  async handleDataChannelMessage(peerId, data) {
    try {
      // Parse the metadata header from the first bytes
      const headerView = new DataView(data, 0, 16);
      const taskId = headerView.getUint32(0);
      const operationType = headerView.getUint8(4);
      const resultFlag = headerView.getUint8(5);
      
      // Extract the tensor data from the rest of the message
      const tensorData = new Float32Array(data, 16);
      
      if (resultFlag === 1) {
        // This is a result from a previously sent task
        const callback = this.taskCallbacks.get(taskId);
        if (callback) {
          callback(tensorData);
          this.taskCallbacks.delete(taskId);
        }
      } else {
        // This is a new task to process
        const result = await this.processTensorOperation(
          taskId,
          operationType,
          tensorData,
          {} // Parameters would need to be included in the header or a separate message
        );
        
        // Send the result back
        this.sendTensorData(peerId, result, taskId, operationType, true);
      }
    } catch (error) {
      console.error('Error handling data channel message:', error);
    }
  }

  /**
   * Process a tensor operation
   * @param {number} taskId The task ID
   * @param {number} operationType The operation type
   * @param {Float32Array} tensorData The tensor data
   * @param {Object} params Additional parameters
   * @returns {Promise<Float32Array>} The result tensor
   */
  async processTensorOperation(taskId, operationType, tensorData, params = {}) {
    // For now, just return the tensor as-is
    // In a real implementation, we would apply the actual operation
    return tensorData;
  }

  /**
   * Send tensor data to a peer
   * @param {string} peerId The peer node ID
   * @param {Float32Array} tensorData The tensor data
   * @param {number} taskId The task ID
   * @param {number} operationType The operation type
   * @param {boolean} isResult Whether this is a result (true) or a task (false)
   */
  sendTensorData(peerId, tensorData, taskId, operationType, isResult = false) {
    if (!this.dataChannels.has(peerId)) {
      throw new Error(`No data channel established with peer ${peerId}`);
    }
    
    const dataChannel = this.dataChannels.get(peerId);
    if (dataChannel.readyState !== 'open') {
      throw new Error(`Data channel to peer ${peerId} is not open`);
    }
    
    // Create a buffer with header (16 bytes) + tensor data
    const buffer = new ArrayBuffer(16 + tensorData.byteLength);
    const headerView = new DataView(buffer, 0, 16);
    
    // Write header information
    headerView.setUint32(0, taskId);
    headerView.setUint8(4, operationType);
    headerView.setUint8(5, isResult ? 1 : 0);
    
    // Copy tensor data
    new Float32Array(buffer, 16).set(tensorData);
    
    // Send the buffer
    dataChannel.send(buffer);
  }

  /**
   * Split a model computation across multiple nodes
   * @param {Object} modelContext The WebLLM model context
   * @param {Array<string>} peerIds Array of peer node IDs
   * @param {Object} inputTensors Input tensor data
   * @returns {Promise<Object>} The computation result
   */
  async executeParallelComputation(modelContext, peerIds, inputTensors) {
    if (!peerIds || peerIds.length === 0) {
      throw new Error('No peers specified for parallel computation');
    }
    
    const taskId = Date.now(); // Simple task ID generation
    this.activeTasks.set(taskId, {
      startTime: Date.now(),
      peers: peerIds,
      status: 'in_progress'
    });
    
    try {
      // For demonstration, split the first input tensor across peers
      const firstTensorName = Object.keys(inputTensors)[0];
      const tensor = inputTensors[firstTensorName];
      
      // For simplicity, assuming tensor has shape information
      const tensorPartitions = TensorOps.splitTensor(
        tensor.data,
        tensor.shape,
        peerIds.length
      );
      
      // Send each partition to a peer
      const partitionResults = await Promise.all(
        peerIds.map((peerId, index) => {
          return new Promise((resolve) => {
            const partitionTaskId = taskId + index;
            
            // Register callback for when the result comes back
            this.taskCallbacks.set(partitionTaskId, (result) => {
              resolve(result);
            });
            
            // Send the partition to the peer
            this.sendTensorData(
              peerId,
              tensorPartitions[index].data,
              partitionTaskId,
              1, // Assuming operation type 1 = process partition
              false
            );
          });
        })
      );
      
      // Merge the results
      const mergedResult = TensorOps.mergeTensorPartitions(
        partitionResults.map((data, index) => ({
          data,
          shape: tensorPartitions[index].shape
        })),
        tensor.shape
      );
      
      this.activeTasks.set(taskId, {
        ...this.activeTasks.get(taskId),
        status: 'completed',
        endTime: Date.now()
      });
      
      return {
        [firstTensorName]: mergedResult
      };
    } catch (error) {
      this.activeTasks.set(taskId, {
        ...this.activeTasks.get(taskId),
        status: 'failed',
        error: error.message,
        endTime: Date.now()
      });
      
      throw error;
    }
  }

  /**
   * Clean up resources
   */
  cleanup() {
    // Close all peer connections
    for (const [peerId, connection] of this.peers.entries()) {
      connection.close();
    }
    
    // Clear all data structures
    this.peers.clear();
    this.dataChannels.clear();
    this.registeredModels.clear();
    this.activeTasks.clear();
    this.taskCallbacks.clear();
    this.operationCallbacks.clear();
  }

  /**
   * Register a callback for a specific operation from a peer
   * @param {string} peerId The peer ID
   * @param {string} operation The operation name
   * @param {Function} callback The callback function
   */
  registerTaskCallback(peerId, operation, callback) {
    const key = `${peerId}:${operation}`;
    this.operationCallbacks.set(key, callback);
  }

  /**
   * Send an operation to another node
   * @param {string} to Target node ID
   * @param {string} taskId Task ID
   * @param {string} operation Operation name
   * @param {Object} data Operation data
   * @returns {Promise} Promise that resolves when operation is complete
   */
  sendOperation(to, taskId, operation, data) {
    return new Promise((resolve, reject) => {
      if (!this.socket) {
        console.error('Cannot send operation: socket is not initialized');
        reject(new Error('Socket not initialized'));
        return;
      }
      
      console.log(`📤 Sending operation ${operation} to ${to} with task ID ${taskId}`);
      
      // Set up a listener for the result
      const resultHandler = (result) => {
        if (result.taskId === taskId && result.from === to) {
          console.log(`📥 Received result from ${to} for task ${taskId}`);
          this.socket.off('operation_result', resultHandler);
          resolve(result.result);
        }
      };
      
      // Listen for operation results
      this.socket.on('operation_result', resultHandler);
      
      // Set a timeout to avoid hanging forever
      const timeout = setTimeout(() => {
        this.socket.off('operation_result', resultHandler);
        console.warn(`⚠️ Operation ${operation} to ${to} timed out after 10 seconds`);
        reject(new Error(`Operation timed out`));
      }, 10000);
      
      // Send the operation
      this.socket.emit('operation', {
        from: this.selfId,
        to,
        taskId,
        operation,
        data
      });
      
      // Log to activity feed
      this.socket.emit('node_activity', {
        nodeId: this.selfId,
        socketId: this.socket?.id,
        action: 'sending_operation',
        prompt: `Sending ${operation} operation to node ${to}`,
        timestamp: new Date().toISOString()
      });
    });
  }

  /**
   * Handle an incoming operation from a peer
   * @param {Object} data The operation data
   */
  handleOperation(data) {
    if (data.to !== this.selfId) return;
    
    const { from, taskId, operation, data: operationData } = data;
    
    console.log(`⚙️ RECEIVED OPERATION from ${from}: ${operation}`, operationData?.layers?.length || 0, 'layers');
    
    // Log the operation in socket
    if (this.socket) {
      this.socket.emit('node_activity', {
        nodeId: this.selfId,
        socketId: this.socket?.id,
        action: 'operation_received',
        prompt: `Received ${operation} operation from node ${from} with ${operationData?.layers?.length || 0} layers to process${operationData?.batchNumber ? ' for batch ' + operationData.batchNumber : ''}`,
        timestamp: new Date().toISOString()
      });
    }
    
    // Handle different operations
    if (operation === 'process_layers') {
      const batchNumber = operationData?.batchNumber || 1;
      console.log(`🔷 Processing ${operationData?.layers?.length || 0} layers for task ${taskId} batch ${batchNumber}`);
      
      // Simulate processing layers
      setTimeout(() => {
        console.log(`✅ Completed processing layers for task ${taskId} batch ${batchNumber}, sending result back to ${from}`);
        
        // Log the result in socket
        if (this.socket) {
          this.socket.emit('node_activity', {
            nodeId: this.selfId,
            socketId: this.socket?.id,
            action: 'operation_completed',
            prompt: `Completed processing ${operationData?.layers?.length || 0} layers for task ${taskId} batch ${batchNumber}`,
            timestamp: new Date().toISOString()
          });
        }
        
        // Send result back to the sender
        this.socket.emit('operation_result', {
          from: this.selfId,
          to: from,
          taskId,
          operation,
          result: {
            success: true,
            processingTime: 500,
            layers: operationData?.layers || [],
            batchNumber: batchNumber,
            partialResult: `Result from node ${this.selfId} for batch ${batchNumber}`
          }
        });
      }, 500);
    }
  }

  /**
   * Handle operation result from a peer
   * @param {Object} data The result data
   */
  handleOperationResult(data) {
    if (data.to !== this.selfId) return;
    
    const { from, operation, result } = data;
    const key = `${from}:${operation}`;
    
    // Find and call the registered callback
    if (this.operationCallbacks.has(key)) {
      const callback = this.operationCallbacks.get(key);
      callback(result);
      
      // Remove the callback after use
      this.operationCallbacks.delete(key);
    }
  }

  /**
   * Directly add a peer to connected peers regardless of WebRTC connection status
   * @param {string} peerId The peer to add
   */
  addDirectPeer(peerId) {
    // Don't try to connect to self
    if (peerId === this.selfId) {
      return false;
    }
    
    // Don't add duplicates - check if we already have this peer
    if (this.connectedPeers && this.connectedPeers.has(peerId)) {
      console.log(`Peer ${peerId} already in connected peers list, skipping add operation`);
      return false;
    }
    
    console.log(`📡 Directly adding peer: ${peerId} to ensure connection`);
    
    // Ensure connectedPeers is initialized if it wasn't already
    if (!this.connectedPeers) {
      this.connectedPeers = new Set();
    }
    
    // Add the peer to our set
    this.connectedPeers.add(peerId);
    
    console.log(`Current connected peers after adding ${peerId}:`, Array.from(this.connectedPeers));
    
    // Log activity to socket for UI
    if (this.socket) {
      this.socket.emit('node_activity', {
        nodeId: this.selfId,
        socketId: this.socket?.id,
        action: 'peer_connected',
        prompt: `Force-added peer node: ${peerId}`,
        timestamp: new Date().toISOString()
      });
    }
    
    return true;
  }

  /**
   * Reset the connected peers list
   */
  resetConnectedPeers() {
    this.connectedPeers = new Set();
    console.log('Reset connected peers list');
    return this;
  }

  /**
   * Handle the peer discovery event triggered when we receive the node list
   * @param {Array} nodes List of nodes in the network
   */
  handlePeerDiscovery(nodes) {
    if (!nodes || !Array.isArray(nodes)) {
      console.log('No nodes received or invalid data');
      return;
    }
    
    // Filter out self
    const peerNodes = nodes.filter(n => n.id !== this.selfId);
    console.log(`Discovered ${peerNodes.length} peer nodes: ${peerNodes.map(n => n.id).join(', ')}`);
    
    // Reset the connected peers list first to avoid duplicates
    this.resetConnectedPeers();
    
    // IMPORTANT FIX: Add each peer exactly once
    for (const peer of peerNodes) {
      this.addDirectPeer(peer.id);
    }
    
    // Attempt WebRTC connections for better performance, but don't rely on them
    for (const peer of peerNodes) {
      this.connectToPeer(peer.id).catch(error => {
        console.warn(`WebRTC connection to ${peer.id} failed, falling back to signaling server: ${error.message}`);
      });
    }
  }
}

export default new TensorParallelManager(); 