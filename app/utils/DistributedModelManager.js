/**
 * Distributed Model Manager
 * Coordinates tensor-level model parallelism across multiple browser nodes
 */
import WebRTCPeerConnection from './webrtc';
import * as tensorUtils from './tensor';

class DistributedModelManager {
    constructor(socket, nodeId, modelId) {
        this.socket = socket;
        this.nodeId = nodeId; // This should be socket.id
        this.modelId = modelId;

        // Initialize WebRTC connection manager
        this.rtcManager = new WebRTCPeerConnection(socket, nodeId);

        // Register callbacks
        this.rtcManager.setDataCallback(this.handleTensorData.bind(this));
        this.rtcManager.setConnectionCallback(this.handleConnectionChange.bind(this));

        // Keep track of available nodes
        this.availableNodes = [];
        this.connectedNodes = new Set();

        // Tensor processing state
        this.pendingOperations = new Map();
        this.completedTensors = new Map();

        // Model state
        this.modelWeights = null;
        this.modelConfig = null;
        this.isModelInitialized = false;
        this.isCoordinator = false;

        // Callback for events
        this.eventCallback = null;

        // Setup socket listeners
        this.setupSocketListeners();
    }

    /**
     * Set up socket listeners for node discovery
     */
    setupSocketListeners() {
        // Listen for new node registrations
        this.socket.on('node_registered', (node) => {
            this.updateNodesList([...this.availableNodes, node]);
        });

        // Listen for node disconnections
        this.socket.on('node_disconnected', (nodeId) => {
            this.availableNodes = this.availableNodes.filter(node => node.id !== nodeId);
            this.connectedNodes.delete(nodeId);

            if (this.eventCallback) {
                this.eventCallback('node-disconnected', { nodeId });
            }
        });

        // Listen for node list updates
        this.socket.on('node_list', (nodes) => {
            this.updateNodesList(nodes);
        });
    }

    /**
     * Update the list of available nodes
     * @param {Array} nodes - List of available nodes
     */
    updateNodesList(nodes) {
        console.log("Updating nodes list:", nodes);

        // Filter nodes to include only those with the same model
        // and exclude our own socket ID
        this.availableNodes = nodes.filter(node => {
            // Make sure we don't try to connect to ourselves
            // Use socketId property or id property depending on what's available
            const isSelf = node.socketId === this.nodeId || node.id === this.nodeId;

            // Ensure it has same model
            const matchingModel = node.model === this.modelId;

            // Ensure the node supports tensor parallelism
            const supportsTensor = node.capabilities &&
                node.capabilities.tensorParallelism === true;

            const result = !isSelf && matchingModel && supportsTensor;
            console.log(`Node ${node.id} (socket: ${node.socketId}): isSelf=${isSelf}, matchingModel=${matchingModel}, supportsTensor=${supportsTensor}, include=${result}`);

            return result;
        });

        console.log(`Found ${this.availableNodes.length} available nodes for tensor parallelism`,
            this.availableNodes.map(n => n.id));

        if (this.eventCallback) {
            this.eventCallback('nodes-updated', {
                count: this.availableNodes.length,
                nodes: this.availableNodes,
                totalConnected: this.connectedNodes.size
            });
        }
    }

    /**
     * Initialize as coordinator node
     * @param {Object} modelConfig - Model configuration
     * @param {Object} modelWeights - Initial model weights
     */
    async initAsCoordinator(modelConfig, modelWeights) {
        this.isCoordinator = true;
        this.modelConfig = modelConfig;
        this.modelWeights = modelWeights;

        // Get current node list
        this.socket.emit('get_nodes');

        if (this.eventCallback) {
            this.eventCallback('initialized-as-coordinator', {
                modelId: this.modelId
            });
        }

        this.isModelInitialized = true;
        return true;
    }

    /**
     * Initialize as worker node
     */
    async initAsWorker() {
        this.isCoordinator = false;

        // Wait for coordinator to send model configuration
        // This will happen through the handleTensorData callback

        if (this.eventCallback) {
            this.eventCallback('initialized-as-worker', {
                modelId: this.modelId
            });
        }

        return true;
    }

    /**
     * Connect to available nodes
     * @returns {Promise<number>} - Number of successful connections
     */
    async connectToNodes() {
        if (this.availableNodes.length === 0) {
            console.log('No available nodes to connect to');
            return 0;
        }

        console.log(`Attempting to connect to ${this.availableNodes.length} nodes:`,
            this.availableNodes.map(n => n.id));

        let successCount = 0;

        for (const node of this.availableNodes) {
            try {
                // Use socketId for WebRTC connection
                const socketId = node.socketId || node.id;

                if (this.connectedNodes.has(socketId)) {
                    console.log(`Already connected to ${socketId}`);
                    successCount++; // Count existing connections
                    continue; // Already connected
                }

                console.log(`Establishing WebRTC connection to ${socketId}...`);

                // Establish WebRTC connection - directly using socket ID
                await this.rtcManager.initConnection(socketId);
                this.connectedNodes.add(socketId);
                successCount++;

                console.log(`Successfully connected to ${socketId}`);

                if (this.eventCallback) {
                    this.eventCallback('node-connected', {
                        nodeId: socketId,
                        totalConnected: this.connectedNodes.size
                    });
                }
            } catch (error) {
                console.error(`Failed to connect to node ${node.id} (socket: ${node.socketId}):`, error);
                // Log the detailed error
                if (this.eventCallback) {
                    this.eventCallback('node-connection-error', {
                        nodeId: node.id,
                        socketId: node.socketId,
                        error: error.message
                    });
                }
            }
        }

        console.log(`Connected to ${successCount}/${this.availableNodes.length} nodes`);
        return successCount;
    }

    /**
     * Handle tensor data received from other nodes
     * @param {string} fromNodeId - Node ID that sent the data
     * @param {ArrayBuffer|string} data - Received data
     */
    handleTensorData(fromNodeId, data) {
        // Check if it's a string (control message) or binary data (tensor)
        if (typeof data === 'string') {
            try {
                const message = JSON.parse(data);

                if (message.type.startsWith('control-')) {
                    this.handleControlMessage(fromNodeId, message);
                } else if (message.type === 'tensor-metadata') {
                    // Store tensor metadata for the upcoming binary data
                    this.pendingOperations.set(fromNodeId, message.metadata);
                }
            } catch (error) {
                console.error('Error parsing message:', error);
            }
        } else if (data instanceof ArrayBuffer) {
            // Process tensor data
            this.processTensorData(fromNodeId, data);
        }
    }

    /**
     * Handle WebRTC connection state changes
     * @param {string} nodeId - Node ID of the connection
     * @param {string} state - New connection state
     */
    handleConnectionChange(nodeId, state) {
        console.log(`Connection state change for ${nodeId}: ${state}`);

        // Update connected nodes tracking
        if (state === 'connected' || state === 'datachannel-open') {
            this.connectedNodes.add(nodeId);
        } else if (state === 'disconnected' || state === 'failed' || state === 'closed' || state === 'datachannel-closed') {
            this.connectedNodes.delete(nodeId);
        }

        // Trigger event callback if defined
        if (this.eventCallback) {
            this.eventCallback('connection-state-change', {
                nodeId,
                state,
                totalConnected: this.connectedNodes.size
            });
        }
    }

    /**
     * Process binary tensor data
     * @param {string} fromNodeId - Node ID that sent the data
     * @param {ArrayBuffer} data - Binary tensor data
     */
    processTensorData(fromNodeId, data) {
        // Get metadata from pending operations
        const metadata = this.pendingOperations.get(fromNodeId);
        if (!metadata) {
            console.error('Received tensor data without metadata');
            return;
        }

        // Clear pending operation
        this.pendingOperations.delete(fromNodeId);

        // Deserialize tensor
        const tensor = tensorUtils.deserializeTensor(data);

        // Store the completed tensor result
        this.completedTensors.set(metadata.operationId, {
            tensor,
            metadata
        });

        // Notify about operation completion
        if (this.eventCallback) {
            this.eventCallback('tensor-operation-completed', {
                operationId: metadata.operationId,
                fromNodeId,
                outputShape: tensor.shape
            });
        }

        // If this completes a distributed operation, trigger the continuation
        this.checkOperationCompletion(metadata.operationId);
    }

    /**
     * Check if a distributed operation is complete
     * @param {string} operationId - Operation identifier
     */
    checkOperationCompletion(operationId) {
        // Implementation depends on the specific operation types
        // For now, we'll assume a simple callback system
        if (this.operationCallbacks && this.operationCallbacks[operationId]) {
            const callback = this.operationCallbacks[operationId];
            const result = this.completedTensors.get(operationId);

            if (result) {
                callback(result);
                this.completedTensors.delete(operationId);
                delete this.operationCallbacks[operationId];
            }
        }
    }

    /**
     * Handle control messages from other nodes
     * @param {string} fromNodeId - Node ID that sent the message
     * @param {Object} message - Control message
     */
    handleControlMessage(fromNodeId, message) {
        const controlType = message.type.replace('control-', '');

        switch (controlType) {
            case 'model-config':
                // Receive model configuration from coordinator
                this.modelConfig = message.payload.config;
                this.isModelInitialized = true;

                if (this.eventCallback) {
                    this.eventCallback('model-config-received', {
                        fromNodeId,
                        modelId: this.modelId
                    });
                }
                break;

            case 'tensor-request':
                // Handle request for tensor computation
                this.handleTensorRequest(fromNodeId, message.payload);
                break;

            case 'operation-complete':
                // Remote node completed an operation
                if (this.eventCallback) {
                    this.eventCallback('remote-operation-completed', {
                        operationId: message.payload.operationId,
                        fromNodeId,
                        timing: message.payload.timing
                    });
                }
                break;

            default:
                console.warn(`Unknown control message type: ${controlType}`);
        }
    }

    /**
     * Handle tensor computation request from coordinator
     * @param {string} fromNodeId - Coordinator node ID
     * @param {Object} request - Computation request
     */
    async handleTensorRequest(fromNodeId, request) {
        const { operationId, operation, inputs, options } = request;

        // Start timing
        const startTime = performance.now();

        try {
            let result;

            // Perform the requested tensor operation
            switch (operation) {
                case 'attention-forward':
                    result = await this.performAttentionForward(inputs, options);
                    break;

                case 'mlp-forward':
                    result = await this.performMLPForward(inputs, options);
                    break;

                case 'attention-backward':
                    result = await this.performAttentionBackward(inputs, options);
                    break;

                case 'mlp-backward':
                    result = await this.performMLPBackward(inputs, options);
                    break;

                default:
                    throw new Error(`Unsupported operation: ${operation}`);
            }

            // Calculate timing
            const endTime = performance.now();
            const timing = endTime - startTime;

            // Send result back to coordinator
            // First, send metadata
            const metadata = {
                operationId,
                originalShape: result.shape,
                dtype: result.tensor.constructor === Float32Array ? 'float32' : 'int32',
                timing
            };

            // Send result metadata
            await this.rtcManager.sendControlMessage(fromNodeId, 'operation-complete', {
                operationId,
                timing,
                shape: result.shape
            });

            // Serialize and send the tensor
            const serialized = tensorUtils.serializeTensor(
                result.tensor,
                result.shape,
                result.tensor.constructor === Float32Array ? 'float32' : 'int32'
            );

            // Send the binary data
            await this.rtcManager.sendTensorData(fromNodeId, serialized, metadata);

            // Log success
            console.log(`Completed operation ${operationId} in ${timing.toFixed(2)} ms`);

            // Update node activity
            this.socket.emit('node_activity', {
                nodeId: this.nodeId,
                socketId: this.socket.id,
                action: 'tensor-operation-completed',
                prompt: `Completed ${operation} in ${timing.toFixed(2)}ms`
            });

        } catch (error) {
            console.error(`Error performing operation ${operationId}:`, error);

            // Send error message back to coordinator
            await this.rtcManager.sendControlMessage(fromNodeId, 'operation-error', {
                operationId,
                error: error.message
            });

            // Update node activity with error
            this.socket.emit('node_activity', {
                nodeId: this.nodeId,
                socketId: this.socket.id,
                action: 'error',
                prompt: `Error in ${operation}: ${error.message}`
            });
        }
    }

    /**
     * Perform attention forward pass (worker node)
     * @param {Object} inputs - Input tensors
     * @param {Object} options - Operation options
     * @returns {Object} - Result tensor with shape
     */
    async performAttentionForward(inputs, options) {
        // Implementation of attention forward pass
        // This is a simplified version - a real implementation would use WebGL or WASM

        const { input, weights, bias } = inputs;
        const { headIndex, numHeads, headDim } = options;

        // Deserialize input if it's serialized
        const inputTensor = input instanceof ArrayBuffer ?
            tensorUtils.deserializeTensor(input) : input;

        // Deserialize weights if serialized
        const weightsTensor = weights instanceof ArrayBuffer ?
            tensorUtils.deserializeTensor(weights) : weights;

        // Placeholder for attention computation
        // In a real implementation, this would do the actual tensor operations
        console.log(`Computing attention forward for head ${headIndex} of ${numHeads}`);

        // Create a mock result (in real implementation, do actual computation)
        const resultShape = [inputTensor.shape[0], inputTensor.shape[1], headDim];
        const resultSize = resultShape.reduce((a, b) => a * b, 1);
        const resultTensor = new Float32Array(resultSize);

        // Fill with mock data (real impl would have actual computation)
        for (let i = 0; i < resultSize; i++) {
            resultTensor[i] = Math.random() * 0.1;
        }

        // Simulate computation time 
        await new Promise(resolve => setTimeout(resolve, 50));

        return { tensor: resultTensor, shape: resultShape };
    }

    /**
     * Perform MLP forward pass (worker node)
     * @param {Object} inputs - Input tensors
     * @param {Object} options - Operation options
     * @returns {Object} - Result tensor with shape
     */
    async performMLPForward(inputs, options) {
        // Implementation of MLP forward pass
        // Similar to attention, this is simplified

        const { input, weights, bias } = inputs;
        const { partitionIndex, numPartitions } = options;

        // Deserialize tensors if needed
        const inputTensor = input instanceof ArrayBuffer ?
            tensorUtils.deserializeTensor(input) : input;

        // Placeholder for MLP computation
        console.log(`Computing MLP forward for partition ${partitionIndex} of ${numPartitions}`);

        // Create a mock result
        const outDim = weights ? weights.shape[1] : inputTensor.shape[inputTensor.shape.length - 1];
        const resultShape = [...inputTensor.shape.slice(0, -1), outDim];
        const resultSize = resultShape.reduce((a, b) => a * b, 1);
        const resultTensor = new Float32Array(resultSize);

        // Fill with mock data
        for (let i = 0; i < resultSize; i++) {
            resultTensor[i] = Math.random() * 0.1;
        }

        // Simulate computation time
        await new Promise(resolve => setTimeout(resolve, 50));

        return { tensor: resultTensor, shape: resultShape };
    }

    /**
     * Set event callback for manager events
     * @param {Function} callback - Event callback function
     */
    setEventCallback(callback) {
        if (typeof callback === 'function') {
            this.eventCallback = callback;
            // Add enhanced debugging for connection events
            if (this.rtcManager) {
                console.log("Adding enhanced debug logging to WebRTC connection events");
                const originalCallback = this.rtcManager.onConnectionCallback;

                this.rtcManager.setConnectionCallback((nodeId, state) => {
                    console.log(`[RTCDebug] Connection state change for ${nodeId}: ${state}`);
                    if (originalCallback) originalCallback(nodeId, state);
                });
            }
        }
    }

    /**
     * Distribute model inference across available nodes
     * @param {Object} inputData - Input data for inference
     * @returns {Promise<Object>} - Inference result
     */
    async runDistributedInference(inputData) {
        if (!this.isModelInitialized) {
            throw new Error('Model is not initialized');
        }

        if (!this.isCoordinator) {
            throw new Error('Only coordinator nodes can initiate inference');
        }

        // Ensure we have connections to worker nodes
        if (this.connectedNodes.size === 0) {
            await this.connectToNodes();

            if (this.connectedNodes.size === 0) {
                // No worker nodes available, fall back to local execution
                console.warn('No worker nodes available, running inference locally');
                return this.runLocalInference(inputData);
            }
        }

        try {
            // Notify about inference start
            if (this.eventCallback) {
                this.eventCallback('inference-started', {
                    numNodes: this.connectedNodes.size + 1 // +1 for this node
                });
            }

            // Log inference start
            this.socket.emit('node_activity', {
                nodeId: this.nodeId,
                socketId: this.socket.id,
                action: 'inference-started',
                prompt: `Starting distributed inference across ${this.connectedNodes.size + 1} nodes`
            });

            // Create distribution plan for model layers
            const nodeArray = [...this.connectedNodes].map(id => {
                return this.availableNodes.find(node => node.id === id) || { id };
            });

            // Add this node to the array for complete distribution
            nodeArray.push({
                id: this.nodeId,
                status: 'online',
                // Add capability data here
            });

            // Create worker assignments
            const modelLayers = {}; // Extract from modelConfig
            const assignment = tensorUtils.createWorkerAssignment(nodeArray, modelLayers);

            // Start the inference pipeline
            // This is a simplification - real implementation would have proper pipeline stages

            // Placeholder for actual distributed inference logic
            // In reality, this would coordinate the flow of tensors between nodes

            // Simulate the core computation
            await new Promise(resolve => setTimeout(resolve, 500));

            // Log inference completion
            this.socket.emit('node_activity', {
                nodeId: this.nodeId,
                socketId: this.socket.id,
                action: 'inference-completed',
                prompt: 'Completed distributed inference'
            });

            // Create a mock result
            return {
                text: 'This is a mock inference result from distributed computation',
                timing: {
                    total: 500,
                    distribution: 20,
                    computation: 450,
                    communication: 30
                }
            };

        } catch (error) {
            console.error('Error during distributed inference:', error);

            // Log error
            this.socket.emit('node_activity', {
                nodeId: this.nodeId,
                socketId: this.socket.id,
                action: 'error',
                prompt: `Inference error: ${error.message}`
            });

            throw error;
        }
    }

    /**
     * Fallback to run inference locally if no nodes are available
     * @param {Object} inputData - Input data for inference
     * @returns {Promise<Object>} - Inference result
     */
    async runLocalInference(inputData) {
        // Placeholder for local inference without distribution
        return {
            text: 'This is a local inference result (no distribution)',
            timing: {
                total: 1000,
                computation: 1000
            }
        };
    }

    /**
     * Clean up all connections and resources
     */
    cleanup() {
        // Close all WebRTC connections
        this.rtcManager.closeAllConnections();

        // Clear state
        this.connectedNodes.clear();
        this.pendingOperations.clear();
        this.completedTensors.clear();

        // Log cleanup
        this.socket.emit('node_activity', {
            nodeId: this.nodeId,
            socketId: this.socket.id,
            action: 'cleanup',
            prompt: 'Cleaned up distributed model manager'
        });

        console.log('Distributed model manager cleaned up');
    }
}

export default DistributedModelManager; 