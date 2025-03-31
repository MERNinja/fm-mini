import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

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

// Store registered tensor models
const tensorModels = {};

// Store tensor parallel availability status
const tensorParallelStatus = {};

// Create a function to get nodes with tensor parallel capability
const getTensorParallelNodes = () => {
    return Object.values(nodes).filter(node => {
        // Check if the node has registered tensor parallel capability
        return tensorParallelStatus[node.id]?.enabled === true;
    });
};

io.on('connection', (socket) => {
    console.log('New client connected:', socket.id);

    // Register a new WebLLM node
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
        if (typeof callback === 'function') {
            // Return all nodes by default
            callback(Object.values(nodes));
        } else {
            socket.emit('node_list', Object.values(nodes));
        }
    });

    // Get only tensor parallel enabled nodes
    socket.on('get_tensor_parallel_nodes', (callback) => {
        const parallelNodes = getTensorParallelNodes();
        
        if (typeof callback === 'function') {
            callback(parallelNodes);
        } else {
            socket.emit('tensor_parallel_nodes', parallelNodes);
        }
    });

    // Handle node status updates
    socket.on('status_update', (data) => {
        if (nodes[data.id]) {
            nodes[data.id].status = data.status;
            io.emit('node_status_update', {
                id: data.id,
                status: data.status
            });
        }
    });

    // Handle messages between nodes
    socket.on('message', (message) => {
        console.log('Message received:', message);
        // Ensure socketId is included in the message
        const messageWithSocket = {
            ...message,
            socketId: message.socketId || socket.id
        };
        
        // Special handling for tensor request messages
        if (message.text && 
           (message.text.includes('[TENSOR_REQUEST]') || 
            message.text.includes('[TENSOR_ACK]') || 
            message.text.includes('[TENSOR_RESULT]'))) {
            
            // Only forward to specific target node, not broadcast
            if (message.to) {
                const targetNodeId = message.to;
                console.log(`Routing tensor message from ${message.from} to specific target ${targetNodeId}`);
                
                // Find the target node's socket
                const targetNode = Object.values(nodes).find(node => node.id === targetNodeId);
                
                if (targetNode && targetNode.socketId) {
                    // Send directly to the target node only
                    io.to(targetNode.socketId).emit('message', messageWithSocket);
                    console.log(`Successfully forwarded tensor message: ${message.text} to ${targetNodeId} (${targetNode.socketId})`);
                } else {
                    console.warn(`Target node ${targetNodeId} not found for tensor message`);
                }
            } else {
                console.warn('Tensor message missing target node ID, dropping message');
            }
            
            // Don't broadcast tensor messages
            return;
        }
        
        // For regular messages, broadcast to all clients
        socket.broadcast.emit('message', messageWithSocket);
    });

    // Handle direct node messages (explicitly directed communication between origin and peer nodes)
    socket.on('direct_node_message', (message) => {
        console.log('Direct node message received:', message.action || 'direct_message', 'from:', message.from, 'to:', message.to);
        
        // Ensure socketId is included in the message
        const messageWithSocket = {
            ...message,
            socketId: message.socketId || socket.id
        };
        
        // These messages should always have a specific target
        if (message.to) {
            const targetNodeId = message.to;
            console.log(`Routing direct node message from ${message.from} to ${targetNodeId}`);
            
            // Find the target node's socket
            const targetNode = Object.values(nodes).find(node => node.id === targetNodeId);
            
            if (targetNode && targetNode.socketId) {
                // Send directly to the target node only
                io.to(targetNode.socketId).emit('direct_node_message', messageWithSocket);
                console.log(`Successfully forwarded direct message to ${targetNodeId} (${targetNode.socketId})`);
            } else {
                console.warn(`Target node ${targetNodeId} not found for direct message`);
            }
        } else {
            console.warn('Direct node message missing target node ID, dropping message');
        }
    });

    // Handle node activity logs
    socket.on('node_activity', (activity) => {
        console.log('Node activity:', activity);
        // Make sure the socketId is included in the broadcasted activity
        const activityWithSocket = {
            ...activity,
            socketId: activity.socketId || socket.id
        };
        
        // Handle private direct messages first - explicitly for private task assignments
        if (activity.private === true || activity.directMessage === true) {
            // This is a private message that should only go to the specific target
            if (activity.targetNodeId) {
                // This is directed at a specific peer node only
                console.log(`Private message from ${activity.nodeId} to ${activity.targetNodeId}`);
                
                // Find the target node's socket ID
                const targetNode = Object.values(nodes).find(node => node.id === activity.targetNodeId);
                
                if (targetNode && targetNode.socketId) {
                    // Send directly to the target node ONLY
                    io.to(targetNode.socketId).emit('node_activity', activityWithSocket);
                    console.log(`Sent private message to ${activity.targetNodeId} (${targetNode.socketId})`);
                } else {
                    console.warn(`Target node ${activity.targetNodeId} not found for private message`);
                }
                
                // IMPORTANT: Do not broadcast private messages to anyone else
                return;
            }
        }
        
        // Special handling for origin node activities
        if (activity.originNode) {
            // This activity is from or about an origin node
            
            // CRITICAL FIX: Only send origin node activities to their specific targets
            // or back to the origin node itself, NEVER broadcast to all nodes
            if (activity.targetNodeId) {
                // This activity is directed at a specific peer node
                console.log(`Origin node ${activity.originNode} sending targeted activity to peer ${activity.targetNodeId}`);
                
                // Find the target node's socket ID
                const targetNode = Object.values(nodes).find(node => node.id === activity.targetNodeId);
                
                if (targetNode && targetNode.socketId) {
                    // Send directly to the target node
                    io.to(targetNode.socketId).emit('node_activity', activityWithSocket);
                    console.log(`Successfully forwarded targeted origin node activity to peer ${activity.targetNodeId}`);
                } else {
                    console.warn(`Target peer node ${activity.targetNodeId} not found for origin node activity`);
                }
            } else {
                // This is an internal origin node activity (result collection, waiting, etc.)
                // ONLY send it back to the origin node, never broadcast it
                const originNodeId = activity.originNode;
                const originNode = Object.values(nodes).find(node => node.id === originNodeId);
                
                if (originNode && originNode.socketId) {
                    console.log(`Sending internal origin node activity back to origin ${originNodeId}`);
                    io.to(originNode.socketId).emit('node_activity', activityWithSocket);
                }
            }
            
            // CRITICAL: Do not broadcast origin node activities
            return;
        }
        
        // Handle activities from peer nodes responding to origin
        if (activity.isPeerResponse && activity.targetNodeId) {
            // This is a response from a peer node targeting the origin node
            console.log(`Peer node ${activity.nodeId} sending response to origin ${activity.targetNodeId}`);
            
            // Find the origin node's socket ID
            const originNode = Object.values(nodes).find(node => node.id === activity.targetNodeId);
            
            if (originNode && originNode.socketId) {
                // Send directly to the origin node only
                io.to(originNode.socketId).emit('node_activity', activityWithSocket);
                console.log(`Successfully forwarded peer response to origin node ${activity.targetNodeId}`);
            } else {
                console.warn(`Origin node ${activity.targetNodeId} not found for peer response`);
            }
            
            // Do not broadcast peer responses
            return;
        }
        
        // For all other general activities (not related to tensor parallelism),
        // broadcast to all clients
        socket.broadcast.emit('node_activity', activityWithSocket);
    });

    // Handle node unregistration
    socket.on('unregister_node', (data) => {
        console.log('Node unregistered:', data.id);

        if (nodes[data.id]) {
            delete nodes[data.id];

            // Notify all clients about the unregistration
            io.emit('node_disconnected', data.id);
        }
    });

    // ===== TENSOR PARALLELISM SIGNALING =====
    
    // Register a model for tensor parallelism
    socket.on('register_tensor_model', (data) => {
        console.log('Tensor model registered:', data);
        
        const { nodeId, modelId, modelInfo } = data;
        
        if (!tensorModels[modelId]) {
            tensorModels[modelId] = {};
        }
        
        tensorModels[modelId][nodeId] = {
            ...modelInfo,
            socketId: socket.id,
            registeredAt: new Date().toISOString()
        };
        
        // Notify all clients about the tensor model registration
        io.emit('tensor_model_registered', {
            nodeId,
            modelId, 
            socketId: socket.id
        });
    });
    
    // Get available tensor models
    socket.on('get_tensor_models', () => {
        socket.emit('tensor_models_list', tensorModels);
    });
    
    // WebRTC signaling for tensor parallelism
    socket.on('tensor_signal', (data) => {
        console.log('Tensor signal:', data.type, 'from:', data.from, 'to:', data.to);
        
        // Find the target node's socket ID
        const targetNodeId = data.to;
        const targetNode = Object.values(nodes).find(node => node.id === targetNodeId);
        
        if (targetNode && targetNode.socketId) {
            // Forward the signal to the target node
            io.to(targetNode.socketId).emit('tensor_signal', data);
        } else {
            console.warn(`Target node ${targetNodeId} not found or has no socket ID`);
        }
    });
    
    // Tensor operation request forwarding
    socket.on('tensor_operation_request', (data) => {
        console.log('Tensor operation request from:', data.from, 'to:', data.to);
        
        // Find the target node's socket ID
        const targetNodeId = data.to;
        const targetNode = Object.values(nodes).find(node => node.id === targetNodeId);
        
        if (targetNode && targetNode.socketId) {
            // Forward the request to the target node
            io.to(targetNode.socketId).emit('tensor_operation_request', data);
        } else {
            console.warn(`Target node ${targetNodeId} not found for tensor operation request`);
        }
    });
    
    // Tensor operation result forwarding
    socket.on('tensor_operation_result', (data) => {
        console.log('Tensor operation result from:', data.from, 'to:', data.to);
        
        // Find the target node's socket ID
        const targetNodeId = data.to;
        const targetNode = Object.values(nodes).find(node => node.id === targetNodeId);
        
        if (targetNode && targetNode.socketId) {
            // Forward the result to the target node
            io.to(targetNode.socketId).emit('tensor_operation_result', data);
        } else {
            console.warn(`Target node ${targetNodeId} not found for tensor operation result`);
        }
    });
    
    // Tensor operation error forwarding
    socket.on('tensor_operation_error', (data) => {
        console.log('Tensor operation error from:', data.from, 'to:', data.to, 'error:', data.error);
        
        // Find the target node's socket ID
        const targetNodeId = data.to;
        const targetNode = Object.values(nodes).find(node => node.id === targetNodeId);
        
        if (targetNode && targetNode.socketId) {
            // Forward the error to the target node
            io.to(targetNode.socketId).emit('tensor_operation_error', data);
        } else {
            console.warn(`Target node ${targetNodeId} not found for tensor operation error`);
        }
    });

    // Generic tensor operation forwarding (for process_layers etc.)
    socket.on('operation', (data) => {
        console.log('Tensor operation from:', data.from, 'to:', data.to, 'operation:', data.operation);
        console.log('Operation data:', JSON.stringify(data.data || {}, null, 2));
        
        // CRITICAL FIX: Enhanced validation of operation data
        if (!data.to || !data.from) {
            console.error('⛔ INVALID OPERATION: Missing to/from fields:', data);
            return;
        }

        // Confirm the sender exists in our nodes registry
        const senderExists = nodes[data.from] !== undefined;
        if (!senderExists) {
            console.error(`⛔ Operation sender node ${data.from} is not registered!`);
            // Register node if it has an id but isn't in our registry
            if (data.from.startsWith('node_')) {
                console.log(`Auto-registering missing node ${data.from}`);
                nodes[data.from] = {
                    id: data.from,
                    socketId: socket.id,
                    autoRegistered: true,
                    status: 'online',
                    connectedAt: new Date().toISOString()
                };
            }
        }
        
        // CRITICAL FIX: Enhanced target node lookup with fallbacks
        // Find the target node's socket ID
        const targetNodeId = data.to;
        let targetNode = Object.values(nodes).find(node => node.id === targetNodeId);
        
        // Additional lookup by socketId as fallback
        if (!targetNode && data.to.includes('socket_')) {
            console.log('Attempting fallback lookup by socket ID');
            targetNode = Object.values(nodes).find(node => node.socketId === data.to);
        }
        
        if (targetNode && targetNode.socketId) {
            // CRITICAL FIX: More reliable forwarding with confirmation
            try {
                // Forward the operation to the target node
                io.to(targetNode.socketId).emit('operation', data);
                
                // Send a direct message as backup route
                io.to(targetNode.socketId).emit('direct_node_message', {
                    from: data.from,
                    to: data.to,
                    action: 'operation_forwarded',
                    taskId: data.taskId,
                    operation: data.operation,
                    timestamp: new Date().toISOString()
                });
                
                console.log(`✅ Successfully forwarded operation to ${targetNodeId} (${targetNode.socketId})`);
                
                // Notify the sender that forwarding was successful
                socket.emit('operation_forwarded', {
                    success: true,
                    to: data.to,
                    operation: data.operation,
                    timestamp: new Date().toISOString()
                });
            } catch (err) {
                console.error(`Failed to forward operation to ${targetNodeId}:`, err);
                
                // Notify the sender of the failure
                socket.emit('operation_forwarded', {
                    success: false,
                    to: data.to,
                    error: err.message,
                    timestamp: new Date().toISOString()
                });
            }
        } else {
            // CRITICAL FIX: Enhanced error reporting and recovery
            console.warn(`⚠️ Target node ${targetNodeId} not found for tensor operation`);
            
            // Check if target ID is valid format
            if (!targetNodeId.startsWith('node_')) {
                console.error(`Invalid target node ID format: ${targetNodeId}. Must start with 'node_'`);
            }
            
            // Log all available nodes to help diagnose the issue
            console.log('Available nodes:', Object.keys(nodes).join(', '));
            
            // Notify the sender that the target wasn't found
            socket.emit('operation_forwarded', {
                success: false,
                to: data.to,
                error: `Target node ${targetNodeId} not found`,
                availableNodes: Object.keys(nodes),
                timestamp: new Date().toISOString()
            });
        }
    });

    // Operation results forwarding
    socket.on('operation_result', (data) => {
        console.log('Operation result from:', data.from, 'to:', data.to);
        console.log('Result data:', JSON.stringify(data.result || {}, null, 2));
        
        // CRITICAL FIX: Enhanced validation of result data
        if (!data.to || !data.from) {
            console.error('⛔ INVALID RESULT: Missing to/from fields:', data);
            return;
        }
        
        // Find the target node's socket ID
        const targetNodeId = data.to;
        let targetNode = Object.values(nodes).find(node => node.id === targetNodeId);
        
        // Fallback lookup by socketId
        if (!targetNode && data.to.includes('socket_')) {
            console.log('Attempting fallback lookup by socket ID');
            targetNode = Object.values(nodes).find(node => node.socketId === data.to);
        }
        
        if (targetNode && targetNode.socketId) {
            // CRITICAL FIX: More reliable result forwarding with confirmation
            try {
                // Forward the result to the target node
                io.to(targetNode.socketId).emit('operation_result', data);
                console.log(`✅ Successfully forwarded result to ${targetNodeId} (${targetNode.socketId})`);
                
                // Notify the sender that forwarding was successful
                socket.emit('result_forwarded', {
                    success: true,
                    to: data.to,
                    operation: data.operation,
                    timestamp: new Date().toISOString()
                });
            } catch (err) {
                console.error(`Failed to forward result to ${targetNodeId}:`, err);
                
                // Notify the sender of the failure
                socket.emit('result_forwarded', {
                    success: false,
                    to: data.to,
                    error: err.message,
                    timestamp: new Date().toISOString()
                });
            }
        } else {
            console.warn(`⚠️ Target node ${targetNodeId} not found for operation result`);
            
            // Log all available nodes to help diagnose the issue
            console.log('Available nodes:', Object.keys(nodes).join(', '));
            
            // Notify the sender that the target wasn't found
            socket.emit('result_forwarded', {
                success: false,
                to: data.to,
                error: `Target node ${targetNodeId} not found`,
                availableNodes: Object.keys(nodes),
                timestamp: new Date().toISOString()
            });
        }
    });

    // Register tensor parallel capability explicitly
    socket.on('register_tensor_parallel', (data) => {
        console.log('Tensor parallel capability registered:', data);
        
        const { nodeId, modelId, enabled } = data;
        
        // Store the tensor parallel status
        if (!tensorParallelStatus[nodeId]) {
            tensorParallelStatus[nodeId] = {};
        }
        
        tensorParallelStatus[nodeId].enabled = enabled;
        tensorParallelStatus[nodeId].modelId = modelId;
        
        // Update the node's status to include tensor parallel capability
        if (nodes[nodeId]) {
            nodes[nodeId].tensorParallelEnabled = enabled;
        }
        
        // Notify all clients about tensor parallel capability
        io.emit('tensor_parallel_status', {
            nodeId,
            enabled,
            modelId
        });
    });
    
    // Get tensor parallel status
    socket.on('get_tensor_parallel_status', (callback) => {
        if (typeof callback === 'function') {
            callback(tensorParallelStatus);
        }
    });

    // Handle disconnection
    socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);

        // Find and remove the disconnected node
        const nodeId = Object.keys(nodes).find(id => nodes[id].socketId === socket.id);

        if (nodeId) {
            console.log('Node disconnected:', nodeId);
            delete nodes[nodeId];
            
            // Also remove tensor parallel status
            if (tensorParallelStatus[nodeId]) {
                delete tensorParallelStatus[nodeId];
            }

            // Notify all clients about the disconnection
            io.emit('node_disconnected', nodeId);
            
            // Remove any tensor models registered by this node
            for (const modelId in tensorModels) {
                if (tensorModels[modelId][nodeId]) {
                    delete tensorModels[modelId][nodeId];
                    
                    // If no nodes are left for this model, remove the model
                    if (Object.keys(tensorModels[modelId]).length === 0) {
                        delete tensorModels[modelId];
                    }
                }
            }
        }
    });
});

// Simple status route to confirm server is running
app.get('/status', (req, res) => {
    res.json({ status: 'WebSocket Signaling Server Running' });
});

// Try multiple ports in case of conflict
function startServer(port) {
    server.listen(port)
        .on('error', (err) => {
            if (err.code === 'EADDRINUSE') {
                console.warn(`Port ${port} is already in use, trying ${port + 1}...`);
                startServer(port + 1);
            } else {
                console.error('Server error:', err);
            }
        })
        .on('listening', () => {
            console.log(`WebSocket Signaling Server running on port ${port}`);
        });
}

const PORT = process.env.PORT || 8080;
startServer(PORT); 