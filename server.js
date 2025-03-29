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
    },
    // Add these settings for Vercel compatibility
    transports: ['websocket', 'polling'],
    path: '/socket.io/',
    pingTimeout: 60000,
    pingInterval: 25000,
    upgradeTimeout: 30000,
    maxHttpBufferSize: 1e8
});

// Serve static files from the dist directory (Vite build output)
app.use(express.static(path.join(__dirname, 'dist')));

// Store connected nodes
const nodes = {};

// Store node capabilities for parallel computing
const nodeCapabilities = {};

// Node type/role tracking
const nodeRoles = {
    coordinator: new Set(),
    worker: new Set()
};

// Store socket ID to node ID mapping
const socketToNodeMap = {};

// Add a health check endpoint for Vercel
app.get('/api/health', (req, res) => {
    res.status(200).json({ status: 'ok', uptime: process.uptime() });
});

io.on('connection', (socket) => {
    console.log('New client connected:', socket.id);

    // Send immediate acknowledgment to client
    socket.emit('connection_ack', { socketId: socket.id });

    // Register a new WebLLM node
    socket.on('register_node', (nodeData) => {
        console.log('Node registered:', nodeData);

        // Add connection timestamp
        const nodeInfo = {
            ...nodeData,
            socketId: socket.id,
            connectedAt: new Date().toISOString(),
            capabilities: nodeData.capabilities || {
                // Default capabilities if none provided
                gpuMemory: 0,      // GPU memory in MB
                cpuCores: 1,       // CPU cores
                tensorParallelism: false // Whether the node supports tensor parallelism
            }
        };

        // Store node info
        nodes[nodeData.id] = nodeInfo;

        // Map socket ID to node ID for easier lookups
        socketToNodeMap[socket.id] = nodeData.id;

        // Store capabilities separately for faster access
        nodeCapabilities[nodeData.id] = nodeInfo.capabilities;

        // Track node role
        if (nodeInfo.role === 'coordinator') {
            nodeRoles.coordinator.add(nodeData.id);
        } else {
            nodeRoles.worker.add(nodeData.id);
        }

        // Notify all clients about the new node
        io.emit('node_registered', nodeInfo);
    });

    // Get all nodes
    socket.on('get_nodes', () => {
        socket.emit('node_list', Object.values(nodes));
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
        // Broadcast message to all other clients
        socket.broadcast.emit('message', messageWithSocket);
    });

    // Handle node activity logs
    socket.on('node_activity', (activity) => {
        console.log('Node activity:', activity);
        // Make sure the socketId is included in the broadcasted activity
        const activityWithSocket = {
            ...activity,
            socketId: activity.socketId || socket.id
        };
        // Broadcast activity to all clients
        socket.broadcast.emit('node_activity', activityWithSocket);
    });

    // Handle node unregistration
    socket.on('unregister_node', (data) => {
        console.log('Node unregistered:', data.id);

        if (nodes[data.id]) {
            // Clean up socket-to-node mapping
            if (nodes[data.id].socketId) {
                delete socketToNodeMap[nodes[data.id].socketId];
            }

            delete nodes[data.id];

            // Notify all clients about the unregistration
            io.emit('node_disconnected', data.id);
        }
    });

    // Handle disconnection
    socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);

        // First check if this socket ID is directly mapped to a node
        if (socketToNodeMap[socket.id]) {
            const nodeId = socketToNodeMap[socket.id];
            console.log(`Disconnected socket ${socket.id} mapped to node ${nodeId}`);

            // Clean up node data
            delete nodes[nodeId];
            delete nodeCapabilities[nodeId];
            delete socketToNodeMap[socket.id];

            // Remove from role tracking
            nodeRoles.coordinator.delete(nodeId);
            nodeRoles.worker.delete(nodeId);

            // Notify all clients about the disconnection
            io.emit('node_disconnected', nodeId);
            // Also notify with socket ID for WebRTC connections
            io.emit('node_disconnected', socket.id);
        }
        // Legacy method: check if node uses socket.id as ID
        else if (nodes[socket.id]) {
            console.log('Node disconnected (using socket.id as node.id):', socket.id);

            // Clean up node data
            delete nodes[socket.id];
            delete nodeCapabilities[socket.id];

            // Remove from role tracking
            nodeRoles.coordinator.delete(socket.id);
            nodeRoles.worker.delete(socket.id);

            // Notify all clients about the disconnection
            io.emit('node_disconnected', socket.id);
        }
        // For backward compatibility, try to find by socketId property
        else {
            const nodeId = Object.keys(nodes).find(id => nodes[id].socketId === socket.id);
            if (nodeId) {
                console.log('Legacy node disconnected:', nodeId);

                // Clean up node data
                delete nodes[nodeId];
                delete nodeCapabilities[nodeId];
                delete socketToNodeMap[socket.id];

                // Remove from role tracking
                nodeRoles.coordinator.delete(nodeId);
                nodeRoles.worker.delete(nodeId);

                // Notify all clients about both the node ID and socket ID
                io.emit('node_disconnected', nodeId);
                io.emit('node_disconnected', socket.id);
            }
        }
    });

    // Handle WebRTC signaling
    socket.on('webrtc_signal', (data) => {
        console.log(`WebRTC signal (${data.type}) from ${data.from} to ${data.to}`);

        // Forward the signal directly to the intended recipient using socket ID
        socket.to(data.to).emit('webrtc_signal', data);
    });

    // Update node capabilities
    socket.on('update_node_capabilities', (data) => {
        console.log(`Updating capabilities for node ${data.id}:`, data.capabilities);

        if (nodes[data.id]) {
            // Update node capabilities
            nodes[data.id].capabilities = data.capabilities;

            // Also update the separate capabilities store
            nodeCapabilities[data.id] = data.capabilities;

            // Update node role if tensor parallelism capability changed
            if (data.capabilities.tensorParallelism === true) {
                nodeRoles.coordinator.add(data.id);
                console.log(`Node ${data.id} added as coordinator for tensor parallelism`);
            } else {
                nodeRoles.coordinator.delete(data.id);
                nodeRoles.worker.add(data.id);
                console.log(`Node ${data.id} set as worker (tensor parallelism disabled)`);
            }

            // Notify all clients about the update
            io.emit('node_capability_update', {
                id: data.id,
                capabilities: data.capabilities
            });
        }
    });

    // Get tensor-parallelism capable nodes
    socket.on('get_tensor_nodes', (modelId) => {
        console.log(`Searching for tensor-capable nodes for model ${modelId}`);
        console.log('Current nodes:', Object.keys(nodes));
        console.log('Node capabilities:', nodeCapabilities);

        // Filter nodes by model and tensor parallelism capability
        const tensorNodes = Object.values(nodes).filter(node => {
            const hasTensorCapability = node.capabilities &&
                node.capabilities.tensorParallelism === true;

            const matchesModel = node.model === modelId;

            console.log(`Node ${node.id}: model=${node.model}, tensorCapability=${hasTensorCapability}`);

            return matchesModel && hasTensorCapability;
        });

        console.log(`Found ${tensorNodes.length} tensor-capable nodes`);
        socket.emit('tensor_node_list', tensorNodes);
    });

    // Get node assignments for tensor parallelism
    socket.on('assign_tensor_nodes', (data) => {
        const { coordinatorId, modelId, assignments } = data;

        // Validate coordinator
        if (!nodes[coordinatorId]) {
            socket.emit('tensor_assignment_error', { error: 'Invalid coordinator ID' });
            return;
        }

        // Update assignments and notify nodes
        Object.entries(assignments).forEach(([nodeId, layers]) => {
            if (nodes[nodeId] && nodes[nodeId].socketId) {
                io.to(nodes[nodeId].socketId).emit('tensor_assignment', {
                    coordinatorId,
                    layers,
                    modelId
                });
            }
        });

        // Confirm assignments
        socket.emit('tensor_assignments_complete', {
            assignedCount: Object.keys(assignments).length
        });
    });
});

// Handle any requests that don't match the ones above
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

const PORT = process.env.PORT || 8080;
server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
}); 