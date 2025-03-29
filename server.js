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

// Serve static files from the dist directory (Vite build output)
app.use(express.static(path.join(__dirname, 'dist')));

// Store connected nodes
const nodes = {};

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
        // Broadcast message to all other clients
        socket.broadcast.emit('message', message);
    });

    // Handle node activity logs
    socket.on('node_activity', (activity) => {
        console.log('Node activity:', activity);
        // Broadcast activity to all clients
        socket.broadcast.emit('node_activity', activity);
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

// Handle any requests that don't match the ones above
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

const PORT = process.env.PORT || 8080;
server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
}); 