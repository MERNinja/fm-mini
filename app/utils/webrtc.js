/**
 * WebRTC utility module for establishing peer-to-peer connections between nodes
 * Handles signaling, ICE candidates, and data channel setup for tensor distribution
 */

class WebRTCPeerConnection {
    constructor(socketConnection, localNodeId) {
        this.socket = socketConnection;
        this.localNodeId = localNodeId;
        this.peerConnections = {}; // Store RTCPeerConnections by nodeId
        this.dataChannels = {}; // Store dataChannels by nodeId
        this.onDataCallback = null; // Callback for received tensor data
        this.onConnectionCallback = null; // Callback for connection status
        this.connectionAttempts = {}; // Track connection attempts
        this.MAX_CONNECTION_ATTEMPTS = 3;
        this.setupSocketListeners();
    }

    /**
     * Set up WebRTC socket event listeners for signaling
     */
    setupSocketListeners() {
        // Listen for signaling messages
        this.socket.on('webrtc_signal', async (data) => {
            const { from, to, signal, type } = data;

            // Only process messages intended for this node
            if (to !== this.localNodeId) return;

            console.log(`Received ${type} signal from ${from}`);

            if (type === 'offer') {
                await this.handleOffer(from, signal);
            } else if (type === 'answer') {
                await this.handleAnswer(from, signal);
            } else if (type === 'ice-candidate') {
                await this.handleIceCandidate(from, signal);
            }
        });

        // Listen for disconnect events to clean up connections
        this.socket.on('node_disconnected', (nodeId) => {
            this.closePeerConnection(nodeId);
        });
    }

    /**
     * Initialize peer connection with a remote node
     * @param {string} remoteNodeId - ID of the remote node to connect to
     * @returns {Promise<RTCDataChannel>} - The data channel for the connection
     */
    async initConnection(remoteNodeId) {
        // Track connection attempts
        if (!this.connectionAttempts[remoteNodeId]) {
            this.connectionAttempts[remoteNodeId] = 0;
        }

        this.connectionAttempts[remoteNodeId]++;
        console.log(`Connection attempt ${this.connectionAttempts[remoteNodeId]} for ${remoteNodeId}`);

        if (this.peerConnections[remoteNodeId]) {
            const connectionState = this.peerConnections[remoteNodeId].connectionState;
            console.log(`Connection to ${remoteNodeId} already exists with state: ${connectionState}`);

            if (connectionState === 'connected' || connectionState === 'completed') {
                return this.dataChannels[remoteNodeId];
            } else if (connectionState === 'failed' || connectionState === 'disconnected' || connectionState === 'closed') {
                console.log(`Cleaning up failed connection to ${remoteNodeId}`);
                this.closePeerConnection(remoteNodeId);
            }
        }

        console.log(`Initializing connection to ${remoteNodeId}`);

        // Create a new RTCPeerConnection with more STUN servers
        const peerConnection = new RTCPeerConnection({
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' },
                { urls: 'stun:stun2.l.google.com:19302' },
                { urls: 'stun:stun3.l.google.com:19302' },
                { urls: 'stun:stun4.l.google.com:19302' }
            ],
            iceCandidatePoolSize: 10
        });

        // Store the peer connection
        this.peerConnections[remoteNodeId] = peerConnection;

        // Create a data channel for tensor transfer
        const dataChannel = peerConnection.createDataChannel('tensorData', {
            ordered: true,          // Reliable delivery
            maxRetransmits: 30,     // Max retransmissions before giving up
            maxPacketLifeTime: 5000 // Max milliseconds to keep retrying
        });

        // Configure data channel
        this.setupDataChannel(dataChannel, remoteNodeId);
        this.dataChannels[remoteNodeId] = dataChannel;

        // Set up ICE candidate handling
        peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                console.log(`Generated ICE candidate for ${remoteNodeId}`);
                // Send ICE candidate to remote node through signaling server
                this.socket.emit('webrtc_signal', {
                    type: 'ice-candidate',
                    from: this.localNodeId,
                    to: remoteNodeId,
                    signal: event.candidate
                });
            }
        };

        // Handle ICE connection state changes
        peerConnection.oniceconnectionstatechange = () => {
            console.log(`ICE connection state with ${remoteNodeId}: ${peerConnection.iceConnectionState}`);
            if (peerConnection.iceConnectionState === 'failed') {
                console.log(`ICE failed for ${remoteNodeId}, attempting restart`);
                try {
                    peerConnection.restartIce();
                } catch (error) {
                    console.error('Failed to restart ICE:', error);
                }
            }
        };

        // Handle state changes
        peerConnection.onconnectionstatechange = () => {
            console.log(`Connection state with ${remoteNodeId}: ${peerConnection.connectionState}`);
            if (this.onConnectionCallback) {
                this.onConnectionCallback(remoteNodeId, peerConnection.connectionState);
            }
        };

        // Handle incoming data channels
        peerConnection.ondatachannel = (event) => {
            console.log(`Received data channel from ${remoteNodeId}`);
            this.setupDataChannel(event.channel, remoteNodeId);
            this.dataChannels[remoteNodeId] = event.channel;
        };

        // Create and send offer
        try {
            const offer = await peerConnection.createOffer();
            await peerConnection.setLocalDescription(offer);

            // Send offer to remote node through signaling server
            this.socket.emit('webrtc_signal', {
                type: 'offer',
                from: this.localNodeId,
                to: remoteNodeId,
                signal: offer
            });
        } catch (error) {
            console.error('Error creating offer:', error);
            this.closePeerConnection(remoteNodeId);
            throw error;
        }

        return dataChannel;
    }

    /**
     * Handle incoming WebRTC offer
     * @param {string} from - Node ID of the sender
     * @param {RTCSessionDescriptionInit} offer - WebRTC offer
     */
    async handleOffer(from, offer) {
        try {
            // Create a new peer connection if it doesn't exist
            if (!this.peerConnections[from]) {
                const peerConnection = new RTCPeerConnection({
                    iceServers: [
                        { urls: 'stun:stun.l.google.com:19302' },
                        { urls: 'stun:stun1.l.google.com:19302' }
                    ]
                });

                this.peerConnections[from] = peerConnection;

                // Set up ICE candidate handling
                peerConnection.onicecandidate = (event) => {
                    if (event.candidate) {
                        this.socket.emit('webrtc_signal', {
                            type: 'ice-candidate',
                            from: this.localNodeId,
                            to: from,
                            signal: event.candidate
                        });
                    }
                };

                // Handle state changes
                peerConnection.onconnectionstatechange = () => {
                    console.log(`Connection state with ${from}: ${peerConnection.connectionState}`);
                    if (this.onConnectionCallback) {
                        this.onConnectionCallback(from, peerConnection.connectionState);
                    }
                };

                // Handle incoming data channels
                peerConnection.ondatachannel = (event) => {
                    console.log(`Received data channel from ${from}`);
                    this.setupDataChannel(event.channel, from);
                    this.dataChannels[from] = event.channel;
                };
            }

            const peerConnection = this.peerConnections[from];

            // Set the remote description (offer)
            await peerConnection.setRemoteDescription(new RTCSessionDescription(offer));

            // Create and send an answer
            const answer = await peerConnection.createAnswer();
            await peerConnection.setLocalDescription(answer);

            this.socket.emit('webrtc_signal', {
                type: 'answer',
                from: this.localNodeId,
                to: from,
                signal: answer
            });
        } catch (error) {
            console.error('Error handling offer:', error);
            this.closePeerConnection(from);
        }
    }

    /**
     * Handle incoming WebRTC answer
     * @param {string} from - Node ID of the sender
     * @param {RTCSessionDescriptionInit} answer - WebRTC answer
     */
    async handleAnswer(from, answer) {
        try {
            const peerConnection = this.peerConnections[from];
            if (peerConnection) {
                await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
            }
        } catch (error) {
            console.error('Error handling answer:', error);
            this.closePeerConnection(from);
        }
    }

    /**
     * Handle incoming ICE candidate
     * @param {string} from - Node ID of the sender
     * @param {RTCIceCandidateInit} candidate - ICE candidate
     */
    async handleIceCandidate(from, candidate) {
        try {
            const peerConnection = this.peerConnections[from];
            if (peerConnection) {
                await peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
            }
        } catch (error) {
            console.error('Error handling ICE candidate:', error);
        }
    }

    /**
     * Configure a data channel for tensor transfer
     * @param {RTCDataChannel} dataChannel - The data channel to configure
     * @param {string} remoteNodeId - ID of the remote node
     */
    setupDataChannel(dataChannel, remoteNodeId) {
        dataChannel.binaryType = 'arraybuffer'; // Use arraybuffer for binary data

        dataChannel.onopen = () => {
            console.log(`Data channel with ${remoteNodeId} is open`);
            if (this.onConnectionCallback) {
                this.onConnectionCallback(remoteNodeId, 'datachannel-open');
            }
        };

        dataChannel.onclose = () => {
            console.log(`Data channel with ${remoteNodeId} is closed`);
            if (this.onConnectionCallback) {
                this.onConnectionCallback(remoteNodeId, 'datachannel-closed');
            }
        };

        dataChannel.onerror = (error) => {
            console.error(`Data channel error with ${remoteNodeId}:`, error);
            if (this.onConnectionCallback) {
                this.onConnectionCallback(remoteNodeId, 'datachannel-error');
            }
        };

        dataChannel.onmessage = (event) => {
            // Callback for tensor data
            if (this.onDataCallback) {
                this.onDataCallback(remoteNodeId, event.data);
            }
        };
    }

    /**
     * Send tensor data to a specific node
     * @param {string} remoteNodeId - ID of the recipient node
     * @param {ArrayBuffer} data - Binary tensor data
     * @param {Object} metadata - Metadata about the tensor (shape, dtype, etc.)
     * @returns {Promise<boolean>} - Whether the send was successful
     */
    async sendTensorData(remoteNodeId, data, metadata) {
        const dataChannel = this.dataChannels[remoteNodeId];

        if (!dataChannel || dataChannel.readyState !== 'open') {
            console.error(`Data channel to ${remoteNodeId} not open`);
            return false;
        }

        try {
            // Send metadata first as JSON
            dataChannel.send(JSON.stringify({
                type: 'tensor-metadata',
                metadata
            }));

            // Send the actual binary tensor data
            dataChannel.send(data);
            return true;
        } catch (error) {
            console.error(`Error sending tensor data to ${remoteNodeId}:`, error);
            return false;
        }
    }

    /**
     * Send a control message to a remote node
     * @param {string} remoteNodeId - ID of the recipient node
     * @param {string} type - Message type
     * @param {Object} payload - Message payload
     * @returns {Promise<boolean>} - Whether the send was successful
     */
    async sendControlMessage(remoteNodeId, type, payload) {
        const dataChannel = this.dataChannels[remoteNodeId];

        if (!dataChannel || dataChannel.readyState !== 'open') {
            console.error(`Data channel to ${remoteNodeId} not open`);
            return false;
        }

        try {
            dataChannel.send(JSON.stringify({
                type: `control-${type}`,
                payload
            }));
            return true;
        } catch (error) {
            console.error(`Error sending control message to ${remoteNodeId}:`, error);
            return false;
        }
    }

    /**
     * Set callback for received tensor data
     * @param {Function} callback - Function to call when tensor data is received
     */
    setDataCallback(callback) {
        this.onDataCallback = callback;
    }

    /**
     * Set callback for connection state changes
     * @param {Function} callback - Function to call when connection state changes
     */
    setConnectionCallback(callback) {
        this.onConnectionCallback = callback;
    }

    /**
     * Close a peer connection
     */
    closePeerConnection(nodeId) {
        console.log(`Closing peer connection with ${nodeId}`);

        if (this.dataChannels[nodeId]) {
            try {
                this.dataChannels[nodeId].close();
            } catch (e) {
                console.error(`Error closing data channel for ${nodeId}:`, e);
            }
            delete this.dataChannels[nodeId];
        }

        if (this.peerConnections[nodeId]) {
            try {
                this.peerConnections[nodeId].close();
            } catch (e) {
                console.error(`Error closing peer connection for ${nodeId}:`, e);
            }
            delete this.peerConnections[nodeId];
        }

        if (this.onConnectionCallback) {
            this.onConnectionCallback(nodeId, 'closed');
        }
    }

    /**
     * Close all peer connections
     */
    closeAllConnections() {
        Object.keys(this.peerConnections).forEach(nodeId => {
            this.closePeerConnection(nodeId);
        });

        console.log('Closed all WebRTC connections');
    }
}

export default WebRTCPeerConnection; 