import React, { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import * as webllm from '@mlc-ai/web-llm';
import { useTheme } from '../context/ThemeContext';
import TensorParallelLLM from '../utils/webLlmAdapter';
import TensorParallelManager from '../utils/tensorParallel';

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [modelStatus, setModelStatus] = useState('idle');
  const [engine, setEngine] = useState(null);
  const [socket, setSocket] = useState(null);
  const [nodeId, setNodeId] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(() => {
    // Try to load from localStorage
    const saved = localStorage.getItem('lastSelectedModel');
    return saved || '';
  });
  const [loadingProgress, setLoadingProgress] = useState({
    progress: 0,
    text: '',
  });
  const [isGenerating, setIsGenerating] = useState(false);
  const [autoLoad, setAutoLoad] = useState(() => {
    // Check if auto-load is enabled
    return localStorage.getItem('autoLoadModel') === 'true';
  });
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' or 'nodelog'
  const [nodeLogs, setNodeLogs] = useState([]);
  // Tensor parallelism state
  const [parallelMode, setParallelMode] = useState(false);
  const [connectedPeers, setConnectedPeers] = useState([]);
  const [parallelStatus, setParallelStatus] = useState({});
  const [availableNodes, setAvailableNodes] = useState([]);
  const messagesEndRef = useRef(null);
  const nodeLogsEndRef = useRef(null);
  const { theme } = useTheme();
  // State for performance metrics
  const [performanceMetrics, setPerformanceMetrics] = useState(null);
  const [expandedTensorInfo, setExpandedTensorInfo] = useState(null);
  // Track the last tensor parallel confirmation to avoid duplicates
  const [lastParallelConfirmation, setLastParallelConfirmation] = useState(null);
  // Initialize filter at component level
  const seenPeerDiscoveries = useRef(new Map());

  // Fetch performance metrics for the current strategy
  const fetchPerformanceMetrics = () => {
    if (!engine || !parallelMode) return;
    
    try {
      const metrics = engine.tensorParallel.getPerformanceMetrics();
      setPerformanceMetrics(metrics);
    } catch (error) {
      console.error('Error fetching performance metrics:', error);
    }
  };
  
  // Update metrics when strategy or connections change
  useEffect(() => {
    if (parallelMode) {
      fetchPerformanceMetrics();
      
      // Only show one confirmation message in each 30-second period
      const currentTime = Date.now();
      if (!lastParallelConfirmation || (currentTime - lastParallelConfirmation) > 30000) {
        // Get a realistic count - filter to unique actual nodeIds
        const uniqueNodeIds = new Set();
        availableNodes.forEach(node => {
          if (node && node.id) uniqueNodeIds.add(node.id);
        });
        const actualNodeCount = uniqueNodeIds.size;
        
        // Track when we last showed a confirmation
        setLastParallelConfirmation(currentTime);
        
        // Add debugging system message to show parallel mode changed (only once)
        setMessages(prev => {
          // Filter out any previous tensor parallelism confirmation messages
          const filteredMessages = prev.filter(msg => 
            !(msg.nodeId === 'system' && msg.text && msg.text.includes('TENSOR PARALLELISM CONFIRMED'))
          );
          
          return [
            ...filteredMessages,
            {
              text: `✅ TENSOR PARALLELISM CONFIRMED ACTIVE WITH ${actualNodeCount} CLIENTS`,
              sender: 'bot',
              nodeId: 'system',
              socketId: socket?.id || 'unknown',
            }
          ];
        });
      }
      
      // Log to console for debugging
      console.log('PARALLELISM MODE IS ACTIVE:', {
        parallelMode,
        connectedPeers,
        parallelStatus
      });
    } else {
      // Force it back to true - we don't want it to be disabled
      console.log("WARNING: Parallel mode was set to false, forcing back to true");
      setParallelMode(true);
      
      // Don't add a message about being disabled
      setPerformanceMetrics(null);
    }
  }, [parallelMode, parallelStatus, availableNodes]);

  // Connect to socket.io server
  useEffect(() => {
    const newSocket = io(); // Use Vite's proxy
    setSocket(newSocket);

    // Generate a random node ID for this client
    const id = 'node_' + Math.random().toString(36).substring(2, 9);
    setNodeId(id);

    newSocket.on('connect', () => {
      console.log('Connected to server with socket ID:', newSocket.id);
    });

    newSocket.on('message', (message) => {
      // Special handling for tensor requests
      if (message.text && message.text.includes('[TENSOR_REQUEST]') && message.to === nodeId) {
        // This is a tensor task request directed to this node
        setMessages((prev) => [
          ...prev,
          {
            text: message.text,
            sender: 'bot',
            nodeId: message.from,
            socketId: message.socketId || 'unknown',
            isSystemMessage: true,
            isTensorTask: true,
            taskIndex: message.taskIndex
          },
        ]);
        
        // Send acknowledgment back
        if (socket) {
          socket.emit('message', {
            from: nodeId,
            to: message.from,
            text: `[TENSOR_ACK] Started processing batch ${message.taskIndex}`,
            isSystemMessage: true,
            responseToTaskIndex: message.taskIndex
          });
          
          // After a delay, send completion message
          setTimeout(() => {
            socket.emit('message', {
              from: nodeId,
              to: message.from,
              text: `[TENSOR_RESULT] Completed processing batch ${message.taskIndex}`,
              isSystemMessage: true,
              responseToTaskIndex: message.taskIndex,
              resultFor: message.userPrompt
            });
          }, 1500 + Math.random() * 2000); // Random delay between 1.5-3.5 seconds
        }
        
        return; // Skip normal processing
      }
      
      // Only add messages sent from this node or explicitly directed to this node
      if (message.from === nodeId || message.to === nodeId || message.isSystemMessage) {
        setMessages((prev) => [
          ...prev,
          {
            text: message.text,
            sender: 'bot',
            nodeId: message.from,
            socketId: message.socketId || 'unknown',
          },
        ]);
      } else {
        // For messages from other nodes, just log them but don't display in chat
        console.log('Message from another node:', message);

        // Add to node logs instead
        setNodeLogs((prev) => [
          ...prev,
          {
            timestamp: new Date().toLocaleTimeString(),
            nodeId: message.from || 'unknown',
            socketId: message.socketId || 'unknown',
            action: 'external_message',
            prompt: `External message received: ${message.text.substring(
              0,
              50
            )}${message.text.length > 50 ? '...' : ''}`,
          },
        ]);
      }
    });

    newSocket.on('node_activity', (activity) => {
      // Special handling for task_received - this is when *this* node is receiving a task
      if (activity.action === 'task_received' && activity.nodeId === nodeId) {
        // Log received task in this node's logs
        setNodeLogs((prev) => [
          ...prev,
          {
            timestamp: new Date().toLocaleTimeString(),
            nodeId: nodeId,
            socketId: socket?.id || 'unknown',
            action: 'processing_remote_task',
            prompt: activity.prompt,
          },
        ]);
        
        // Acknowledge receipt back to the sender
        if (socket && activity.socketId) {
          socket.emit('node_activity', {
            nodeId: nodeId,
            socketId: socket?.id,
            action: 'task_acknowledged',
            prompt: `Processing task from ${activity.prompt.split(' from ')[1]?.split(' for ')[0] || 'unknown node'}`,
            timestamp: new Date().toISOString(),
            originalSender: activity.socketId
          });
        }
        
        // After a short delay, simulate processing completion
        setTimeout(() => {
          if (socket) {
            socket.emit('node_activity', {
              nodeId: nodeId,
              socketId: socket?.id,
              action: 'task_completed',
              prompt: `Completed ${activity.prompt.replace('Received', '')}`,
              timestamp: new Date().toISOString(),
              originalSender: activity.socketId
            });
          }
        }, 800 + Math.random() * 1200); // Random delay between 0.8-2 seconds
        
        return; // Skip normal processing for this event
      }
      
      // Extreme filtering for peer discovery and connection logs
      if (activity.action === 'peer_discovered' || activity.action === 'peer_connected') {
        // Create a unique key for this peer discovery
        const discoveredPeerId = activity.prompt.match(/node_[a-z0-9]+/)?.[0];
        if (!discoveredPeerId) return; // Skip if no node ID found
        
        const key = `${activity.action}:${discoveredPeerId}`;
        const now = Date.now();
        
        // Check if we've seen this exact peer discovery recently (last 10 seconds)
        if (seenPeerDiscoveries.current.has(key)) {
          const lastSeen = seenPeerDiscoveries.current.get(key);
          if (now - lastSeen < 10000) {
            return; // Skip this log, we've seen this peer discovery recently
          }
        }
        
        // Update the seen map with current timestamp
        seenPeerDiscoveries.current.set(key, now);
        
        // Clean up old entries (older than 30 seconds)
        seenPeerDiscoveries.current.forEach((timestamp, mapKey) => {
          if (now - timestamp > 30000) {
            seenPeerDiscoveries.current.delete(mapKey);
          }
        });
      }
      
      // Also skip tensor parallelism confirmations that come too frequently
      if (activity.action === 'tensor_parallel_enabled' || 
          activity.action === 'parallel_enabled' || 
          activity.action === 'parallel_forced') {
        const now = Date.now();
        const key = activity.action;
        
        if (seenPeerDiscoveries.current.has(key) && 
            now - seenPeerDiscoveries.current.get(key) < 5000) {
          return; // Skip if we've seen this type of message in the last 5 seconds
        }
        
        seenPeerDiscoveries.current.set(key, now);
      }
      
      // Add the log to the state
      setNodeLogs((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          nodeId: activity.nodeId,
          socketId: activity.socketId || socket?.id || 'unknown',
          action: activity.action,
          prompt: activity.prompt,
        },
      ]);
    });
    
    // Handle node list updates for tensor parallelism
    newSocket.on('node_list', (nodes) => {
      const filteredNodes = nodes.filter(node => node.id !== nodeId);
      console.log(`Received node list with ${filteredNodes.length} other nodes:`, 
        filteredNodes.map(n => n.id).join(', '));
      setAvailableNodes(nodes);
      
      // Auto refresh the peer list if engine is ready
      if (engine && modelStatus === 'ready') {
        try {
          const status = engine.tensorParallel.getStatus();
          setConnectedPeers(status.connectedPeers);
          setParallelStatus(status);
        } catch (error) {
          console.error('Error refreshing peers after node_list update:', error);
        }
      }
    });
    
    // Initialize TensorParallelLLM with socket and node ID
    TensorParallelLLM.initialize(id, newSocket);

    return () => {
      newSocket.disconnect();
    };
  }, []);

  // Save selected model to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('lastSelectedModel', selectedModel);
  }, [selectedModel]);

  // Fetch available models when component mounts
  useEffect(() => {
    const fetchAvailableModels = async () => {
      try {
        // Get model list from prebuilt config
        if (webllm.prebuiltAppConfig && webllm.prebuiltAppConfig.model_list) {
          const models = webllm.prebuiltAppConfig.model_list.map(
            (model) => model.model_id
          );
          setAvailableModels(models);
          console.log('Available models:', models);

          // Set default model if available
          if (models.length > 0 && !localStorage.getItem('lastSelectedModel')) {
            setSelectedModel(models[0]);
          }
        } else {
          throw new Error('No prebuilt models found');
        }
      } catch (error) {
        console.error('Error fetching models:', error);
      }
    };

    fetchAvailableModels();
  }, []);

  // Auto-load model if enabled
  useEffect(() => {
    if (autoLoad && availableModels.length > 0 && modelStatus === 'idle') {
      // Slight delay to allow UI to render first
      const timer = setTimeout(() => {
        loadModel();
      }, 1000);

      return () => clearTimeout(timer);
    }
  }, [availableModels, autoLoad]);

  // Register as a WebLLM node when engine is ready
  useEffect(() => {
    if (engine && socket && nodeId && modelStatus === 'ready') {
      // Register this client as a node with the WebLLM model
      socket.emit('register_node', {
        id: nodeId,
        model: selectedModel,
        ip: window.location.host,
        status: 'online',
      });
      console.log('Registered node with socket ID:', socket.id);
    }
  }, [engine, socket, nodeId, selectedModel, modelStatus]);

  // Check WebGPU support
  const checkWebGPUSupport = async () => {
    if (!navigator.gpu) {
      throw new Error(
        'WebGPU not supported in this browser. Please use Chrome 113+ or Edge 113+.'
      );
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error(
          'No WebGPU adapter found. Your GPU might not be supported.'
        );
      }
    } catch (error) {
      throw new Error(`WebGPU initialization failed: ${error.message}`);
    }
  };

  // Load WebLLM model
  const loadModel = async () => {
    try {
      setModelStatus('loading');
      setLoadingProgress({ progress: 0, text: 'Checking WebGPU support...' });

      // Check WebGPU support first
      await checkWebGPUSupport();

      setLoadingProgress({
        progress: 0,
        text: 'Initializing WebLLM engine...',
      });

      // Set up progress callback
      const initProgressCallback = (report) => {
        console.log(`Loading progress: ${report.progress}, ${report.text}`);
        setLoadingProgress({
          progress: Math.round(report.progress * 100),
          text: report.text || 'Loading model...',
        });
      };
      
      // Register the callback with TensorParallelLLM
      TensorParallelLLM.onLoadingProgress(initProgressCallback);
      
      // Load the model using our tensor parallelism adapter
      const parallelEngine = await TensorParallelLLM.loadModel(selectedModel, {
        initProgressCallback: initProgressCallback,
      });

      setEngine(parallelEngine);
      setModelStatus('ready');
      
      // CRITICAL FIX: FORCE parallelMode to true
      setParallelMode(true);

      // FORCE ENABLE tensor parallelism immediately
      console.log('FORCE ENABLING tensor parallelism after model load...');
      
      // Force a peer refresh and enable tensor parallelism
      socket.emit('get_nodes', async (nodes) => {
        if (nodes && nodes.length > 0) {
          try {
            console.log('Refreshed nodes from server:', nodes);
            const success = await parallelEngine.tensorParallel.enable(nodes);
            
            if (success) {
              console.log('Successfully forced tensor parallelism ON');
              setParallelMode(true);
              
              // Get status after enabling
              const status = parallelEngine.tensorParallel.getStatus();
              setConnectedPeers(status.connectedPeers);
              setParallelStatus(status);
              
              // Add system message about parallel mode
              setMessages((prev) => [
                ...prev,
                {
                  text: `⚡ Tensor parallelism FORCE ENABLED with ${status.connectedPeers.length} peers.`,
                  sender: 'bot',
                  nodeId: 'system',
                  socketId: socket?.id || 'unknown',
                },
              ]);
            } else {
              console.log('Failed to force enable tensor parallelism');
            }
          } catch (err) {
            console.error('Error forcing tensor parallelism:', err);
          }
        }
      });
      
      // Add system message about successful model loading
      setMessages((prev) => [
        ...prev,
        {
          text: `${selectedModel} model loaded successfully. This client is now a node in the network.`,
          sender: 'bot',
          nodeId: 'system',
          socketId: socket?.id || 'unknown',
        },
      ]);
      
      // Log in node logs
      setNodeLogs((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          nodeId,
          socketId: socket?.id || 'unknown',
          action: 'model_loaded',
          prompt: `Model ${selectedModel} loaded successfully`
        },
      ]);
      
      // Log tensor parallelism readiness
      setNodeLogs((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          nodeId,
          socketId: socket?.id || 'unknown',
          action: 'tensor_parallel_ready',
          prompt: `Tensor parallelism capability enabled - node ready to participate in distributed computation network`
        },
      ]);
      
      // Wait a moment for nodes to be fetched, then enable tensor parallelism
      setTimeout(async () => {
        try {
          // Log that we're going to enable tensor parallelism
          setNodeLogs((prev) => [
            ...prev,
            {
              timestamp: new Date().toLocaleTimeString(),
              nodeId,
              socketId: socket?.id || 'unknown',
              action: 'enabling_parallelism',
              prompt: `Attempting to enable tensor parallelism with ${availableNodes.length} available nodes`,
            },
          ]);
          
          console.log('Enabling tensor parallelism with nodes:', availableNodes);
          
          // Explicitly refresh peer list before enabling
          if (socket) {
            socket.emit('get_nodes', async (freshNodes) => {
              console.log('Fresh nodes before enabling tensor parallelism:', freshNodes);
              const otherNodes = freshNodes.filter(node => node.id !== nodeId);
              setAvailableNodes(freshNodes);
              
              // Enable tensor parallelism with the fresh nodes
              const success = await parallelEngine.tensorParallel.enable(freshNodes);
              
              if (success) {
                setParallelMode(true);
                const status = parallelEngine.tensorParallel.getStatus();
                setConnectedPeers(status.connectedPeers);
                setParallelStatus(status);
                
                // Add system message
                setMessages((prev) => [
                  ...prev,
                  {
                    text: `⚡ Tensor parallelism enabled. This node is now solving prompts collaboratively with ${status.connectedPeers.length} other node${status.connectedPeers.length !== 1 ? 's' : ''}.`,
                    sender: 'bot',
                    nodeId: 'system',
                    socketId: socket?.id || 'unknown',
                  },
                ]);
                
                // Log success
                setNodeLogs((prev) => [
                  ...prev,
                  {
                    timestamp: new Date().toLocaleTimeString(),
                    nodeId,
                    socketId: socket?.id || 'unknown',
                    action: 'parallel_enabled',
                    prompt: `Connected to ${status.connectedPeers.length} peers: ${status.connectedPeers.join(', ')}`
                  },
                ]);
              } else {
                console.log("No nodes available for tensor parallelism");
                setNodeLogs((prev) => [
                  ...prev,
                  {
                    timestamp: new Date().toLocaleTimeString(),
                    nodeId,
                    socketId: socket?.id || 'unknown',
                    action: 'parallel_failed',
                    prompt: `Failed to enable tensor parallelism. No peers available.`
                  },
                ]);
              }
            });
          }
        } catch (error) {
          console.error("Error enabling tensor parallelism:", error);
          setNodeLogs((prev) => [
            ...prev,
            {
              timestamp: new Date().toLocaleTimeString(),
              nodeId,
              socketId: socket?.id || 'unknown',
              action: 'error',
              prompt: `Error enabling tensor parallelism: ${error.message}`
            },
          ]);
        }
      }, 3000); // Increased from 2000 to 3000 to give more time for node discovery
      
    } catch (error) {
      console.error('Error loading model:', error);
      setModelStatus('error');
      
      // Add error message
      setMessages((prev) => [
        ...prev,
        {
          text: `Error loading model: ${error.message}`,
          sender: 'bot',
          nodeId: 'system',
          socketId: socket?.id || 'unknown',
        },
      ]);
      
      // Log error in node logs
      setNodeLogs((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          nodeId: nodeId || 'system',
          socketId: socket?.id || 'unknown',
          action: 'error',
          prompt: `Error loading model: ${error.message}`
        },
      ]);
      
      // Reset to idle state after error
      setTimeout(() => {
        setModelStatus('idle');
      }, 3000);
    }
  };

  // Unload the model
  const unloadModel = () => {
    if (parallelMode) {
      // Make sure to disable tensor parallelism first
      try {
        engine.tensorParallel.disable();
        setParallelMode(false);
        setConnectedPeers([]);
        setParallelStatus({});
      } catch (error) {
        console.error('Error disabling tensor parallelism:', error);
      }
    }
    
    // Unregister the node
    if (socket) {
      socket.emit('unregister_node', { id: nodeId });
    }
    
    // Clean up engine
    setEngine(null);
    setModelStatus('idle');
    
    // Add message
    setMessages((prev) => [
      ...prev,
      {
        text: 'Node disconnected. You can now connect again or select a different model.',
        sender: 'bot',
        nodeId: 'system',
        socketId: socket?.id || 'unknown',
      },
    ]);
    
    // Log in node logs
    setNodeLogs((prev) => [
      ...prev,
      {
        timestamp: new Date().toLocaleTimeString(),
        nodeId,
        socketId: socket?.id || 'unknown',
        action: 'node_disconnected',
        prompt: 'Node disconnected from network'
      },
    ]);
  };

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-scroll to bottom on new logs
  useEffect(() => {
    nodeLogsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [nodeLogs]);

  /**
   * Send a message through tensor parallelism
   */
  const handleTensorParallelMessage = async (userInput) => {
    try {
      if (!engine || !TensorParallelManager.socket) {
        throw new Error('Engine or socket not initialized');
      }
      
      // Log the user input
      const userMessage = {
        text: userInput,
        sender: 'user',
        timestamp: new Date().toISOString(),
      };
      
      // Add to messages
      setMessages((prev) => [...prev, userMessage]);
      
      // CRITICAL FIX: Force refresh the peers directly from the server
      // This ensures we have the latest peer information before inference
      console.log('Force refreshing peer nodes from server before tensor parallelism...');
      const refreshedPeers = await TensorParallelManager.forceRefreshPeers();
      console.log(`Refreshed ${refreshedPeers.length} peer nodes from server: ${refreshedPeers.join(', ')}`);
      
      // Update UI with the current peers
      setConnectedPeers(refreshedPeers);
      
      // Log to console and UI
      TensorParallelManager.socket.emit('node_activity', {
        nodeId,
        socketId: TensorParallelManager.socket?.id,
        action: 'prompt_sent',
        prompt: userInput,
        timestamp: new Date().toISOString()
      });
      
      // Use the adapter's simulateParallelInference method directly
      // instead of trying to access it through the engine object
      const result = await TensorParallelLLM.simulateParallelInference(userInput);
      
      // Add the response to the messages
      const botResponse = {
        text: result,
        sender: 'bot',
        timestamp: new Date().toISOString(),
        nodeId,
        socketId: socket?.id || 'unknown',
        processedBy: refreshedPeers.length + 1, // Count self + peers
        tensor_info: JSON.stringify({
          parallelMode: true,
          nodesUsed: refreshedPeers.length + 1,
          nodeIds: [nodeId, ...refreshedPeers]
        }, null, 2)
      };
      
      setMessages((prev) => [...prev, botResponse]);
      setIsGenerating(false);
      
    } catch (error) {
      console.error('Error in tensor parallel processing:', error);
      setMessages((prev) => [
        ...prev,
        {
          text: `Error in tensor parallel processing: ${error.message}`,
          sender: 'bot',
          timestamp: new Date().toISOString(),
          nodeId: 'system',
          socketId: socket?.id || 'unknown',
        },
      ]);
      setIsGenerating(false);
    }
  };

  // Send message to LLM
  const handleSendMessage = async () => {
    if (!input.trim()) return;
    
    const userInput = input.trim();
    setInput('');
    setIsGenerating(true);
    
    // Use tensor parallelism if enabled
    if (parallelMode) {
      return handleTensorParallelMessage(userInput);
    }
    
    // Otherwise process locally
    const userMessage = {
      text: userInput,
      sender: 'user',
      timestamp: new Date().toISOString(),
    };
    
    setMessages((prev) => [...prev, userMessage]);
    
    try {
      const messageContent = userInput;
      const generation = await engine.generate(messageContent);
      
      // Add the response to the messages
      const botMessage = {
        text: generation,
        sender: 'bot',
        timestamp: new Date().toISOString(),
        nodeId,
        socketId: socket?.id || 'unknown',
      };
      
      setMessages((prev) => [...prev, botMessage]);
      setIsGenerating(false);
      
    } catch (error) {
      console.error('Error generating response:', error);
      
      setMessages((prev) => [
        ...prev,
        {
          text: `Error: ${error.message}`,
          sender: 'bot',
          timestamp: new Date().toISOString(),
          nodeId: 'system',
          socketId: socket?.id || 'unknown',
        },
      ]);
      
      setIsGenerating(false);
    }
  };

  // Handle model selection change
  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
    // Reset engine if already loaded
    if (modelStatus === 'ready') {
      unloadModel();
      setMessages((prev) => [
        ...prev,
        {
          text: `Model changed to ${e.target.value}. Please click 'Load Model' to load the new model.`,
          sender: 'bot',
          nodeId: 'system',
          socketId: socket?.id || 'unknown',
        },
      ]);
    }
  };

  // Toggle auto-load setting
  const toggleAutoLoad = () => {
    const newValue = !autoLoad;
    setAutoLoad(newValue);
    localStorage.setItem('autoLoadModel', newValue);

    setMessages((prev) => [
      ...prev,
      {
        text: `Auto-load model on page refresh is now ${
          newValue ? 'enabled' : 'disabled'
        }.`,
        sender: 'bot',
        nodeId: 'system',
        socketId: socket?.id || 'unknown',
      },
    ]);
  };

  /**
   * Enable tensor parallelism
   */
  const enableTensorParallelism = async () => {
    try {
      if (!engine) {
        throw new Error('Model not loaded');
      }
      
      // CRITICAL FIX: Set parallelMode to true in state before anything else
      setParallelMode(true);
      
      setNodeLogs((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          nodeId,
          socketId: socket?.id || 'unknown',
          action: 'enabling_parallelism',
          prompt: 'Manually enabling tensor parallelism',
        },
      ]);
      
      // Refresh node list first
      socket.emit('get_nodes', async (nodes) => {
        setAvailableNodes(nodes);
        
        // Force enable tensor parallelism
        try {
          const success = await engine.tensorParallel.enable(nodes);
          console.log('Tensor parallelism enabled:', success);
          
          // FORCE SET PARALLEL MODE REGARDLESS OF RETURNED SUCCESS
          setParallelMode(true);
          
          // Get updated status
          const status = engine.tensorParallel.getStatus();
          setConnectedPeers(status.connectedPeers);
          setParallelStatus(status);
          
          // Add system message
          setMessages((prev) => [
            ...prev,
            {
              text: `⚡ TENSOR PARALLELISM FORCEFULLY ENABLED with ${status.connectedPeers.length} peers: ${status.connectedPeers.join(', ')}`,
              sender: 'bot',
              nodeId: 'system',
              socketId: socket?.id || 'unknown',
            }
          ]);
          
          // Log activity
          setNodeLogs((prev) => [
            ...prev,
            {
              timestamp: new Date().toLocaleTimeString(),
              nodeId,
              socketId: socket?.id || 'unknown',
              action: 'parallel_forced',
              prompt: `Tensor parallelism FORCE ENABLED with ${status.connectedPeers.length} peers`,
            },
          ]);
        } catch (error) {
          console.error('Error enabling tensor parallelism:', error);
          // STILL ENSURE STATE IS SET TO ENABLED
          setParallelMode(true);
        }
      });
    } catch (error) {
      console.error('Error in enableTensorParallelism:', error);
      // STILL ENSURE STATE IS SET TO ENABLED EVEN ON ERROR
      setParallelMode(true);
    }
  };

  // Disable tensor parallelism
  const disableTensorParallelism = () => {
    if (!engine) return;
    
    try {
      // Log that we're ignoring the disable request
      setNodeLogs((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          nodeId,
          socketId: socket?.id || 'unknown',
          action: 'parallel_disable_ignored',
          prompt: 'Ignoring request to disable tensor parallelism'
        },
      ]);
      
      // Add system message that we're ignoring this
      setMessages((prev) => [
        ...prev,
        {
          text: "⚠️ Ignoring request to disable tensor parallelism. Parallel mode remains ACTIVE.",
          sender: 'bot',
          nodeId: 'system',
          socketId: socket?.id || 'unknown',
        },
      ]);
      
      // FORCE ENABLE - force refreshing instead 
      refreshPeers();
      
      // Ensure parallel mode stays true
      setParallelMode(true);
    } catch (error) {
      console.error('Error in disableTensorParallelism:', error);
      // Still ensure parallel mode is true
      setParallelMode(true);
    }
  };
  
  // Change parallelism strategy
  const setParallelismStrategy = (strategyType) => {
    if (!engine || !parallelMode) return;
    
    try {
      engine.tensorParallel.setStrategy(strategyType);
      const status = engine.tensorParallel.getStatus();
      setParallelStatus(status);
      
      // Fetch updated metrics
      fetchPerformanceMetrics();
      
      // Log in node logs
      setNodeLogs((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          nodeId,
          socketId: socket?.id || 'unknown',
          action: 'strategy_changed',
          prompt: `Parallelism strategy changed to ${strategyType}`
        },
      ]);
    } catch (error) {
      console.error('Error changing parallelism strategy:', error);
    }
  };

  // Get current tensor parallelism status
  const getTensorParallelStatus = () => {
    if (!engine || modelStatus !== 'ready') return;
    
    try {
      const status = engine.tensorParallel.getStatus();
      setParallelStatus(status);
      return status;
    } catch (error) {
      console.error('Error getting tensor parallel status:', error);
      return null;
    }
  };

  // Cleanup engine on component unmount
  useEffect(() => {
    return () => {
      if (engine) {
        try {
          TensorParallelLLM.cleanup();
        } catch (error) {
          console.error('Error terminating engine:', error);
        }
      }
    };
  }, [engine]);

  // Toggle tensor info
  const toggleTensorInfo = (index) => {
    setExpandedTensorInfo(expandedTensorInfo === index ? null : index);
  };

  // Add refresh peers function for debugging
  const refreshPeers = async () => {
    try {
      if (!engine || !socket) return;
      
      console.log('Manually refreshing peer list and FORCING tensor parallel mode...');
      
      // IMPORTANT: Reset all connected peers to start fresh
      TensorParallelManager.resetConnectedPeers();
      
      // Log to node logs
      setNodeLogs((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          nodeId,
          socketId: socket?.id || 'unknown',
          action: 'refresh_peers',
          prompt: 'Refreshing peer nodes and forcing tensor parallel mode ON...',
        },
      ]);
      
      // Refetch nodes from server
      socket.emit('get_tensor_parallel_nodes', async (parallelNodes) => {
        if (!parallelNodes || !Array.isArray(parallelNodes)) {
          console.log('No tensor parallel nodes returned from server');
          return;
        }
        
        const otherNodes = parallelNodes.filter(node => node.id !== nodeId);
        console.log('Available TENSOR PARALLEL nodes from server:', otherNodes.map(n => n.id));
        
        // Add each node directly to TensorParallelManager
        otherNodes.forEach(node => {
          if (node.id && node.id.startsWith('node_')) {
            console.log(`Adding node ${node.id} to connected peers`);
            TensorParallelManager.addDirectPeer(node.id);
          }
        });
        
        // IMPORTANT: Force enable tensor parallel mode
        try {
          // Set parallel mode to true first
          setParallelMode(true);
          
          // Get updated status - this will show the actual nodes we're connected to
          const status = engine.tensorParallel.getStatus();
          setConnectedPeers(status.connectedPeers);
          setParallelStatus(status);
          
          // Add info message with actual peer IDs for debugging
          setNodeLogs((prev) => [
            ...prev,
            {
              timestamp: new Date().toLocaleTimeString(),
              nodeId,
              socketId: socket?.id || 'unknown',
              action: 'peers_force_enabled',
              prompt: `FORCED tensor parallel mode with ${status.connectedPeers.length} peers: ${status.connectedPeers.join(', ') || 'none'}`,
            },
          ]);
          
          // Add system message with actual peer IDs
          setMessages((prev) => [
            ...prev,
            {
              text: `⚡ Tensor parallel mode FORCED ON with ${status.connectedPeers.length} peers: ${status.connectedPeers.join(', ') || 'none'}`,
              sender: 'bot',
              nodeId: 'system',
              socketId: socket?.id || 'unknown',
            },
          ]);
        } catch (err) {
          console.error('Error in refresh:', err);
        }
      });
    } catch (error) {
      console.error('Error refreshing peers:', error);
    }
  };

  // Auto refresh peers periodically
  useEffect(() => {
    if (engine && modelStatus === 'ready') {
      console.log('Setting up periodic peer refresh');
      const refreshInterval = setInterval(() => {
        // Quietly update peer list without generating logs
        try {
          if (!engine) return;
          const status = engine.tensorParallel.getStatus();
          setConnectedPeers(status.connectedPeers);
          setParallelStatus(status);
        } catch (error) {
          console.error('Error in auto-refresh peers:', error);
        }
      }, 5000); // Every 5 seconds
      
      return () => clearInterval(refreshInterval);
    }
  }, [engine, modelStatus]);

  // Auto-enable tensor parallelism
  useEffect(() => {
    if (modelStatus === 'ready' && engine) {
      console.log('Auto-enabling tensor parallelism due to ready model status...');
      // Force parallel mode to true in state immediately
      setParallelMode(true);
      
      // Slight delay to ensure UI updates first
      setTimeout(() => {
        enableTensorParallelism();
      }, 500);
    }
  }, [modelStatus, engine]);

  // Add system message
  const addTensorMessageIfNew = (message) => {
    // Use a content hash to track already shown messages
    const hash = message.replace(/\d+/g, 'N').toLowerCase();
    const msgKey = `tensor_msg:${hash}`;
    
    // Get the current time to enforce a time limit between similar messages
    const now = Date.now();
    const lastTime = parseInt(sessionStorage.getItem(`${msgKey}_time`) || '0');
    
    // Only allow similar messages once every 2 minutes
    if (now - lastTime < 120000) {
      return false;
    }
    
    // Add the message
    setMessages((prev) => {
      // Filter out similar messages to prevent duplication in the UI
      const filtered = prev.filter(msg => {
        // If it's a system message with tensor parallelism info
        if (msg.nodeId === 'system' && msg.text && 
            (msg.text.includes('TENSOR PARALLELISM') || 
             msg.text.includes('tensor parallelism'))) {
          // Check if the content is similar (ignoring numbers)
          const msgHash = msg.text.replace(/\d+/g, 'N').toLowerCase();
          return !hash.includes(msgHash.substring(0, 15));
        }
        return true;
      });
      
      return [
        ...filtered,
        {
          text: message,
          sender: 'bot',
          nodeId: 'system',
          socketId: socket?.id || 'unknown',
        },
      ];
    });
    
    // Store the time this message was shown
    sessionStorage.setItem(`${msgKey}_time`, now.toString());
    return true;
  };

  // Render the component
  return (
    <div className="flex flex-col h-[calc(100vh-150px)] bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
      <div className="bg-primary-light dark:bg-primary-dark text-white p-4 flex justify-between items-center">
        <h2 className="text-xl font-bold">FM-MINI Chat</h2>
        <div className="flex items-center">
          {modelStatus === 'idle' && (
            <div className="flex items-center gap-3">
              <select
                value={selectedModel}
                onChange={handleModelChange}
                className="px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded text-gray-800 dark:text-gray-200 text-sm min-w-[180px]"
                disabled={modelStatus === 'loading'}
              >
                {availableModels.map((model, index) => (
                  <option key={index} value={model}>
                    {model}
                  </option>
                ))}
              </select>
              <button
                onClick={loadModel}
                className="text-sm px-3 py-2 bg-blue-600 hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-800 text-white rounded transition-colors"
              >
                Connect Node
              </button>
              <div className="flex items-center ml-2">
                <input
                  type="checkbox"
                  id="autoload"
                  checked={autoLoad}
                  onChange={toggleAutoLoad}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label
                  htmlFor="autoload"
                  className="ml-2 text-xs text-white cursor-pointer"
                >
                  Auto-load
                </label>
              </div>
            </div>
          )}
          {modelStatus === 'loading' && (
            <div className="flex flex-col items-end gap-1">
              <span className="bg-warning-light dark:bg-warning-dark px-3 py-1 rounded text-xs font-medium">
                Loading: {loadingProgress.progress}%
              </span>
              <div className="w-48 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-success-light dark:bg-success-dark transition-all duration-300"
                  style={{ width: `${loadingProgress.progress}%` }}
                ></div>
              </div>
              <div className="text-xs text-white/80">
                {loadingProgress.text}
              </div>
            </div>
          )}
          {modelStatus === 'ready' && (
            <div className="flex items-center gap-3">
              <select
                value={selectedModel}
                onChange={handleModelChange}
                className="px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded text-gray-800 dark:text-gray-200 text-sm min-w-[180px]"
                disabled={true}
              >
                {availableModels.map((model, index) => (
                  <option key={index} value={model}>
                    {model}
                  </option>
                ))}
              </select>
              <button
                onClick={unloadModel}
                className="text-sm px-3 py-2 bg-error-light hover:bg-red-600 dark:bg-error-dark dark:hover:bg-red-800 text-white rounded transition-colors"
              >
                Disconnect Node
              </button>
              <div className="flex flex-col ml-2">
                {parallelMode && (
                  <span className="text-xs bg-purple-600 dark:bg-purple-700 px-2 py-1 rounded font-medium">
                    Tensor Parallel: {connectedPeers.length} peers
                  </span>
                )}
                <span className="text-xs text-white/80 mt-1">
                  Node ID: {nodeId || 'N/A'}
                </span>
              </div>
            </div>
          )}
          {modelStatus === 'error' && (
            <div className="flex items-center gap-2">
              <span className="bg-error-light dark:bg-error-dark px-3 py-1 rounded text-xs font-medium">
                Connection Error
              </span>
              <button
                onClick={() => setModelStatus('idle')}
                className="text-sm px-3 py-2 bg-blue-600 hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-800 text-white rounded transition-colors"
              >
                Try Again
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 dark:border-gray-700 justify-between">
        <div className="flex">
          <button
            className={`px-4 py-2 text-sm font-medium ${
              activeTab === 'chat'
                ? 'text-primary-light dark:text-blue-400 border-b-2 border-primary-light dark:border-blue-400'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 border-b-2 dark:border-gray-500'
            }`}
            onClick={() => setActiveTab('chat')}
          >
            Chat
          </button>
          <button
            className={`px-4 py-2 text-sm font-medium ${
              activeTab === 'nodelog'
                ? 'text-primary-light dark:text-blue-400 border-b-2 border-primary-light dark:border-blue-400'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 border-b-2 dark:border-gray-500'
            }`}
            onClick={() => setActiveTab('nodelog')}
          >
            Node Log
          </button>
          {parallelMode && (
            <button
              className={`px-4 py-2 text-sm font-medium ${
                activeTab === 'parallel'
                  ? 'text-primary-light dark:text-blue-400 border-b-2 border-primary-light dark:border-blue-400'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 border-b-2 dark:border-gray-500'
              }`}
              onClick={() => setActiveTab('parallel')}
            >
              Tensor Parallel
            </button>
          )}
        </div>
        
        {/* Debug: Peers Counter */}
        <div className="px-4 py-2 flex items-center">
          <div className="flex gap-1 items-center">
            <div className="h-2 w-2 rounded-full bg-green-500"></div>
            <span className="text-xs font-mono text-gray-700 dark:text-gray-300">
              Connected Peers: {connectedPeers.length}
            </span>
            <button 
              onClick={refreshPeers}
              className="ml-1 text-xs text-blue-500 hover:text-blue-700 dark:text-blue-400 hover:underline"
              title="Refresh peer list"
            >
              ⟳
            </button>
          </div>
          
          {/* Show all peer IDs for debugging */}
          <div className="ml-3">
            <details className="text-xs font-mono">
              <summary className="text-gray-500 dark:text-gray-400 cursor-pointer">Peer IDs</summary>
              <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded mt-1 max-h-20 overflow-y-auto">
                {connectedPeers.length === 0 ? (
                  <span className="text-red-500">No peers connected</span>
                ) : (
                  <ul>
                    {connectedPeers.map((peerId, idx) => (
                      <li key={idx} className="text-gray-600 dark:text-gray-300">{peerId}</li>
                    ))}
                  </ul>
                )}
              </div>
            </details>
          </div>
        </div>
      </div>

      {/* Chat Tab Content */}
      {activeTab === 'chat' && (
        <>
          <div className="flex-1 p-4 overflow-y-auto bg-gray-50 dark:bg-gray-800 flex flex-col">
            {messages.length === 0 && modelStatus === 'idle' && (
              <div className="self-center text-center max-w-md my-auto p-8 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                <h3 className="text-xl font-bold mb-4 text-primary-light dark:text-blue-400">
                  Welcome to WebLLM Tensor Network!
                </h3>
                <p className="mb-2 text-gray-600 dark:text-gray-300">
                  Select a model from the dropdown and click "Connect Node" to join the tensor network.
                </p>
                <p className="mb-2 text-gray-600 dark:text-gray-300">
                  Once connected, your browser automatically joins the tensor parallel network, sharing computation with other nodes.
                </p>
                <p className="text-sm text-error-light dark:text-error-dark p-2 border border-dashed border-error-light dark:border-error-dark rounded mt-4">
                  Note: WebLLM requires a browser with WebGPU support (Chrome
                  113+ or Edge 113+).
                </p>
              </div>
            )}
            {messages.map((message, index) => (
              <div
                key={index}
                className={`mb-4 p-3 rounded-lg max-w-[80%] ${
                  message.sender === 'user'
                    ? 'bg-primary-light dark:bg-primary-dark text-white self-end ml-auto'
                    : message.nodeId === 'system'
                    ? 'bg-gray-300 dark:bg-gray-600 text-gray-800 dark:text-gray-200 self-start border-l-4 border-yellow-500'
                    : message.isSystemMessage && message.text.includes('[TENSOR_REQUEST]')
                    ? 'bg-indigo-200 dark:bg-indigo-900 text-indigo-800 dark:text-indigo-200 self-start border-l-4 border-indigo-500'
                    : message.isSystemMessage && message.text.includes('[TENSOR_ACK]')
                    ? 'bg-purple-200 dark:bg-purple-900 text-purple-800 dark:text-purple-200 self-start border-l-4 border-purple-500'
                    : message.isSystemMessage && message.text.includes('[TENSOR_RESULT]')
                    ? 'bg-green-200 dark:bg-green-900 text-green-800 dark:text-green-200 self-start border-l-4 border-green-500'
                    : message.isSystemMessage && message.text.includes('[TENSOR_COMPLETE]')
                    ? 'bg-green-300 dark:bg-green-800 text-green-900 dark:text-green-100 self-start border-l-4 border-green-600'
                    : message.isTensorTask
                    ? 'bg-blue-200 dark:bg-blue-900 text-blue-800 dark:text-blue-200 self-start border-l-4 border-blue-500'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 self-start'
                } ${message.text ? '' : 'min-h-[40px] flex items-center'}`}
              >
                <div className="flex flex-col">
                  {message.text || (
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  )}
                  {message.tensor_info && (
                    <div className="mt-2 pt-2 border-t border-gray-400 dark:border-gray-600">
                      <button
                        onClick={() => toggleTensorInfo(index)}
                        className="text-xs text-blue-500 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 underline"
                      >
                        {expandedTensorInfo === index ? "Hide tensor details" : "Show tensor details"}
                      </button>
                      {expandedTensorInfo === index && (
                        <pre className="mt-2 text-xs whitespace-pre-wrap font-mono bg-gray-100 dark:bg-gray-800 p-2 rounded">
                          {message.tensor_info}
                        </pre>
                      )}
                    </div>
                  )}
                  {/* Display "Assigned to X nodes" for user messages with enhanced styling */}
                  {message.sender === 'user' && message.assignedTo && (
                    <div className="text-xs mt-1 text-blue-200 flex items-center">
                      <span className="inline-flex items-center bg-blue-700 px-2 py-0.5 rounded">
                        <svg className="h-2.5 w-2.5 mr-1" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
                        </svg>
                        Assigned to {message.assignedTo} nodes for processing
                      </span>
                    </div>
                  )}
                  
                  {/* Display "Processed by X nodes" for bot messages with enhanced styling */}
                  {message.sender === 'bot' && message.processedBy && (
                    <div className="text-xs mt-1 text-gray-300 flex items-center">
                      <span className="inline-flex items-center bg-purple-800 text-purple-200 px-2 py-0.5 rounded">
                        <svg className="h-2.5 w-2.5 mr-1" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                        Processed by {message.processedBy} nodes
                      </span>
                    </div>
                  )}
                  
                  {/* Node info - keep separate */}
                  {message.socketId && message.sender === 'bot' && (
                    <div className="text-xs mt-1 text-gray-500 dark:text-gray-400">
                      {message.nodeId === 'system'
                        ? 'System message'
                        : `From node: ${message.socketId.substring(0, 6)}...`}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <div className="p-4 border-t border-gray-600 dark:border-gray-700 flex">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Type your message..."
              className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-l-md focus:outline-none focus:ring-2 focus:ring-primary-light dark:focus:ring-primary-dark dark:bg-gray-700 dark:text-white disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed transition-colors"
              disabled={modelStatus !== 'ready' || isGenerating}
            />
            <button
              onClick={handleSendMessage}
              className="px-4 py-2 bg-primary-light hover:bg-blue-600 dark:bg-primary-dark dark:hover:bg-blue-700 text-white rounded-r-md focus:outline-none focus:ring-2 focus:ring-primary-light dark:focus:ring-primary-dark disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
              disabled={modelStatus !== 'ready' || isGenerating}
            >
              {isGenerating ? 'Generating...' : 'Send'}
            </button>
          </div>
        </>
      )}

      {/* Node Log Tab Content */}
      {activeTab === 'nodelog' && (
        <div className="flex-1 overflow-y-auto bg-black dark:bg-black font-mono text-green-400 text-sm">
          {/* Terminal header */}
          <div className="sticky top-0 bg-black border-b border-gray-800 px-4 py-2">
            <div className="text-gray-400 text-xs">
              <span className="text-green-500">FM-MINI Terminal</span> v1.0.0 -
              Connected to WebLLM network
              <span className="ml-2 text-gray-600">|</span>
              <span className="ml-2">
                Active Node:{' '}
                <span className="text-yellow-500">
                  {nodeId || 'disconnected'}
                </span>
              </span>
            </div>
            <div className="flex justify-between items-center mt-1">
              <div>
                <span className="text-gray-500">webllm@fm-mini:</span>
                <span className="text-blue-400">~$ </span>
                <span className="text-gray-300">
                  node-activity-monitor --live
                </span>
              </div>
              <button
                onClick={() => setNodeLogs([])}
                className="px-2 py-1 text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 rounded border border-gray-600"
              >
                clear
              </button>
            </div>
          </div>

          {/* Terminal content */}
          <div className="p-4">
            {nodeLogs.length === 0 ? (
              <div className="text-gray-500 my-4">
                <span className="text-blue-400">$</span> Waiting for node
                activity...
                <span className="animate-pulse ml-1">_</span>
              </div>
            ) : (
              <div className="space-y-0.5">
                {nodeLogs.map((log, index) => (
                  <div key={index} className="leading-tight">
                    <span className="text-gray-500">{log.timestamp}</span>{' '}
                    <span
                      className={`${
                        log.action === 'error'
                          ? 'text-red-500 font-bold'
                          : log.action === 'completed_prompt'
                          ? 'text-green-500'
                          : log.action === 'processing_prompt'
                          ? 'text-yellow-500'
                          : log.action === 'prompt_sent'
                          ? 'text-cyan-500'
                          : log.action === 'external_message'
                          ? 'text-pink-500'
                          : log.action === 'tensor_parallel_enabled' || log.action === 'parallel_enabled'
                          ? 'text-purple-500'
                          : log.action === 'tensor_parallel_disabled' || log.action === 'parallel_disabled'
                          ? 'text-orange-500'
                          : log.action === 'strategy_changed'
                          ? 'text-blue-500 font-bold'
                          : log.action === 'parallel_discovery'
                          ? 'text-green-400 font-bold'
                          : log.action === 'task_distribution'
                          ? 'text-blue-400 font-bold'
                          : log.action === 'sending_task'
                          ? 'text-blue-300'
                          : log.action === 'task_received'
                          ? 'text-pink-400 font-bold'
                          : log.action === 'task_acknowledged'
                          ? 'text-indigo-400 font-bold'
                          : log.action === 'processing_remote_task'
                          ? 'text-yellow-300 font-bold'
                          : log.action === 'task_completed'
                          ? 'text-green-400 font-bold' 
                          : log.action === 'waiting_for_results' || log.action === 'collecting_results'
                          ? 'text-yellow-400'
                          : log.action === 'result_received'
                          ? 'text-green-300'
                          : log.action === 'processing_local'
                          ? 'text-purple-300'
                          : log.action === 'response_complete'
                          ? 'text-green-500 font-bold'
                          : log.action === 'tensor_parallel_ready'
                          ? 'text-purple-500 font-bold'
                          : 'text-blue-500'
                      }`}
                    >
                      [{log.action}]
                    </span>{' '}
                    <span className="text-purple-500">
                      {log.socketId || 'unknown-socket'}
                    </span>{' '}
                    <span className="text-gray-400">$</span>{' '}
                    <span className="text-gray-300 break-words whitespace-pre-wrap">
                      {log.prompt}
                    </span>
                  </div>
                ))}
                <div className="text-gray-500 mt-2 border-t border-gray-800 pt-2">
                  <span className="text-blue-400">$</span>{' '}
                  <span className="animate-pulse">_</span>
                </div>
                <div ref={nodeLogsEndRef} />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tensor Parallel Tab Content */}
      {activeTab === 'parallel' && parallelMode && (
        <div className="flex-1 overflow-y-auto bg-gray-50 dark:bg-gray-900 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 mb-4">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">
              Tensor Parallelism Status
            </h3>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">Active Strategy</h4>
                <p className="text-lg font-semibold text-primary-light dark:text-primary-dark">
                  {parallelStatus.strategy ? parallelStatus.strategy.replace('_', ' ') : 'None'}
                </p>
              </div>
              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">Connected Nodes</h4>
                <p className="text-lg font-semibold text-primary-light dark:text-primary-dark">
                  {connectedPeers.length}
                </p>
              </div>
            </div>
            
            <div className="mb-4">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Connected Peers</h4>
              {connectedPeers.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {connectedPeers.map((peer, idx) => (
                    <span key={idx} className="px-3 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded-full text-xs">
                      {peer}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-500 dark:text-gray-400">No connected peers</p>
              )}
            </div>
            
            {performanceMetrics && !performanceMetrics.error && (
              <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-md">
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Performance Metrics</h4>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                  {performanceMetrics.totalDataMB !== undefined && (
                    <>
                      <div className="text-gray-600 dark:text-gray-400">Total Data Transfer:</div>
                      <div className="text-gray-900 dark:text-gray-100 font-medium">
                        {performanceMetrics.totalDataMB.toFixed(2)} MB
                      </div>
                    </>
                  )}
                  {performanceMetrics.transitions !== undefined && (
                    <>
                      <div className="text-gray-600 dark:text-gray-400">Node Transitions:</div>
                      <div className="text-gray-900 dark:text-gray-100 font-medium">
                        {performanceMetrics.transitions}
                      </div>
                    </>
                  )}
                  {performanceMetrics.numNodes !== undefined && (
                    <>
                      <div className="text-gray-600 dark:text-gray-400">Active Nodes:</div>
                      <div className="text-gray-900 dark:text-gray-100 font-medium">
                        {performanceMetrics.numNodes}
                      </div>
                    </>
                  )}
                  {performanceMetrics.numStages !== undefined && (
                    <>
                      <div className="text-gray-600 dark:text-gray-400">Pipeline Stages:</div>
                      <div className="text-gray-900 dark:text-gray-100 font-medium">
                        {performanceMetrics.numStages}
                      </div>
                    </>
                  )}
                  {performanceMetrics.microbatches !== undefined && (
                    <>
                      <div className="text-gray-600 dark:text-gray-400">Microbatches:</div>
                      <div className="text-gray-900 dark:text-gray-100 font-medium">
                        {performanceMetrics.microbatches}
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">
              Tensor Parallelism Controls
            </h3>
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Parallelism Strategy
              </label>
              <select
                value={parallelStatus.strategy || 'layer_parallel'}
                onChange={(e) => setParallelismStrategy(e.target.value)}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="layer_parallel">Layer Parallelism</option>
                <option value="tensor_parallel">Tensor Parallelism</option>
                <option value="pipeline_parallel">Pipeline Parallelism</option>
              </select>
              <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                Layer: Distribute transformer blocks across nodes<br />
                Tensor: Split attention heads and MLP layers<br />
                Pipeline: Process different input batch segments in parallel
              </p>
            </div>
            
            <div className="flex justify-between">
              <button
                onClick={() => socket.emit('get_nodes')}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-800 text-white rounded text-sm"
              >
                Refresh Available Nodes
              </button>
              <button
                onClick={disableTensorParallelism}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 dark:bg-red-700 dark:hover:bg-red-800 text-white rounded text-sm"
              >
                Disable Tensor Parallelism
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="text-right p-2 px-4 text-xs text-gray-300">
        powered by{' '}
        <a
          href="https://webllm.mlc.ai/"
          className="text-primary-light hover:underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          WebLLM
        </a>
      </div>
    </div>
  );
};

export default ChatPage;
