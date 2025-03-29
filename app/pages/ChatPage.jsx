import React, { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import * as webllm from '@mlc-ai/web-llm';
import { useTheme } from '../context/ThemeContext';
import DistributedModelManager from '../utils/DistributedModelManager';
import NodeList from '../components/NodeList';

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
  const [isDistributedGenerating, setIsDistributedGenerating] = useState(false);
  const [autoLoad, setAutoLoad] = useState(() => {
    // Check if auto-load is enabled
    return localStorage.getItem('autoLoadModel') === 'true';
  });
  const [activeTab, setActiveTab] = useState('chat'); // 'chat', 'nodelog', or 'network'
  const [nodeLogs, setNodeLogs] = useState([]);
  const messagesEndRef = useRef(null);
  const nodeLogsEndRef = useRef(null);
  const { theme } = useTheme();
  const [distributedManager, setDistributedManager] = useState(null);
  const [tensorParallelismEnabled, setTensorParallelismEnabled] =
    useState(false);
  const [connectedPeers, setConnectedPeers] = useState(0);
  const [tensorNodes, setTensorNodes] = useState([]);

  // Connect to socket.io server
  useEffect(() => {
    // Use proper connection options for compatibility with Vercel
    const newSocket = io({
      path: '/socket.io/',
      transports: ['websocket', 'polling'],
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 20000,
    });

    setSocket(newSocket);

    // Generate a random node ID for this client
    const id = 'node_' + Math.random().toString(36).substring(2, 9);
    setNodeId(id);

    newSocket.on('connect', () => {
      console.log('Connected to server with socket ID:', newSocket.id);
      // Add connection logs
      setNodeLogs((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          nodeId: id,
          socketId: newSocket.id || 'unknown',
          action: 'socket-connected',
          prompt: `Socket connected with ID: ${newSocket.id}`,
        },
      ]);
    });

    newSocket.on('connect_error', (error) => {
      console.error('Socket connection error:', error);
      setNodeLogs((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          nodeId: id,
          socketId: 'unknown',
          action: 'error',
          prompt: `Socket connection error: ${error.message}`,
        },
      ]);
    });

    newSocket.on('disconnect', (reason) => {
      console.log('Socket disconnected:', reason);
      setNodeLogs((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          nodeId: id,
          socketId: newSocket.id || 'unknown',
          action: 'socket-disconnected',
          prompt: `Socket disconnected: ${reason}`,
        },
      ]);
    });

    // Acknowledgment from server upon successful connection
    newSocket.on('connection_ack', (data) => {
      console.log('Server acknowledged connection:', data);
      setNodeLogs((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          nodeId: id,
          socketId: data.socketId || 'unknown',
          action: 'socket-ack',
          prompt: `Connection acknowledged by server`,
        },
      ]);
    });

    newSocket.on('message', (message) => {
      // Only add messages sent from this node or explicitly directed to this node
      if (message.from === nodeId || message.to === nodeId) {
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
        id: socket.id, // Use socket.id instead of nodeId
        model: selectedModel,
        ip: window.location.host,
        status: 'online',
        role: tensorParallelismEnabled ? 'coordinator' : 'worker',
        capabilities: {
          tensorParallelism: tensorParallelismEnabled,
          gpuMemory: 1024, // Mock value, could be detected from WebGPU in a real implementation
          cpuCores: navigator.hardwareConcurrency || 4,
        },
      });
      console.log('Registered node with socket ID:', socket.id);
    }
  }, [
    engine,
    socket,
    nodeId,
    selectedModel,
    modelStatus,
    tensorParallelismEnabled,
  ]);

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
      // Create MLCEngine with the selected model
      const mlcEngine = await webllm.CreateMLCEngine(selectedModel, {
        initProgressCallback: initProgressCallback,
      });

      // // Wait for initialization to complete
      // await mlcEngine.waitForInitialization();

      setEngine(mlcEngine);
      setModelStatus('ready');

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
    } catch (error) {
      console.error('Error loading model:', error);
      setModelStatus('error');
      setMessages((prev) => [
        ...prev,
        {
          text: `Error loading model: ${error.message}`,
          sender: 'bot',
          nodeId: 'system',
          socketId: socket?.id || 'unknown',
        },
      ]);
    }
  };

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-scroll to bottom on new logs
  useEffect(() => {
    nodeLogsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [nodeLogs]);

  // Update tensor parallelism capability when it changes
  useEffect(() => {
    if (socket && nodeId && modelStatus === 'ready') {
      // Update this node's capabilities
      socket.emit('update_node_capabilities', {
        id: socket.id, // Use socket.id instead of nodeId
        capabilities: {
          tensorParallelism: tensorParallelismEnabled,
          gpuMemory: 1024,
          cpuCores: navigator.hardwareConcurrency || 4,
        },
      });

      // Log the change
      setNodeLogs((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          nodeId,
          socketId: socket?.id || 'unknown',
          action: tensorParallelismEnabled
            ? 'tensor-parallelism-enabled'
            : 'tensor-parallelism-disabled',
          prompt: `Tensor parallelism ${
            tensorParallelismEnabled ? 'enabled' : 'disabled'
          }`,
        },
      ]);

      // Request tensor-capable nodes if enabled
      if (tensorParallelismEnabled) {
        socket.emit('get_tensor_nodes', selectedModel);

        // Add listener for tensor node list if not already added
        if (!socket._listeners || !socket._listeners['tensor_node_list']) {
          socket.on('tensor_node_list', (nodes) => {
            console.log('Tensor-capable nodes:', nodes);

            // Update the tensor nodes state
            setTensorNodes(nodes);

            setNodeLogs((prev) => [
              ...prev,
              {
                timestamp: new Date().toLocaleTimeString(),
                nodeId,
                socketId: socket?.id || 'unknown',
                action: 'tensor-nodes-discovered',
                prompt: `Found ${nodes.length} tensor-capable nodes: ${nodes
                  .map((n) => n.id)
                  .join(', ')}`,
              },
            ]);
          });
          socket._listeners = socket._listeners || {};
          socket._listeners['tensor_node_list'] = true;
        }
      }
    }
  }, [tensorParallelismEnabled, socket, nodeId, modelStatus, selectedModel]);

  // Initialize distributed model manager when socket and nodeId are ready
  useEffect(() => {
    if (socket && nodeId && !distributedManager) {
      // Use socket.id instead of nodeId for distributed manager
      const manager = new DistributedModelManager(
        socket,
        socket.id, // Use socket.id instead of nodeId
        selectedModel || ''
      );

      // Set up event handler
      manager.setEventCallback((event, data) => {
        console.log(`DistributedModelManager event: ${event}`, data);

        // Log events to node logs
        setNodeLogs((prev) => [
          ...prev,
          {
            timestamp: new Date().toLocaleTimeString(),
            nodeId,
            socketId: socket?.id || 'unknown',
            action: `tensor-${event}`,
            prompt: `${event}: ${JSON.stringify(data)}`,
          },
        ]);

        // Update connected peers count
        if (event === 'node-connected' || event === 'nodes-updated') {
          setConnectedPeers(data.totalConnected || 0);
        }
      });

      setDistributedManager(manager);
    }
  }, [socket, nodeId, selectedModel]);

  // Update model ID in distributed manager when model changes
  useEffect(() => {
    if (distributedManager && selectedModel) {
      distributedManager.modelId = selectedModel;
    }
  }, [selectedModel, distributedManager]);

  // Initialize the distributed manager when engine is loaded
  useEffect(() => {
    if (
      distributedManager &&
      engine &&
      modelStatus === 'ready' &&
      tensorParallelismEnabled
    ) {
      // Initialize as coordinator since this node has the model loaded
      const initDistributed = async () => {
        try {
          // Extract model config and weights from engine
          // This is a simplification - real implementation would extract actual model weights
          const modelConfig = {
            model: selectedModel,
            hidden_size: 768,
            num_heads: 12,
            num_layers: 12,
          };

          // Mock weights for demonstration
          const modelWeights = {
            mock: true,
          };

          await distributedManager.initAsCoordinator(modelConfig, modelWeights);

          // Connect to available nodes
          const connectedCount = await distributedManager.connectToNodes();

          setNodeLogs((prev) => [
            ...prev,
            {
              timestamp: new Date().toLocaleTimeString(),
              nodeId,
              socketId: socket?.id || 'unknown',
              action: 'tensor-parallelism-initialized',
              prompt: `Connected to ${connectedCount} worker nodes for tensor parallelism`,
            },
          ]);
        } catch (error) {
          console.error('Error initializing distributed model:', error);

          setNodeLogs((prev) => [
            ...prev,
            {
              timestamp: new Date().toLocaleTimeString(),
              nodeId,
              socketId: socket?.id || 'unknown',
              action: 'error',
              prompt: `Error initializing tensor parallelism: ${error.message}`,
            },
          ]);
        }
      };

      initDistributed();
    }
  }, [
    distributedManager,
    engine,
    modelStatus,
    tensorParallelismEnabled,
    selectedModel,
  ]);

  // Clean up distributed manager on unmount or model changes
  useEffect(() => {
    return () => {
      if (distributedManager) {
        distributedManager.cleanup();
      }
    };
  }, [distributedManager]);

  // Handle send message
  const handleSendMessage = async () => {
    if (!input.trim() || isGenerating) return;

    // Add user message to chat
    const userMessage = {
      text: input,
      sender: 'user',
      nodeId: nodeId,
      socketId: socket?.id,
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');

    // Log the prompt in node logs
    setNodeLogs((prev) => [
      ...prev,
      {
        timestamp: new Date().toLocaleTimeString(),
        nodeId,
        socketId: socket?.id || 'unknown',
        action: 'prompt_sent',
        prompt: userMessage.text,
      },
    ]);

    // If engine is ready, generate a response
    if (engine && modelStatus === 'ready') {
      try {
        setIsGenerating(true);

        // Create message array for chat completion
        const messageHistory = messages
          .filter((msg) => msg.sender === 'user' || msg.sender === 'bot')
          .map((msg) => ({
            role: msg.sender === 'user' ? 'user' : 'assistant',
            content: msg.text,
          }));

        // Add current user message
        messageHistory.push({ role: 'user', content: userMessage.text });

        // Add empty bot message that will be updated
        const newMessageIndex = messages.length + 1;
        setMessages((prev) => [...prev, { text: '', sender: 'bot' }]);

        // Log that this node is processing
        setNodeLogs((prev) => [
          ...prev,
          {
            timestamp: new Date().toLocaleTimeString(),
            nodeId,
            socketId: socket?.id || 'unknown',
            action: 'processing_prompt',
            prompt: userMessage.text,
          },
        ]);

        let fullResponse = '';
        let timingInfo = {};

        // Check if we should use distributed inference
        if (
          tensorParallelismEnabled &&
          distributedManager &&
          connectedPeers > 0
        ) {
          // Use distributed tensor parallelism
          setIsDistributedGenerating(true);

          setNodeLogs((prev) => [
            ...prev,
            {
              timestamp: new Date().toLocaleTimeString(),
              nodeId,
              socketId: socket?.id || 'unknown',
              action: 'tensor-inference-start',
              prompt: `Starting distributed inference with ${connectedPeers} connected peers`,
            },
          ]);

          const result = await distributedManager.runDistributedInference({
            messages: messageHistory,
            temperature: 0.7,
            max_tokens: 800,
          });

          setIsDistributedGenerating(false);
          fullResponse = result.text;
          timingInfo = result.timing;

          // Log detailed timing information
          setNodeLogs((prev) => [
            ...prev,
            {
              timestamp: new Date().toLocaleTimeString(),
              nodeId,
              socketId: socket?.id || 'unknown',
              action: 'tensor-inference-details',
              prompt: `Timing - Total: ${timingInfo.total}ms | Network: ${
                timingInfo.network || 0
              }ms | Compute: ${timingInfo.computation || 0}ms | Workers: ${
                result.workersUsed || connectedPeers
              }`,
            },
          ]);

          // Update the message with the result
          setMessages((prev) => {
            const updated = [...prev];
            updated[newMessageIndex] = {
              text: fullResponse,
              sender: 'bot',
              nodeId: nodeId,
              socketId: socket?.id,
              distributed: true,
              timing: timingInfo,
            };
            return updated;
          });

          // Log that distributed processing completed
          setNodeLogs((prev) => [
            ...prev,
            {
              timestamp: new Date().toLocaleTimeString(),
              nodeId,
              socketId: socket?.id || 'unknown',
              action: 'distributed-inference-completed',
              prompt: `Completed in ${timingInfo.total}ms across ${
                connectedPeers + 1
              } nodes`,
            },
          ]);
        } else {
          // Use standard WebLLM inference
          const startTime = performance.now();

          // Use streaming for more responsive UI
          const chunks = await engine.chat.completions.create({
            messages: messageHistory,
            temperature: 0.7,
            max_tokens: 800,
            stream: true,
          });

          // Process each chunk as it arrives
          for await (const chunk of chunks) {
            if (chunk.choices && chunk.choices[0]?.delta?.content) {
              const contentDelta = chunk.choices[0].delta.content;
              fullResponse += contentDelta;

              // Update the message with the accumulated text
              setMessages((prev) => {
                const updated = [...prev];
                updated[newMessageIndex] = {
                  text: fullResponse,
                  sender: 'bot',
                  nodeId: nodeId,
                  socketId: socket?.id,
                  distributed: false,
                };
                return updated;
              });
            }
          }

          const endTime = performance.now();
          timingInfo = {
            total: endTime - startTime,
            computation: endTime - startTime,
          };
        }

        // Log that this node completed processing
        setNodeLogs((prev) => [
          ...prev,
          {
            timestamp: new Date().toLocaleTimeString(),
            nodeId,
            socketId: socket?.id || 'unknown',
            action: 'completed_prompt',
            prompt:
              userMessage.text.substring(0, 30) +
              (userMessage.text.length > 30 ? '...' : ''),
          },
        ]);

        // Broadcast message to other nodes if connected to socket
        if (socket) {
          socket.emit('message', {
            from: nodeId,
            socketId: socket.id,
            text: fullResponse,
          });

          // Broadcast node activity
          socket.emit('node_activity', {
            nodeId,
            socketId: socket.id,
            action: 'response_generated',
            prompt:
              userMessage.text.substring(0, 30) +
              (userMessage.text.length > 30 ? '...' : ''),
          });
        }
      } catch (error) {
        console.error('Error generating response:', error);
        setMessages((prev) => [
          ...prev,
          {
            text: `Error generating response: ${error.message}`,
            sender: 'bot',
            nodeId: nodeId,
            socketId: socket?.id || 'unknown',
          },
        ]);

        // Log error in node logs
        setNodeLogs((prev) => [
          ...prev,
          {
            timestamp: new Date().toLocaleTimeString(),
            nodeId,
            socketId: socket?.id || 'unknown',
            action: 'error',
            prompt: `Error: ${error.message}`,
          },
        ]);
      } finally {
        setIsGenerating(false);
      }
    } else {
      // If model not loaded, prompt user to load it
      setMessages((prev) => [
        ...prev,
        {
          text: "Please load the WebLLM model first by clicking the 'Load Model' button.",
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
          nodeId,
          socketId: socket?.id || 'unknown',
          action: 'error',
          prompt: 'Model not loaded',
        },
      ]);
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

  // Unload the current model
  const unloadModel = () => {
    if (engine) {
      try {
        engine.terminate();
      } catch (error) {
        console.error('Error terminating engine:', error);
      }
    }

    // Notify the server that this node is no longer active
    if (socket && nodeId) {
      socket.emit('unregister_node', {
        id: nodeId,
      });
    }

    setModelStatus('idle');
    setEngine(null);

    setMessages((prev) => [
      ...prev,
      {
        text: 'Model unloaded. You can now select a different model.',
        sender: 'bot',
        nodeId: 'system',
        socketId: socket?.id || 'unknown',
      },
    ]);
  };

  // Cleanup engine on component unmount
  useEffect(() => {
    return () => {
      if (engine) {
        try {
          engine.terminate();
        } catch (error) {
          console.error('Error terminating engine:', error);
        }
      }
    };
  }, [engine]);

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
                Load Model & Host Node
              </button>
              <div className="flex items-center ml-2">
                <input
                  type="checkbox"
                  id="tensor-parallelism"
                  checked={tensorParallelismEnabled}
                  onChange={() => setTensorParallelismEnabled((prev) => !prev)}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label
                  htmlFor="tensor-parallelism"
                  className="ml-2 text-xs text-white cursor-pointer"
                >
                  Tensor Parallelism
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
            <div className="flex flex-col items-end gap-1">
              <div className="flex items-center gap-2">
                <span className="bg-success-light dark:bg-success-dark px-3 py-1 rounded text-xs font-medium">
                  Model Ready: {selectedModel}
                </span>
                {tensorParallelismEnabled && (
                  <span
                    className={`px-3 py-1 rounded text-xs font-medium flex items-center ${
                      connectedPeers > 0
                        ? 'bg-info-light dark:bg-info-dark'
                        : 'bg-yellow-600 dark:bg-yellow-700'
                    }`}
                  >
                    <span
                      className={`w-2 h-2 rounded-full mr-1 ${
                        connectedPeers > 0
                          ? 'bg-green-500 animate-pulse'
                          : 'bg-yellow-300'
                      }`}
                    ></span>
                    Tensor Parallelism: {connectedPeers} peers
                    {isDistributedGenerating && (
                      <span className="ml-1 bg-blue-700 text-white text-[10px] px-1 py-0.5 rounded-sm animate-pulse">
                        ACTIVE
                      </span>
                    )}
                  </span>
                )}
                <button
                  onClick={() => {
                    unloadModel();
                    setMessages((prev) => [
                      ...prev,
                      {
                        text: 'Model unloaded. You can now select a different model.',
                        sender: 'bot',
                        nodeId: 'system',
                        socketId: socket?.id || 'unknown',
                      },
                    ]);
                  }}
                  className="bg-blue-600 hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-800 px-2 py-1 rounded text-xs text-white"
                >
                  Change Model
                </button>
              </div>
              <span className="text-xs text-white/80">
                Socket ID: {socket?.id || 'N/A'}
              </span>
              {tensorParallelismEnabled && (
                <div className="flex items-center mt-1 gap-2">
                  <button
                    onClick={() => {
                      // Force refresh tensor nodes
                      socket.emit('get_tensor_nodes', selectedModel);
                      setNodeLogs((prev) => [
                        ...prev,
                        {
                          timestamp: new Date().toLocaleTimeString(),
                          nodeId,
                          socketId: socket?.id || 'unknown',
                          action: 'manual-refresh',
                          prompt: 'Manually refreshing tensor node list',
                        },
                      ]);
                    }}
                    className="bg-green-600 px-2 py-1 rounded text-xs text-white"
                  >
                    Refresh Tensor Nodes
                  </button>
                  <button
                    onClick={() => {
                      if (distributedManager) {
                        // Get all socket IDs from tensorNodes
                        const socketIds = tensorNodes
                          .map((node) => node.id)
                          .filter((id) => id !== socket.id);

                        if (socketIds.length === 0) {
                          setNodeLogs((prev) => [
                            ...prev,
                            {
                              timestamp: new Date().toLocaleTimeString(),
                              nodeId,
                              socketId: socket?.id || 'unknown',
                              action: 'manual-connect-error',
                              prompt: `No tensor-capable nodes found to connect to`,
                            },
                          ]);
                          return;
                        }

                        // Log the IDs we're trying to connect to
                        setNodeLogs((prev) => [
                          ...prev,
                          {
                            timestamp: new Date().toLocaleTimeString(),
                            nodeId,
                            socketId: socket?.id || 'unknown',
                            action: 'manual-connect-attempt',
                            prompt: `Attempting to connect to: ${socketIds.join(
                              ', '
                            )}`,
                          },
                        ]);

                        // Force connect to each socket ID
                        const connectPromises = socketIds.map((id) =>
                          distributedManager.rtcManager
                            .initConnection(id)
                            .then(() => ({ id, success: true }))
                            .catch((err) => ({
                              id,
                              success: false,
                              error: err.message,
                            }))
                        );

                        Promise.all(connectPromises).then((results) => {
                          const successful = results.filter(
                            (r) => r.success
                          ).length;

                          setNodeLogs((prev) => [
                            ...prev,
                            {
                              timestamp: new Date().toLocaleTimeString(),
                              nodeId,
                              socketId: socket?.id || 'unknown',
                              action: 'manual-connect-complete',
                              prompt: `Connected to ${successful}/${
                                socketIds.length
                              } nodes: ${results
                                .map(
                                  (r) =>
                                    `${r.id.substring(0, 6)}(${
                                      r.success ? 'OK' : 'FAIL'
                                    })`
                                )
                                .join(', ')}`,
                            },
                          ]);
                        });
                      }
                    }}
                    className="bg-purple-600 px-2 py-1 rounded text-xs text-white"
                  >
                    Force Connect
                  </button>
                </div>
              )}
            </div>
          )}
          {modelStatus === 'error' && (
            <span className="bg-error-light dark:bg-error-dark px-3 py-1 rounded text-xs font-medium">
              Error
            </span>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 dark:border-gray-700">
        <button
          className={`px-6 py-3 text-sm font-medium ${
            activeTab === 'chat'
              ? 'text-primary-light dark:text-blue-400 border-b-2 border-primary-light dark:border-blue-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 border-b-2 border-transparent'
          }`}
          onClick={() => setActiveTab('chat')}
        >
          Chat
        </button>
        <button
          className={`px-6 py-3 text-sm font-medium ${
            activeTab === 'nodelog'
              ? 'text-primary-light dark:text-blue-400 border-b-2 border-primary-light dark:border-blue-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 border-b-2 border-transparent'
          }`}
          onClick={() => setActiveTab('nodelog')}
        >
          Node Log
        </button>
        <button
          className={`px-6 py-3 text-sm font-medium ${
            activeTab === 'network'
              ? 'text-primary-light dark:text-blue-400 border-b-2 border-primary-light dark:border-blue-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 border-b-2 border-transparent'
          }`}
          onClick={() => setActiveTab('network')}
        >
          Network Status
        </button>
      </div>

      {/* Chat Tab Content */}
      {activeTab === 'chat' && (
        <>
          <div className="flex-1 p-4 overflow-y-auto bg-gray-50 dark:bg-gray-800 flex flex-col">
            {messages.length === 0 && modelStatus === 'idle' && (
              <div className="self-center text-center max-w-md my-auto p-8 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                <h3 className="text-xl font-bold mb-4 text-primary-light dark:text-blue-400">
                  Welcome to WebLLM Chat!
                </h3>
                <p className="mb-2 text-gray-600 dark:text-gray-300">
                  Select a model from the dropdown and click "Load Model & Host
                  Node" to start chatting.
                </p>
                <p className="mb-2 text-gray-600 dark:text-gray-300">
                  Once the model is loaded, your browser becomes a node in the
                  network.
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
                    : message.distributed
                    ? 'bg-blue-100 dark:bg-blue-900 text-gray-800 dark:text-gray-200 self-start border-l-4 border-blue-500'
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
                  {message.socketId && message.sender === 'bot' && (
                    <div className="text-xs mt-1 text-gray-500 dark:text-gray-400">
                      {message.nodeId === 'system' ? (
                        'System message'
                      ) : message.distributed ? (
                        <span className="flex items-center">
                          <span className="bg-blue-500 text-white px-2 py-0.5 rounded-full mr-1 animate-pulse">
                            Distributed
                          </span>
                          <span>
                            Across {connectedPeers + 1} nodes (
                            {message.timing?.total || '?'}ms)
                          </span>
                        </span>
                      ) : (
                        `From node: ${message.socketId.substring(0, 6)}...`
                      )}
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
                Active Socket:{' '}
                <span className="text-yellow-500">
                  {socket?.id || 'disconnected'}
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

            {/* Tensor parallelism debug info */}
            {tensorParallelismEnabled && (
              <div className="mt-3 border-t border-gray-800 pt-2">
                <div className="text-yellow-400 text-xs uppercase font-bold mb-1">
                  Tensor Parallelism Debug:
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                  <div>
                    <span className="text-gray-400">Status:</span>{' '}
                    <span className="text-green-500">ENABLED</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Connected Peers:</span>{' '}
                    <span
                      className={`text-${
                        connectedPeers > 0 ? 'green' : 'red'
                      }-500`}
                    >
                      {connectedPeers}{' '}
                      {isDistributedGenerating && (
                        <span className="text-blue-400 animate-pulse ml-1">
                          [ACTIVE]
                        </span>
                      )}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Node ID:</span>{' '}
                    <span className="text-blue-400">{nodeId}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Socket ID:</span>{' '}
                    <span className="text-blue-400">
                      {socket?.id.substring(0, 8)}...
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Node Role:</span>{' '}
                    <span className="text-pink-500">
                      {distributedManager?.isCoordinator
                        ? 'Coordinator'
                        : 'Worker'}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Model:</span>{' '}
                    <span className="text-cyan-500">{selectedModel}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Network Status:</span>{' '}
                    <span
                      className={`${
                        connectedPeers > 0
                          ? 'text-green-500'
                          : 'text-yellow-500'
                      }`}
                    >
                      {connectedPeers > 0 ? 'Connected' : 'Waiting for peers'}
                    </span>
                  </div>
                  <div>
                    <button
                      onClick={() => {
                        if (socket) {
                          socket.emit('get_nodes');
                          socket.emit('get_tensor_nodes', selectedModel);
                          setNodeLogs((prev) => [
                            ...prev,
                            {
                              timestamp: new Date().toLocaleTimeString(),
                              nodeId,
                              socketId: socket?.id || 'unknown',
                              action: 'debug-refresh',
                              prompt:
                                'Manually refreshing all node information',
                            },
                          ]);
                        }
                      }}
                      className="bg-gray-800 hover:bg-gray-700 text-green-400 px-2 py-0.5 rounded text-[10px] border border-gray-600"
                    >
                      Refresh Nodes
                    </button>
                  </div>
                </div>

                {/* Known tensor nodes */}
                <div className="mt-2 text-xs">
                  <div className="text-gray-400">
                    Known tensor nodes:{' '}
                    <span className="text-yellow-500">
                      {tensorNodes.length}
                    </span>
                  </div>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {tensorNodes.map((node) => (
                      <span
                        key={node.id}
                        className="bg-gray-900 px-1.5 py-0.5 rounded border border-gray-700 text-[10px]"
                      >
                        <span className="text-blue-400">
                          {node.id.substring(0, 6)}
                        </span>
                        {node.id === nodeId && (
                          <span className="text-green-500 ml-1">(self)</span>
                        )}
                      </span>
                    ))}
                    {tensorNodes.length === 0 && (
                      <span className="text-gray-500 text-[10px]">
                        No tensor-capable nodes found
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}
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

      {/* Network Status Tab Content */}
      {activeTab === 'network' && <NodeList socket={socket} />}

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
