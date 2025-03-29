import React, { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import * as webllm from '@mlc-ai/web-llm';
import { useTheme } from '../context/ThemeContext';

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
  const messagesEndRef = useRef(null);
  const nodeLogsEndRef = useRef(null);
  const { theme } = useTheme();

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
      setMessages((prev) => [
        ...prev,
        {
          text: message.text,
          sender: 'bot',
        },
      ]);
    });

    newSocket.on('node_activity', (activity) => {
      setNodeLogs((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          nodeId: activity.nodeId,
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

  // Handle send message
  const handleSendMessage = async () => {
    if (!input.trim() || isGenerating) return;

    // Add user message to chat
    const userMessage = { text: input, sender: 'user' };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');

    // Log the prompt in node logs
    setNodeLogs((prev) => [
      ...prev,
      {
        timestamp: new Date().toLocaleTimeString(),
        nodeId,
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
            action: 'processing_prompt',
            prompt: userMessage.text,
          },
        ]);

        // Use streaming for more responsive UI
        const chunks = await engine.chat.completions.create({
          messages: messageHistory,
          temperature: 0.7,
          max_tokens: 800,
          stream: true,
        });

        let fullResponse = '';

        // Process each chunk as it arrives
        for await (const chunk of chunks) {
          if (chunk.choices && chunk.choices[0]?.delta?.content) {
            const contentDelta = chunk.choices[0].delta.content;
            fullResponse += contentDelta;

            // Update the message with the accumulated text
            setMessages((prev) => {
              const updated = [...prev];
              updated[newMessageIndex] = { text: fullResponse, sender: 'bot' };
              return updated;
            });
          }
        }

        // Log that this node completed processing
        setNodeLogs((prev) => [
          ...prev,
          {
            timestamp: new Date().toLocaleTimeString(),
            nodeId,
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
            text: fullResponse,
          });

          // Broadcast node activity
          socket.emit('node_activity', {
            nodeId,
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
          },
        ]);

        // Log error in node logs
        setNodeLogs((prev) => [
          ...prev,
          {
            timestamp: new Date().toLocaleTimeString(),
            nodeId,
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
        },
      ]);

      // Log error in node logs
      setNodeLogs((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          nodeId,
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
              {/* <div className="flex items-center ml-2">
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
              </div> */}
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
                <button
                  onClick={() => {
                    unloadModel();
                    setMessages((prev) => [
                      ...prev,
                      {
                        text: 'Model unloaded. You can now select a different model.',
                        sender: 'bot',
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
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 self-start'
                } ${message.text ? '' : 'min-h-[40px] flex items-center'}`}
              >
                {message.text || (
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                )}
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
                          : 'text-blue-500'
                      }`}
                    >
                      [{log.action}]
                    </span>{' '}
                    <span className="text-purple-500">{log.nodeId}</span>{' '}
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
