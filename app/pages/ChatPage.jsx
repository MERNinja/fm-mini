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
  const messagesEndRef = useRef(null);
  const { theme } = useTheme();

  // Connect to socket.io server
  useEffect(() => {
    const newSocket = io(); // Use Vite's proxy
    setSocket(newSocket);

    // Generate a random node ID for this client
    const id = 'node_' + Math.random().toString(36).substring(2, 9);
    setNodeId(id);

    newSocket.on('connect', () => {
      console.log('Connected to server');
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

  // Handle send message
  const handleSendMessage = async () => {
    if (!input.trim() || isGenerating) return;

    // Add user message to chat
    const userMessage = { text: input, sender: 'user' };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');

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

        // Broadcast message to other nodes if connected to socket
        if (socket) {
          socket.emit('message', {
            from: nodeId,
            text: fullResponse,
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
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-800 text-white rounded transition-colors"
              >
                Load Model
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
              <span className="text-xs text-white/80">Node ID: {nodeId}</span>
            </div>
          )}
          {modelStatus === 'error' && (
            <span className="bg-error-light dark:bg-error-dark px-3 py-1 rounded text-xs font-medium">
              Error
            </span>
          )}
        </div>
      </div>

      <div className="flex-1 p-4 overflow-y-auto bg-gray-50 dark:bg-gray-800 flex flex-col">
        {messages.length === 0 && modelStatus === 'idle' && (
          <div className="self-center text-center max-w-md my-auto p-8 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-xl font-bold mb-4 text-primary-light dark:text-blue-400">
              Welcome to WebLLM Chat!
            </h3>
            <p className="mb-2 text-gray-600 dark:text-gray-300">
              Select a model from the dropdown and click "Load Model" to start
              chatting.
            </p>
            <p className="mb-2 text-gray-600 dark:text-gray-300">
              Once the model is loaded, your browser becomes a node in the
              network.
            </p>
            <p className="text-sm text-error-light dark:text-error-dark p-2 border border-dashed border-error-light dark:border-error-dark rounded mt-4">
              Note: WebLLM requires a browser with WebGPU support (Chrome 113+
              or Edge 113+).
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
