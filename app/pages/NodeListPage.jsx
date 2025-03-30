import React, { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import { useTheme } from '../context/ThemeContext';

const NodeListPage = () => {
  const [nodes, setNodes] = useState([]);
  const [socket, setSocket] = useState(null);
  const [loading, setLoading] = useState(true);
  const [tensorModels, setTensorModels] = useState({});
  const [tensorConnections, setTensorConnections] = useState([]);
  const [tensorStatus, setTensorStatus] = useState({});
  const { theme } = useTheme();

  // Sort nodes by connectedAt in descending order (newest first)
  const sortNodesByDate = (nodeList) => {
    return [...nodeList].sort((a, b) => {
      // Handle cases where connectedAt might be missing
      if (!a.connectedAt) return 1;
      if (!b.connectedAt) return -1;
      return new Date(b.connectedAt) - new Date(a.connectedAt);
    });
  };

  useEffect(() => {
    // Connect to socket.io server
    const newSocket = io(); // Use Vite's proxy
    setSocket(newSocket);

    newSocket.on('connect', () => {
      console.log('Connected to server with socket ID:', newSocket.id);
      // Immediately request nodes list when connected
      newSocket.emit('get_nodes');
      newSocket.emit('get_tensor_models');
      newSocket.emit('get_tensor_parallel_status');
    });

    newSocket.on('node_list', (data) => {
      setNodes(data || []);
      setLoading(false);
    });

    newSocket.on('tensor_models_list', (data) => {
      setTensorModels(data || {});
    });

    newSocket.on('node_registered', (node) => {
      setNodes(prev => {
        const exists = prev.find(n => n.id === node.id);
        if (exists) {
          return prev.map(n => n.id === node.id ? node : n);
        }
        return [...prev, node];
      });
    });

    newSocket.on('node_disconnected', (nodeId) => {
      setNodes(prev => prev.filter(node => node.id !== nodeId));
    });

    newSocket.on('tensor_model_registered', (data) => {
      // Refresh tensor models list
      newSocket.emit('get_tensor_models');
    });

    newSocket.on('tensor_parallel_status', (data) => {
      setTensorStatus(prev => ({
        ...prev,
        [data.nodeId]: data.enabled
      }));
    });

    // Request initial tensor status
    newSocket.emit('get_tensor_parallel_status', (data) => {
      const statusMap = {};
      Object.keys(data || {}).forEach(nodeId => {
        statusMap[nodeId] = data[nodeId].enabled;
      });
      setTensorStatus(statusMap);
    });

    // Listen for tensor connection updates
    newSocket.on('tensor_signal', (data) => {
      if (data.type === 'offer' || data.type === 'answer') {
        // Add or update connection
        setTensorConnections(prev => {
          const connIndex = prev.findIndex(conn => 
            (conn.from === data.from && conn.to === data.to) ||
            (conn.from === data.to && conn.to === data.from)
          );
          
          if (connIndex >= 0) {
            // Update existing connection
            const updated = [...prev];
            updated[connIndex] = {
              ...updated[connIndex],
              lastActive: new Date().toISOString(),
              type: data.type
            };
            return updated;
          } else {
            // Add new connection
            return [...prev, {
              from: data.from,
              to: data.to,
              type: data.type,
              established: data.type === 'answer',
              lastActive: new Date().toISOString()
            }];
          }
        });
      }
    });

    return () => {
      newSocket.disconnect();
    };
  }, []);

  // Function to get status style
  const getStatusStyle = (status) => {
    switch (status) {
      case 'online':
        return 'bg-success-light dark:bg-success-dark text-white';
      case 'offline':
        return 'bg-error-light dark:bg-error-dark text-white';
      case 'busy':
        return 'bg-warning-light dark:bg-warning-dark text-white';
      default:
        return 'bg-gray-400 dark:bg-gray-600 text-white';
    }
  };

  // Format timestamp
  const formatTime = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    // Show date and time for better clarity
    return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
  };
  
  // Check if a node has tensor parallelism capabilities
  const hasTensorCapability = (nodeId) => {
    // First check the explicit tensor status
    if (tensorStatus[nodeId] === true) {
      return true;
    }
    
    // Then check if the node has explicitly registered tensor capability
    const node = nodes.find(n => n.id === nodeId);
    if (node && node.tensorParallelEnabled === true) {
      return true;
    }
    
    // Fallback to checking tensor models
    for (const modelId in tensorModels) {
      if (tensorModels[modelId][nodeId]) {
        return true;
      }
    }
    
    return false;
  };
  
  // Get tensor connections for a specific node
  const getNodeTensorConnections = (nodeId) => {
    return tensorConnections.filter(conn => 
      conn.from === nodeId || conn.to === nodeId
    );
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">
          Connected Nodes
        </h2>
        <div className="flex space-x-2">
          <button
            onClick={() => {
              // Refresh both nodes and tensor status
              socket?.emit('get_nodes');
              socket?.emit('get_tensor_models');
              socket?.emit('get_tensor_parallel_status');
            }}
            className="px-4 py-2 bg-primary-light hover:bg-blue-600 dark:bg-primary-dark dark:hover:bg-blue-700 text-white rounded-md transition-colors"
            aria-label="Refresh node list and tensor status"
          >
            Refresh Nodes
          </button>
          <button
            onClick={() => socket?.emit('get_tensor_models')}
            className="px-4 py-2 bg-secondary-light hover:bg-secondary-dark dark:bg-secondary-dark dark:hover:bg-secondary-light text-white rounded-md transition-colors"
            aria-label="Refresh tensor models"
          >
            Refresh Tensor Models
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center items-center py-10">
          <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-primary-light dark:border-primary-dark"></div>
        </div>
      ) : nodes.length === 0 ? (
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6 text-center">
          <p className="text-gray-500 dark:text-gray-400">No nodes connected</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Node ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  IP Address
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Model
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Tensor Parallel
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider flex items-center">
                  Connected Since
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-4 w-4 ml-1 text-primary-light dark:text-primary-dark"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 15l7-7 7 7"
                    />
                  </svg>
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
              {nodes.map((node) => (
                <tr
                  key={node.id}
                  className={`hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors ${
                    hasTensorCapability(node.id) 
                      ? 'border-l-4 border-green-500 dark:border-green-700' 
                      : ''
                  }`}
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                    <div className="flex flex-col">
                      <span>{node.id || 'N/A'}</span>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        Socket: {node.socketId || 'N/A'}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                    {node.ip}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                    {node.model || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full w-20 justify-center ${getStatusStyle(
                        node.status
                      )}`}
                    >
                      {node.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    {hasTensorCapability(node.id) ? (
                      <div>
                        <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 mb-1">
                          Available
                        </span>
                        <div className="text-xs text-gray-600 dark:text-gray-400">
                          {getNodeTensorConnections(node.id).length > 0 ? (
                            <span>{getNodeTensorConnections(node.id).length} connections</span>
                          ) : (
                            <span>Ready for parallelism</span>
                          )}
                        </div>
                      </div>
                    ) : (
                      <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200">
                        Not available
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                    {formatTime(node.connectedAt)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      
      {/* Tensor Parallel Network Visualization */}
      {tensorConnections.length > 0 && (
        <div className="mt-8">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            Tensor Parallel Network
          </h3>
          <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-900">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      From Node
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      To Node
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Last Active
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                  {tensorConnections.map((connection, index) => (
                    <tr
                      key={index}
                      className="hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                    >
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                        {connection.from}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                        {connection.to}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                            connection.established
                              ? 'bg-success-light dark:bg-success-dark text-white'
                              : 'bg-warning-light dark:bg-warning-dark text-white'
                          }`}
                        >
                          {connection.established ? 'Established' : 'Negotiating'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                        {formatTime(connection.lastActive)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NodeListPage;
