import React, { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import { useTheme } from '../context/ThemeContext';

const NodeListPage = () => {
  const [nodes, setNodes] = useState([]);
  const [socket, setSocket] = useState(null);
  const [loading, setLoading] = useState(true);
  const { theme } = useTheme();

  useEffect(() => {
    // Connect to socket.io server
    const newSocket = io(); // Use Vite's proxy
    setSocket(newSocket);

    newSocket.on('connect', () => {
      console.log('Connected to server');
      // Request current node list
      newSocket.emit('get_nodes');
    });

    // Listen for node list updates
    newSocket.on('node_list', (nodeList) => {
      setNodes(nodeList);
      setLoading(false);
    });

    // Listen for new node registrations
    newSocket.on('node_registered', (node) => {
      setNodes((prev) => [...prev, node]);
    });

    // Listen for node disconnections
    newSocket.on('node_disconnected', (nodeId) => {
      setNodes((prev) => prev.filter((node) => node.id !== nodeId));
    });

    // Listen for node status updates
    newSocket.on('node_status_update', (update) => {
      setNodes((prev) =>
        prev.map((node) =>
          node.id === update.id ? { ...node, status: update.status } : node
        )
      );
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
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">
          Connected Nodes
        </h2>
        <button
          onClick={() => socket?.emit('get_nodes')}
          className="px-4 py-2 bg-primary-light hover:bg-blue-600 dark:bg-primary-dark dark:hover:bg-blue-700 text-white rounded-md transition-colors"
          aria-label="Refresh node list"
        >
          Refresh
        </button>
      </div>

      {loading ? (
        <div className="py-8 text-center text-gray-600 dark:text-gray-400">
          <div className="flex justify-center mb-4">
            <svg
              className="animate-spin h-8 w-8 text-primary-light dark:text-primary-dark"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
          </div>
          Loading nodes...
        </div>
      ) : nodes.length === 0 ? (
        <div className="p-8 text-center border border-dashed border-gray-300 dark:border-gray-700 rounded-md bg-gray-50 dark:bg-gray-900">
          <p className="text-lg font-medium mb-2 text-gray-700 dark:text-gray-300">
            No nodes are currently connected.
          </p>
          <p className="text-gray-500 dark:text-gray-400">
            Load a WebLLM model on the Chat page to create a new node.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  ID
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
                  Connected Since
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
              {nodes.map((node) => (
                <tr
                  key={node.id}
                  className="hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                    {node.id}
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
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                    {formatTime(node.connectedAt)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default NodeListPage;
