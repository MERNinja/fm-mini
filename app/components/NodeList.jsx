import React, { useState, useEffect } from 'react';
import { useTheme } from '../context/ThemeContext';

const NodeList = ({ socket, existingNodes = null }) => {
  const [nodes, setNodes] = useState(existingNodes || []);
  const [loading, setLoading] = useState(existingNodes ? false : true);
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
    if (!socket) return;

    // Request current node list
    socket.emit('get_nodes');

    // Listen for node list updates
    const nodeListHandler = (nodeList) => {
      // Sort nodes by connectedAt in descending order (newest first)
      setNodes(sortNodesByDate(nodeList));
      setLoading(false);
    };

    // Listen for new node registrations
    const nodeRegisteredHandler = (node) => {
      setNodes((prev) => {
        // Add new node and re-sort
        const updated = [...prev, node];
        return sortNodesByDate(updated);
      });
    };

    // Listen for node disconnections
    const nodeDisconnectedHandler = (nodeId) => {
      setNodes((prev) =>
        sortNodesByDate(prev.filter((node) => node.id !== nodeId))
      );
    };

    // Listen for node status updates
    const nodeStatusUpdateHandler = (update) => {
      setNodes((prev) => {
        const updated = prev.map((node) =>
          node.id === update.id ? { ...node, status: update.status } : node
        );
        // Maintain the sort order
        return sortNodesByDate(updated);
      });
    };

    // Register event handlers
    socket.on('node_list', nodeListHandler);
    socket.on('node_registered', nodeRegisteredHandler);
    socket.on('node_disconnected', nodeDisconnectedHandler);
    socket.on('node_status_update', nodeStatusUpdateHandler);

    return () => {
      // Clean up event handlers
      socket.off('node_list', nodeListHandler);
      socket.off('node_registered', nodeRegisteredHandler);
      socket.off('node_disconnected', nodeDisconnectedHandler);
      socket.off('node_status_update', nodeStatusUpdateHandler);
    };
  }, [socket]);

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

  // Function to check if a node has tensor parallelism enabled
  const hasTensorParallelism = (node) => {
    return node.capabilities && node.capabilities.tensorParallelism === true;
  };

  // Format timestamp
  const formatTime = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    // Show date and time for better clarity
    return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
  };

  return (
    <div className="flex-1 overflow-y-auto bg-white dark:bg-gray-800">
      <div className="flex justify-between items-center p-4 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-xl font-bold text-gray-800 dark:text-gray-100">
          Network Status
        </h2>
        <button
          onClick={() => socket?.emit('get_nodes')}
          className="px-3 py-1 bg-primary-light hover:bg-blue-600 dark:bg-primary-dark dark:hover:bg-blue-700 text-white text-sm rounded-md transition-colors"
        >
          Refresh Nodes
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
        <div className="p-8 text-center border border-dashed border-gray-300 dark:border-gray-700 rounded-md m-4 bg-gray-50 dark:bg-gray-900">
          <p className="text-lg font-medium mb-2 text-gray-700 dark:text-gray-300">
            No nodes are currently connected.
          </p>
          <p className="text-gray-500 dark:text-gray-400">
            Load a WebLLM model on the Chat page to create a new node.
          </p>
        </div>
      ) : (
        <div className="p-4 overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700 rounded-lg overflow-hidden">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Socket ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Model
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Tensor Support
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Connected Since
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {nodes.map((node) => (
                <tr
                  key={node.id}
                  className="hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                    {node.socketId ? (
                      <>
                        <span className="font-mono">
                          {node.socketId.substring(0, 8)}
                        </span>
                        <span className="text-gray-500 dark:text-gray-400">
                          ...
                        </span>
                      </>
                    ) : (
                      'N/A'
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                    {node.model || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusStyle(
                        node.status
                      )}`}
                    >
                      {node.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {hasTensorParallelism(node) ? (
                      <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-500 text-white">
                        Enabled
                      </span>
                    ) : (
                      <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-gray-400 text-white">
                        Disabled
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
    </div>
  );
};

export default NodeList;
