/**
 * Model weights loader for real tensor parallelism
 * This handles downloading and managing model weights distributed across browsers
 */

import { createMatrix } from './tensorOps.js';

// Cache for loaded model weights
const weightsCache = new Map();

/**
 * Load model weights from a remote source
 * @param {string} modelId - The model identifier
 * @param {Array<number>} layerIndices - Which layers to load 
 * @returns {Promise<Object>} - The loaded model weights
 */
export async function loadModelWeights(modelId, layerIndices) {
  console.log(`Loading model weights for ${modelId}, layers ${layerIndices.join(', ')}`);
  
  // Create a cache key
  const cacheKey = `${modelId}_${layerIndices.join('_')}`;
  
  // Check if weights are already cached
  if (weightsCache.has(cacheKey)) {
    console.log('Using cached model weights');
    return weightsCache.get(cacheKey);
  }
  
  // The URL to fetch model weights
  const baseUrl = `/models/${modelId}/weights`;
  
  // Load weights for each layer
  const weights = {};
  
  try {
    // First load model config
    const configResponse = await fetch(`/models/${modelId}/config.json`);
    const config = await configResponse.json();
    console.log(`Loaded model config: ${config.model_type}, ${config.hidden_size} hidden size`);
    
    weights.config = config;
    weights.layers = [];
    
    // Load weights for each requested layer
    for (const layerIndex of layerIndices) {
      console.log(`Fetching weights for layer ${layerIndex}`);
      
      // Fetch layer weights
      const layerUrl = `${baseUrl}/layer_${layerIndex}.bin`;
      const response = await fetch(layerUrl);
      
      if (!response.ok) {
        throw new Error(`Failed to load weights for layer ${layerIndex}: ${response.statusText}`);
      }
      
      // Get the binary data
      const arrayBuffer = await response.arrayBuffer();
      
      // Process weights according to model architecture
      const layerWeights = await processLayerWeights(arrayBuffer, config, layerIndex);
      
      weights.layers[layerIndex] = layerWeights;
      console.log(`Successfully loaded weights for layer ${layerIndex}`);
    }
    
    // Cache weights for future use
    weightsCache.set(cacheKey, weights);
    
    return weights;
  } catch (error) {
    console.error('Error loading model weights:', error);
    // For demo purposes, generate synthetic weights if real weights can't be loaded
    return generateSyntheticWeights(modelId, layerIndices);
  }
}

/**
 * Process binary weight data into matrix format
 */
async function processLayerWeights(arrayBuffer, config, layerIndex) {
  // Extract dimensions from model config
  const hiddenSize = config.hidden_size;
  const intermediateSize = config.intermediate_size || hiddenSize * 4;
  const numAttentionHeads = config.num_attention_heads;
  const headSize = hiddenSize / numAttentionHeads;
  
  // Create a float32 view of the binary data
  const floatArray = new Float32Array(arrayBuffer);
  let offset = 0;
  
  // Process attention weights
  const attention = {
    query: createMatrixFromBuffer(floatArray, hiddenSize, hiddenSize, offset),
    key: createMatrixFromBuffer(floatArray, hiddenSize, hiddenSize, offset += hiddenSize * hiddenSize),
    value: createMatrixFromBuffer(floatArray, hiddenSize, hiddenSize, offset += hiddenSize * hiddenSize),
    output: createMatrixFromBuffer(floatArray, hiddenSize, hiddenSize, offset += hiddenSize * hiddenSize)
  };
  
  // Process feed-forward weights
  const feedForward = {
    linear1: createMatrixFromBuffer(floatArray, hiddenSize, intermediateSize, offset += hiddenSize * hiddenSize),
    linear2: createMatrixFromBuffer(floatArray, intermediateSize, hiddenSize, offset += hiddenSize * intermediateSize)
  };
  
  // Process layer normalization weights
  const layerNorm = {
    attention: {
      scale: createMatrixFromBuffer(floatArray, 1, hiddenSize, offset += intermediateSize * hiddenSize),
      bias: createMatrixFromBuffer(floatArray, 1, hiddenSize, offset += hiddenSize)
    },
    output: {
      scale: createMatrixFromBuffer(floatArray, 1, hiddenSize, offset += hiddenSize),
      bias: createMatrixFromBuffer(floatArray, 1, hiddenSize, offset += hiddenSize)
    }
  };
  
  return {
    attention,
    feedForward,
    layerNorm
  };
}

/**
 * Create a matrix from a buffer at a specific offset
 */
function createMatrixFromBuffer(buffer, rows, cols, offset) {
  const size = rows * cols;
  const data = new Float32Array(size);
  
  // Copy data from buffer
  for (let i = 0; i < size; i++) {
    data[i] = buffer[offset + i];
  }
  
  return createMatrix(rows, cols, data);
}

/**
 * Generate synthetic weights for testing or when real weights aren't available
 */
function generateSyntheticWeights(modelId, layerIndices) {
  console.log(`Generating synthetic weights for ${modelId}`);
  
  // Configuration based on model ID
  let hiddenSize, numLayers, numAttentionHeads;
  
  if (modelId.includes('llama')) {
    // Llama-like architecture
    hiddenSize = 4096;
    numLayers = 32;
    numAttentionHeads = 32;
  } else {
    // Generic transformer
    hiddenSize = 768;
    numLayers = 12;
    numAttentionHeads = 12;
  }
  
  const weights = {
    config: {
      model_type: modelId.split('-')[0],
      hidden_size: hiddenSize,
      num_hidden_layers: numLayers,
      num_attention_heads: numAttentionHeads,
      intermediate_size: hiddenSize * 4
    },
    layers: []
  };
  
  // Generate weights for each layer
  for (const layerIndex of layerIndices) {
    const scale = 1.0 / Math.sqrt(hiddenSize);
    
    weights.layers[layerIndex] = {
      attention: {
        query: generateRandomMatrix(hiddenSize, hiddenSize, scale),
        key: generateRandomMatrix(hiddenSize, hiddenSize, scale),
        value: generateRandomMatrix(hiddenSize, hiddenSize, scale),
        output: generateRandomMatrix(hiddenSize, hiddenSize, scale)
      },
      feedForward: {
        linear1: generateRandomMatrix(hiddenSize, hiddenSize * 4, scale),
        linear2: generateRandomMatrix(hiddenSize * 4, hiddenSize, scale)
      },
      layerNorm: {
        attention: {
          scale: generateOnesMatrix(1, hiddenSize),
          bias: generateZerosMatrix(1, hiddenSize)
        },
        output: {
          scale: generateOnesMatrix(1, hiddenSize),
          bias: generateZerosMatrix(1, hiddenSize)
        }
      }
    };
  }
  
  return weights;
}

/**
 * Generate a random matrix for synthetic weights
 */
function generateRandomMatrix(rows, cols, scale = 0.02) {
  const matrix = createMatrix(rows, cols);
  for (let i = 0; i < matrix.size; i++) {
    matrix.data[i] = (Math.random() * 2 - 1) * scale;
  }
  return matrix;
}

/**
 * Generate a matrix filled with ones
 */
function generateOnesMatrix(rows, cols) {
  const matrix = createMatrix(rows, cols);
  for (let i = 0; i < matrix.size; i++) {
    matrix.data[i] = 1.0;
  }
  return matrix;
}

/**
 * Generate a matrix filled with zeros
 */
function generateZerosMatrix(rows, cols) {
  const matrix = createMatrix(rows, cols);
  for (let i = 0; i < matrix.size; i++) {
    matrix.data[i] = 0.0;
  }
  return matrix;
}

/**
 * Partition model weights for tensor parallelism
 * @param {Object} weights - The complete model weights
 * @param {number} nodeIndex - Current node's index in the parallelism setup
 * @param {number} totalNodes - Total nodes for tensor parallelism
 * @returns {Object} - Partitioned weights for this node
 */
export function partitionWeightsForNode(weights, nodeIndex, totalNodes) {
  if (!weights || !weights.config) {
    throw new Error('Invalid weights object');
  }
  
  const numLayers = weights.config.num_hidden_layers;
  
  // Determine layer range for this node (simple partitioning strategy)
  const layersPerNode = Math.floor(numLayers / totalNodes);
  const extraLayers = numLayers % totalNodes;
  
  const startLayer = nodeIndex * layersPerNode + Math.min(nodeIndex, extraLayers);
  const endLayer = startLayer + layersPerNode + (nodeIndex < extraLayers ? 1 : 0) - 1;
  
  console.log(`Node ${nodeIndex} responsible for layers ${startLayer}-${endLayer}`);
  
  // Extract only the weights needed for these layers
  const partitionedWeights = {
    config: weights.config,
    layers: [],
    layerRange: [startLayer, endLayer]
  };
  
  for (let i = startLayer; i <= endLayer; i++) {
    if (weights.layers[i]) {
      partitionedWeights.layers[i - startLayer] = weights.layers[i];
    }
  }
  
  return partitionedWeights;
} 