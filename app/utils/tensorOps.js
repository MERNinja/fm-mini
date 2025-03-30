/**
 * Tensor Operations for WebLLM
 * This file extends WebLLM with tensor-level operations for model parallelism
 */

/**
 * Tensor class for representing model tensors with metadata
 */
export class Tensor {
  /**
   * Create a new tensor
   * @param {Float32Array|Array<number>} data Tensor data
   * @param {Array<number>} shape Tensor shape
   * @param {string} dtype Data type ('float32' by default)
   * @param {Object} metadata Additional metadata
   */
  constructor(data, shape, dtype = 'float32', metadata = {}) {
    // Convert Array to Float32Array if needed
    this.data = data instanceof Float32Array ? data : new Float32Array(data);
    this.shape = shape;
    this.dtype = dtype;
    this.metadata = metadata;
  }

  /**
   * Get the total number of elements in the tensor
   */
  get size() {
    return this.shape.reduce((a, b) => a * b, 1);
  }

  /**
   * Create a tensor from binary data
   * @param {ArrayBuffer} buffer The binary data
   * @param {Array<number>} shape The tensor shape
   * @param {string} dtype The data type
   * @returns {Tensor} A new tensor
   */
  static fromBuffer(buffer, shape, dtype = 'float32') {
    if (dtype === 'float32') {
      return new Tensor(new Float32Array(buffer), shape, dtype);
    } else if (dtype === 'int32') {
      return new Tensor(new Int32Array(buffer), shape, dtype);
    } else {
      throw new Error(`Unsupported data type: ${dtype}`);
    }
  }

  /**
   * Convert the tensor to a binary buffer
   * @returns {ArrayBuffer} The binary representation
   */
  toBuffer() {
    return this.data.buffer;
  }

  /**
   * Reshape the tensor
   * @param {Array<number>} newShape The new shape
   * @returns {Tensor} A new tensor with the same data but different shape
   */
  reshape(newShape) {
    // Check if sizes match
    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== this.size) {
      throw new Error(`Cannot reshape tensor of size ${this.size} to shape with size ${newSize}`);
    }
    
    return new Tensor(this.data, newShape, this.dtype, this.metadata);
  }

  /**
   * Split the tensor along a dimension
   * @param {number} dim The dimension to split
   * @param {number} numSplits Number of splits
   * @returns {Array<Tensor>} Array of tensor chunks
   */
  split(dim, numSplits) {
    if (dim >= this.shape.length) {
      throw new Error(`Cannot split dimension ${dim} in tensor with shape ${this.shape}`);
    }
    
    const dimSize = this.shape[dim];
    if (dimSize % numSplits !== 0) {
      throw new Error(`Dimension ${dim} size ${dimSize} is not divisible by ${numSplits}`);
    }
    
    const chunkSize = dimSize / numSplits;
    const result = [];
    
    // Calculate the size of each chunk in elements
    const totalElements = this.size;
    const elementsPerChunk = totalElements / numSplits;
    
    // Create new shape for chunks
    const newShape = [...this.shape];
    newShape[dim] = chunkSize;
    
    // Create chunks
    for (let i = 0; i < numSplits; i++) {
      const start = i * elementsPerChunk;
      const end = start + elementsPerChunk;
      const chunkData = this.data.slice(start, end);
      
      result.push(new Tensor(chunkData, newShape, this.dtype, {
        ...this.metadata,
        originalDim: dim,
        chunkIndex: i,
        numChunks: numSplits
      }));
    }
    
    return result;
  }

  /**
   * Concatenate tensors along a dimension
   * @param {Array<Tensor>} tensors Array of tensors to concatenate
   * @param {number} dim Dimension to concatenate along
   * @returns {Tensor} The concatenated tensor
   */
  static concat(tensors, dim) {
    if (!tensors || tensors.length === 0) {
      throw new Error('No tensors to concatenate');
    }
    
    // Check that shapes match except for the concatenation dimension
    const firstShape = tensors[0].shape;
    const dtype = tensors[0].dtype;
    
    for (let i = 1; i < tensors.length; i++) {
      const shape = tensors[i].shape;
      
      if (shape.length !== firstShape.length) {
        throw new Error('Tensors must have the same number of dimensions');
      }
      
      if (tensors[i].dtype !== dtype) {
        throw new Error('Tensors must have the same data type');
      }
      
      for (let j = 0; j < shape.length; j++) {
        if (j !== dim && shape[j] !== firstShape[j]) {
          throw new Error(`Tensor shapes must match except for dimension ${dim}`);
        }
      }
    }
    
    // Calculate new shape
    const newShape = [...firstShape];
    newShape[dim] = tensors.reduce((sum, tensor) => sum + tensor.shape[dim], 0);
    
    // Create new data array
    const totalElements = tensors.reduce((sum, tensor) => sum + tensor.size, 0);
    const newData = new Float32Array(totalElements);
    
    // Copy data from each tensor
    let offset = 0;
    for (const tensor of tensors) {
      newData.set(tensor.data, offset);
      offset += tensor.size;
    }
    
    return new Tensor(newData, newShape, dtype);
  }

  /**
   * Create a tensor filled with ones
   * @param {Array<number>} shape The tensor shape
   * @returns {Tensor} A new tensor filled with ones
   */
  static ones(shape) {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size).fill(1);
    return new Tensor(data, shape);
  }

  /**
   * Create a tensor filled with zeros
   * @param {Array<number>} shape The tensor shape
   * @returns {Tensor} A new tensor filled with zeros
   */
  static zeros(shape) {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    return new Tensor(data, shape);
  }
}

/**
 * Helper functions for integrating with WebLLM
 */
export const WebLLMTensorOps = {
  /**
   * Extract tensors from a WebLLM model
   * @param {Object} engine The WebLLM engine
   * @param {string} layerName The layer name
   * @returns {Object} Object containing tensors
   */
  async extractTensors(engine, layerName) {
    // This is a placeholder - actual implementation would depend on access to WebLLM internals
    // which may require modifications to the WebLLM library
    console.warn('extractTensors is a stub - actual implementation requires WebLLM modifications');
    
    // Mock implementation for demonstration
    const hiddenSize = 768; // Example size
    
    return {
      weights: Tensor.ones([hiddenSize, hiddenSize]),
      bias: Tensor.zeros([hiddenSize])
    };
  },
  
  /**
   * Send a tensor to another node for processing
   * @param {Tensor} tensor The tensor to send
   * @param {string} operationType The operation to perform
   * @param {string} nodeId The target node ID
   * @param {Object} options Additional options
   * @returns {Promise<Tensor>} The processed tensor
   */
  async sendTensorForProcessing(tensor, operationType, nodeId, options = {}) {
    // Convert tensor to ArrayBuffer for transmission
    const buffer = tensor.toBuffer();
    
    // Transmit via WebRTC - this would be implemented in the TensorParallelManager
    // This is a placeholder that simulates processing
    console.warn('sendTensorForProcessing is a stub - actual implementation requires WebRTC data transmission');
    
    // Mock implementation for demonstration - just return the input tensor
    return tensor;
  },
  
  /**
   * Apply attention computation split across multiple nodes
   * @param {Tensor} inputTensor The input tensor
   * @param {Array<string>} nodeIds Array of node IDs to use
   * @param {Object} attentionConfig Attention configuration
   * @returns {Promise<Tensor>} The result tensor
   */
  async parallelAttention(inputTensor, nodeIds, attentionConfig) {
    if (!nodeIds || nodeIds.length === 0) {
      throw new Error('No nodes available for parallel attention');
    }
    
    // Split attention heads across nodes
    const numHeads = attentionConfig.numHeads || 12;
    const headsPerNode = Math.ceil(numHeads / nodeIds.length);
    
    // This is a simplified implementation that doesn't actually split computation
    // Real implementation would require access to WebLLM's internal attention mechanism
    console.warn('parallelAttention is a stub - actual implementation requires WebLLM modifications');
    
    // Mock implementation - just return the input tensor
    return inputTensor;
  }
};

export default {
  Tensor,
  WebLLMTensorOps
}; 