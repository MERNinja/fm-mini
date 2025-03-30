/**
 * Tensor Parallelism Strategies
 * This file contains implementations of different parallelism strategies for LLM inference
 */

/**
 * Strategy types
 */
export const StrategyType = {
  LAYER_PARALLEL: 'layer_parallel',
  TENSOR_PARALLEL: 'tensor_parallel',
  PIPELINE_PARALLEL: 'pipeline_parallel',
  EXPERT_PARALLEL: 'expert_parallel',
};

/**
 * Base class for parallelism strategies
 */
export class ParallelismStrategy {
  constructor(options = {}) {
    this.options = options;
  }

  /**
   * Partition the model across nodes
   * @param {Object} modelConfig The model configuration
   * @param {Array<string>} nodeIds Available node IDs
   * @returns {Object} Mapping of layer/component names to node IDs
   */
  partition(modelConfig, nodeIds) {
    throw new Error('Method not implemented');
  }

  /**
   * Calculate the communication cost of the strategy
   * @param {Object} partitioning The partitioning scheme
   * @param {Object} modelConfig The model configuration
   * @returns {Object} Communication cost metrics
   */
  calculateCommunicationCost(partitioning, modelConfig) {
    throw new Error('Method not implemented');
  }
}

/**
 * Simple layer-wise parallelism strategy
 * Distributes transformer layers across available nodes
 */
export class LayerParallelStrategy extends ParallelismStrategy {
  constructor(options = {}) {
    super({
      keepEmbeddingsLocal: true,
      keepOutputLocal: true,
      ...options,
    });
  }

  /**
   * Partition the model by layer
   * @param {Object} modelConfig The model configuration
   * @param {Array<string>} nodeIds Available node IDs
   * @returns {Object} Mapping of layer names to node IDs
   */
  partition(modelConfig, nodeIds) {
    if (!nodeIds || nodeIds.length === 0) {
      throw new Error('No nodes available for partitioning');
    }

    const localNodeId = this.options.localNodeId;
    if (!localNodeId) {
      throw new Error('Local node ID is required');
    }

    // Include local node in the available nodes
    const allNodes = [...nodeIds];
    if (!allNodes.includes(localNodeId)) {
      allNodes.push(localNodeId);
    }

    const partitioning = {};
    
    // Keep embeddings and output layers on the local node if configured
    if (this.options.keepEmbeddingsLocal) {
      partitioning['embeddings'] = localNodeId;
    }
    
    if (this.options.keepOutputLocal) {
      partitioning['lm_head'] = localNodeId;
      partitioning['final_layernorm'] = localNodeId;
    }

    // Get the number of layers to distribute
    const numLayers = modelConfig.num_hidden_layers || 12;
    
    // Calculate layers per node
    const layersPerNode = Math.ceil(numLayers / allNodes.length);
    
    // Assign layers to nodes
    for (let i = 0; i < numLayers; i++) {
      const assignedNodeIndex = Math.floor(i / layersPerNode) % allNodes.length;
      partitioning[`transformer_block_${i}`] = allNodes[assignedNodeIndex];
    }

    return partitioning;
  }

  /**
   * Calculate the communication cost of layer parallelism
   * @param {Object} partitioning The partitioning scheme
   * @param {Object} modelConfig The model configuration
   * @returns {Object} Communication cost metrics
   */
  calculateCommunicationCost(partitioning, modelConfig) {
    // Simplified cost model:
    // - Each transition between nodes requires transferring the hidden states
    // - Cost is proportional to hidden size * batch size * sequence length
    
    const hiddenSize = modelConfig.hidden_size || 768;
    const batchSize = this.options.batchSize || 1;
    const seqLength = this.options.seqLength || 1024;
    
    // Count transitions between nodes
    let transitions = 0;
    let currentNode = null;
    
    // Sort layer names to find correct sequence
    const layerNames = Object.keys(partitioning)
      .filter(name => name.startsWith('transformer_block_'))
      .sort((a, b) => {
        const numA = parseInt(a.split('_').pop());
        const numB = parseInt(b.split('_').pop());
        return numA - numB;
      });
    
    for (const layerName of layerNames) {
      const nodeId = partitioning[layerName];
      if (currentNode && nodeId !== currentNode) {
        transitions++;
      }
      currentNode = nodeId;
    }
    
    // Data transferred per transition (in elements)
    const dataPerTransition = hiddenSize * batchSize * seqLength;
    
    // Total data transferred (in elements)
    const totalDataTransferred = transitions * dataPerTransition;
    
    // Assuming 4 bytes per element (float32)
    const totalDataBytes = totalDataTransferred * 4;
    
    return {
      transitions,
      dataPerTransition,
      totalDataTransferred,
      totalDataBytes,
      totalDataMB: totalDataBytes / (1024 * 1024)
    };
  }
}

/**
 * Tensor parallelism strategy for attention heads
 * Splits attention operations across multiple nodes
 */
export class TensorParallelStrategy extends ParallelismStrategy {
  constructor(options = {}) {
    super({
      minNodesRequired: 2,
      ...options,
    });
  }

  /**
   * Partition the model by splitting attention heads and MLP layers
   * @param {Object} modelConfig The model configuration
   * @param {Array<string>} nodeIds Available node IDs
   * @returns {Object} Mapping of component names to node IDs
   */
  partition(modelConfig, nodeIds) {
    if (!nodeIds || nodeIds.length < this.options.minNodesRequired) {
      throw new Error(`At least ${this.options.minNodesRequired} nodes required for tensor parallelism`);
    }

    const localNodeId = this.options.localNodeId;
    if (!localNodeId) {
      throw new Error('Local node ID is required');
    }

    // Include local node in the available nodes
    const allNodes = [...nodeIds];
    if (!allNodes.includes(localNodeId)) {
      allNodes.push(localNodeId);
    }

    const partitioning = {};
    const numLayers = modelConfig.num_hidden_layers || 12;
    const numAttentionHeads = modelConfig.num_attention_heads || 12;
    
    // Embeddings and final layer are on the local node
    partitioning['embeddings'] = localNodeId;
    partitioning['lm_head'] = localNodeId;
    partitioning['final_layernorm'] = localNodeId;
    
    // For each transformer layer, split attention heads and MLP across nodes
    for (let i = 0; i < numLayers; i++) {
      // For each attention head in this layer
      const headsPerNode = Math.ceil(numAttentionHeads / allNodes.length);
      
      for (let j = 0; j < numAttentionHeads; j++) {
        const nodeIndex = Math.floor(j / headsPerNode) % allNodes.length;
        partitioning[`transformer_block_${i}_attention_head_${j}`] = allNodes[nodeIndex];
      }
      
      // Split MLP sublayers across nodes
      // Use interleaving to balance computation
      partitioning[`transformer_block_${i}_mlp_fc1`] = allNodes[i % allNodes.length];
      partitioning[`transformer_block_${i}_mlp_fc2`] = allNodes[(i + 1) % allNodes.length];
      
      // Layer normalization operations on the local node for simplicity
      partitioning[`transformer_block_${i}_ln1`] = localNodeId;
      partitioning[`transformer_block_${i}_ln2`] = localNodeId;
    }

    return partitioning;
  }

  /**
   * Calculate the communication cost of tensor parallelism
   * @param {Object} partitioning The partitioning scheme
   * @param {Object} modelConfig The model configuration
   * @returns {Object} Communication cost metrics
   */
  calculateCommunicationCost(partitioning, modelConfig) {
    // In tensor parallelism, we need:
    // 1. All-to-all communication after attention heads computation
    // 2. All-to-all communication after MLP layers
    
    const hiddenSize = modelConfig.hidden_size || 768;
    const batchSize = this.options.batchSize || 1;
    const seqLength = this.options.seqLength || 1024;
    const numLayers = modelConfig.num_hidden_layers || 12;
    
    // Nodes involved in computation
    const uniqueNodeIds = new Set(Object.values(partitioning));
    const numNodes = uniqueNodeIds.size;
    
    // All-to-all communication cost per layer:
    // Each node sends data to all other nodes
    const allToAllCostPerLayer = (numNodes - 1) * (hiddenSize * batchSize * seqLength / numNodes);
    
    // We do this twice per layer (attention + MLP)
    const totalAllToAllCost = numLayers * 2 * allToAllCostPerLayer;
    
    // Assuming 4 bytes per element (float32)
    const totalDataBytes = totalAllToAllCost * 4;
    
    return {
      numNodes,
      allToAllCostPerLayer,
      totalAllToAllCost,
      totalDataBytes,
      totalDataMB: totalDataBytes / (1024 * 1024)
    };
  }
}

/**
 * Pipeline parallelism strategy
 * Builds a pipeline of nodes processing different microbatches
 */
export class PipelineParallelStrategy extends ParallelismStrategy {
  constructor(options = {}) {
    super({
      numMicrobatches: 4,
      ...options,
    });
  }

  /**
   * Partition the model in a pipeline fashion
   * @param {Object} modelConfig The model configuration
   * @param {Array<string>} nodeIds Available node IDs
   * @returns {Object} Mapping of layer ranges to node IDs
   */
  partition(modelConfig, nodeIds) {
    if (!nodeIds || nodeIds.length === 0) {
      throw new Error('No nodes available for partitioning');
    }

    const localNodeId = this.options.localNodeId;
    if (!localNodeId) {
      throw new Error('Local node ID is required');
    }

    // Include local node in the available nodes
    const allNodes = [...nodeIds];
    if (!allNodes.includes(localNodeId)) {
      allNodes.push(localNodeId);
    }

    const partitioning = {
      type: 'pipeline',
      microbatches: this.options.numMicrobatches,
      stages: []
    };
    
    const numLayers = modelConfig.num_hidden_layers || 12;
    const layersPerNode = Math.ceil(numLayers / allNodes.length);
    
    // Create pipeline stages
    for (let i = 0; i < allNodes.length; i++) {
      const startLayer = i * layersPerNode;
      const endLayer = Math.min((i + 1) * layersPerNode - 1, numLayers - 1);
      
      // Skip if no layers to assign
      if (startLayer > numLayers - 1) continue;
      
      partitioning.stages.push({
        nodeId: allNodes[i],
        startLayer,
        endLayer,
        // First stage includes embeddings
        includesEmbeddings: i === 0,
        // Last stage includes final layer norm and output
        includesOutput: endLayer === numLayers - 1
      });
    }

    return partitioning;
  }

  /**
   * Calculate the communication cost of pipeline parallelism
   * @param {Object} partitioning The partitioning scheme
   * @param {Object} modelConfig The model configuration
   * @returns {Object} Communication cost metrics
   */
  calculateCommunicationCost(partitioning, modelConfig) {
    const hiddenSize = modelConfig.hidden_size || 768;
    const batchSize = this.options.batchSize || 1;
    const seqLength = this.options.seqLength || 1024;
    const microbatches = partitioning.microbatches;
    
    // Number of pipeline stages
    const numStages = partitioning.stages.length;
    
    // In pipeline parallelism, the communication happens between adjacent stages
    // Each microbatch transitions between stages (numStages - 1) times
    const transitionsPerMicrobatch = numStages - 1;
    
    // Data transferred per transition (in elements)
    // Size is divided by microbatches
    const dataPerTransition = hiddenSize * (batchSize / microbatches) * seqLength;
    
    // Total transitions
    const totalTransitions = microbatches * transitionsPerMicrobatch;
    
    // Total data transferred (in elements)
    const totalDataTransferred = totalTransitions * dataPerTransition;
    
    // Assuming 4 bytes per element (float32)
    const totalDataBytes = totalDataTransferred * 4;
    
    return {
      numStages,
      microbatches,
      transitionsPerMicrobatch,
      totalTransitions,
      dataPerTransition,
      totalDataTransferred,
      totalDataBytes,
      totalDataMB: totalDataBytes / (1024 * 1024)
    };
  }
}

/**
 * Factory function to create a strategy based on type
 * @param {string} type Strategy type
 * @param {Object} options Strategy options
 * @returns {ParallelismStrategy} Strategy instance
 */
export function createStrategy(type, options = {}) {
  switch (type) {
    case StrategyType.LAYER_PARALLEL:
      return new LayerParallelStrategy(options);
    case StrategyType.TENSOR_PARALLEL:
      return new TensorParallelStrategy(options);
    case StrategyType.PIPELINE_PARALLEL:
      return new PipelineParallelStrategy(options);
    default:
      throw new Error(`Unknown strategy type: ${type}`);
  }
}

export default {
  StrategyType,
  createStrategy,
  LayerParallelStrategy,
  TensorParallelStrategy,
  PipelineParallelStrategy
}; 