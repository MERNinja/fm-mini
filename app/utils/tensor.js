/**
 * Tensor operations for distributed model parallelism
 * Handles tensor partitioning, merging, and serialization
 */

/**
 * Serialize a tensor to binary format
 * @param {Float32Array|Int32Array} tensor - Tensor data
 * @param {Array<number>} shape - Tensor shape
 * @param {string} dtype - Data type ('float32', 'int32', etc.)
 * @returns {ArrayBuffer} - Serialized tensor
 */
export const serializeTensor = (tensor, shape, dtype) => {
    // Calculate total size
    const totalSize = shape.reduce((a, b) => a * b, 1);
    if (tensor.length !== totalSize) {
        throw new Error(`Tensor length (${tensor.length}) doesn't match shape (${shape.join('x')})`);
    }

    // Create buffer with extra space for metadata
    const metadataSize = 4 + (4 * shape.length) + 4; // Format: [dtype_len(4), shape_dims(4*n), tensor_type(4)]
    const buffer = new ArrayBuffer(metadataSize + tensor.byteLength);
    const view = new DataView(buffer);

    // Set dtype string length
    const encoder = new TextEncoder();
    const dtypeEncoded = encoder.encode(dtype);
    view.setUint32(0, dtypeEncoded.length, true);

    // Set shape dimensions
    view.setUint32(4, shape.length, true);
    shape.forEach((dim, i) => {
        view.setUint32(8 + (i * 4), dim, true);
    });

    // Set tensor type (0 for int32, 1 for float32)
    const tensorType = dtype === 'float32' ? 1 : 0;
    view.setUint32(8 + (shape.length * 4), tensorType, true);

    // Copy tensor data
    const dataView = new DataView(buffer, metadataSize);
    if (dtype === 'float32') {
        const floatArray = new Float32Array(buffer, metadataSize, totalSize);
        floatArray.set(tensor);
    } else if (dtype === 'int32') {
        const intArray = new Int32Array(buffer, metadataSize, totalSize);
        intArray.set(tensor);
    } else {
        throw new Error(`Unsupported data type: ${dtype}`);
    }

    return buffer;
};

/**
 * Deserialize a binary tensor
 * @param {ArrayBuffer} buffer - Serialized tensor
 * @returns {Object} - Deserialized tensor with data, shape, and dtype
 */
export const deserializeTensor = (buffer) => {
    const view = new DataView(buffer);

    // Get dtype
    const dtypeLength = view.getUint32(0, true);
    const dtypeBytes = new Uint8Array(buffer, 4, dtypeLength);
    const decoder = new TextDecoder();
    const dtype = decoder.decode(dtypeBytes);

    // Get shape
    const shapeLength = view.getUint32(4 + dtypeLength, true);
    const shape = [];
    for (let i = 0; i < shapeLength; i++) {
        shape.push(view.getUint32(8 + dtypeLength + (i * 4), true));
    }

    // Get tensor type
    const tensorType = view.getUint32(8 + dtypeLength + (shapeLength * 4), true);

    // Calculate data offset
    const metadataSize = 4 + dtypeLength + 4 + (shapeLength * 4) + 4;

    // Create appropriate typed array
    let data;
    if (tensorType === 1) { // float32
        data = new Float32Array(buffer, metadataSize);
    } else if (tensorType === 0) { // int32
        data = new Int32Array(buffer, metadataSize);
    } else {
        throw new Error(`Unsupported tensor type: ${tensorType}`);
    }

    return { data, shape, dtype };
};

/**
 * Split a tensor along a specified dimension
 * @param {Float32Array|Int32Array} tensor - Input tensor
 * @param {Array<number>} shape - Tensor shape
 * @param {number} dim - Dimension to split along
 * @param {number} numParts - Number of parts to split into
 * @returns {Array<Object>} - Array of split tensors with their shapes
 */
export const splitTensor = (tensor, shape, dim, numParts) => {
    if (dim >= shape.length) {
        throw new Error(`Dimension ${dim} is out of bounds for shape ${shape}`);
    }

    // Calculate split size
    const dimSize = shape[dim];
    const splitSize = Math.floor(dimSize / numParts);
    const remainder = dimSize % numParts;

    // Calculate strides
    const strides = new Array(shape.length).fill(1);
    for (let i = shape.length - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    const result = [];
    let offset = 0;

    for (let i = 0; i < numParts; i++) {
        // Calculate the size for this split (handling remainder)
        const currentSplitSize = splitSize + (i < remainder ? 1 : 0);
        if (currentSplitSize === 0) continue; // Skip empty splits

        // Create new shape with split dimension
        const newShape = [...shape];
        newShape[dim] = currentSplitSize;

        // Calculate size of this split
        const totalSize = newShape.reduce((a, b) => a * b, 1);
        const newTensor = tensor.constructor === Float32Array
            ? new Float32Array(totalSize)
            : new Int32Array(totalSize);

        // Copy data to the new tensor
        if (dim === shape.length - 1) {
            // Fast path for last dimension
            const sliceStart = offset;
            const sliceEnd = offset + currentSplitSize * strides[dim];
            newTensor.set(tensor.slice(sliceStart, sliceEnd));
            offset = sliceEnd;
        } else {
            // General case
            let srcIdx = 0;
            let dstIdx = 0;

            // Helper function for recursive copying
            const copySubTensor = (curDim, srcOffset, dstOffset, dimIndices) => {
                if (curDim === shape.length) {
                    // Copy a single element
                    newTensor[dstOffset] = tensor[srcOffset];
                    return;
                }

                if (curDim === dim) {
                    // At the split dimension, only iterate over the assigned part
                    for (let j = 0; j < currentSplitSize; j++) {
                        const newDimIndices = [...dimIndices, j];
                        copySubTensor(
                            curDim + 1,
                            srcOffset + (i * splitSize + j) * strides[curDim],
                            dstOffset + j * strides[curDim],
                            newDimIndices
                        );
                    }
                } else {
                    // For other dimensions, iterate over all indices
                    for (let j = 0; j < shape[curDim]; j++) {
                        const newDimIndices = [...dimIndices, j];
                        copySubTensor(
                            curDim + 1,
                            srcOffset + j * strides[curDim],
                            dstOffset + j * strides[curDim],
                            newDimIndices
                        );
                    }
                }
            };

            copySubTensor(0, 0, 0, []);
        }

        result.push({
            tensor: newTensor,
            shape: newShape,
            offset: [i * splitSize, currentSplitSize]
        });
    }

    return result;
};

/**
 * Merge split tensors back into a single tensor
 * @param {Array<Object>} splitTensors - Array of split tensors with their shapes
 * @param {Array<number>} originalShape - Original shape of the tensor
 * @param {number} dim - Dimension that was split
 * @returns {Object} - Merged tensor with its shape
 */
export const mergeTensors = (splitTensors, originalShape, dim) => {
    // Calculate total size of the merged tensor
    const totalSize = originalShape.reduce((a, b) => a * b, 1);

    // Create the merged tensor
    const firstTensor = splitTensors[0].tensor;
    const mergedTensor = firstTensor.constructor === Float32Array
        ? new Float32Array(totalSize)
        : new Int32Array(totalSize);

    // Calculate strides for indexing
    const strides = new Array(originalShape.length).fill(1);
    for (let i = originalShape.length - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * originalShape[i + 1];
    }

    // Copy data from each split tensor
    for (const { tensor, shape, offset } of splitTensors) {
        const [startIdx, length] = offset;

        if (dim === originalShape.length - 1) {
            // Fast path for last dimension
            const destOffset = startIdx;
            for (let i = 0; i < tensor.length; i++) {
                mergedTensor[destOffset + i] = tensor[i];
            }
        } else {
            // General case requires mapping indices
            const copySubTensor = (curDim, srcOffset, dstOffset, dimIndices) => {
                if (curDim === shape.length) {
                    // Copy a single element
                    mergedTensor[dstOffset] = tensor[srcOffset];
                    return;
                }

                if (curDim === dim) {
                    // At the split dimension, map indices to the original tensor
                    for (let j = 0; j < shape[curDim]; j++) {
                        const newDimIndices = [...dimIndices, j];
                        copySubTensor(
                            curDim + 1,
                            srcOffset + j * strides[curDim],
                            dstOffset + (j + startIdx) * strides[curDim],
                            newDimIndices
                        );
                    }
                } else {
                    // For other dimensions, iterate over all indices
                    for (let j = 0; j < shape[curDim]; j++) {
                        const newDimIndices = [...dimIndices, j];
                        copySubTensor(
                            curDim + 1,
                            srcOffset + j * strides[curDim],
                            dstOffset + j * strides[curDim],
                            newDimIndices
                        );
                    }
                }
            };

            copySubTensor(0, 0, 0, []);
        }
    }

    return { tensor: mergedTensor, shape: originalShape };
};

/**
 * Distribute tensors among available nodes
 * @param {Object} modelWeights - Object containing model weights
 * @param {Array<Object>} nodes - Available compute nodes
 * @returns {Object} - Distribution plan with node assignments
 */
export const createDistributionPlan = (modelWeights, nodes) => {
    if (!nodes || nodes.length === 0) {
        throw new Error('No nodes available for distribution');
    }

    const weightKeys = Object.keys(modelWeights);
    const numNodes = nodes.length;
    const plan = {};

    // Simple round-robin distribution for now
    // More sophisticated strategies can be implemented based on node capabilities
    weightKeys.forEach((key, index) => {
        const nodeIndex = index % numNodes;
        const nodeId = nodes[nodeIndex].id;

        if (!plan[nodeId]) {
            plan[nodeId] = [];
        }

        plan[nodeId].push(key);
    });

    return plan;
};

/**
 * Split attention heads across nodes
 * @param {Float32Array} qkvWeights - QKV weight matrix (3 x hidden_size x num_heads x head_dim)
 * @param {Array<number>} shape - Shape of the weight matrix
 * @param {number} numNodes - Number of nodes to distribute to
 * @returns {Array<Object>} - Split tensors with metadata
 */
export const splitAttentionHeads = (qkvWeights, shape, numNodes) => {
    // Assuming shape is [3, hidden_size, num_heads, head_dim]
    // We'll split along the num_heads dimension (dim=2)
    return splitTensor(qkvWeights, shape, 2, numNodes);
};

/**
 * Split MLP layer across nodes
 * @param {Float32Array} mlpWeights - MLP weight matrix
 * @param {Array<number>} shape - Shape of the weight matrix
 * @param {number} numNodes - Number of nodes to distribute to
 * @returns {Array<Object>} - Split tensors with metadata
 */
export const splitMLPLayer = (mlpWeights, shape, numNodes) => {
    // Assuming shape is [hidden_size, intermediate_size]
    // We'll split along the intermediate_size dimension (dim=1)
    return splitTensor(mlpWeights, shape, 1, numNodes);
};

/**
 * Helper to determine if a layer should be parallelized
 * @param {string} layerName - Name of the layer
 * @returns {boolean} - Whether the layer can be parallelized
 */
export const isParallelizableLayer = (layerName) => {
    // Check for attention or MLP layers which are suitable for parallelism
    return (
        layerName.includes('attention') ||
        layerName.includes('mlp') ||
        layerName.includes('feed_forward')
    );
};

/**
 * Determine parallelism strategy for a layer
 * @param {string} layerName - Name of the layer
 * @returns {string} - Strategy ('attention-parallel', 'mlp-parallel', or 'none')
 */
export const getParallelismStrategy = (layerName) => {
    if (layerName.includes('attention')) {
        return 'attention-parallel';
    } else if (layerName.includes('mlp') || layerName.includes('feed_forward')) {
        return 'mlp-parallel';
    } else {
        return 'none';
    }
};

/**
 * Create a worker assignment based on node capabilities
 * @param {Array<Object>} nodes - Available nodes with capabilities
 * @param {Object} modelLayers - Model layers to distribute
 * @returns {Object} - Node to layer assignments
 */
export const createWorkerAssignment = (nodes, modelLayers) => {
    const assignment = {};
    const layerKeys = Object.keys(modelLayers);

    // Prioritize nodes with better capabilities
    const sortedNodes = [...nodes].sort((a, b) => {
        // Sort by GPU capability, then by CPU speed
        if (a.gpuMemory !== b.gpuMemory) {
            return b.gpuMemory - a.gpuMemory;
        }
        return b.cpuCores - a.cpuCores;
    });

    // Group layers by type for better distribution
    const attentionLayers = [];
    const mlpLayers = [];
    const otherLayers = [];

    layerKeys.forEach(key => {
        const strategy = getParallelismStrategy(key);
        if (strategy === 'attention-parallel') {
            attentionLayers.push(key);
        } else if (strategy === 'mlp-parallel') {
            mlpLayers.push(key);
        } else {
            otherLayers.push(key);
        }
    });

    // Distribute attention layers first (most parallelizable)
    attentionLayers.forEach((layer, index) => {
        const nodeIndex = index % sortedNodes.length;
        const nodeId = sortedNodes[nodeIndex].id;

        if (!assignment[nodeId]) {
            assignment[nodeId] = [];
        }

        assignment[nodeId].push({
            layer,
            strategy: 'attention-parallel'
        });
    });

    // Then distribute MLP layers
    mlpLayers.forEach((layer, index) => {
        const nodeIndex = index % sortedNodes.length;
        const nodeId = sortedNodes[nodeIndex].id;

        if (!assignment[nodeId]) {
            assignment[nodeId] = [];
        }

        assignment[nodeId].push({
            layer,
            strategy: 'mlp-parallel'
        });
    });

    // Finally, distribute other layers
    otherLayers.forEach((layer, index) => {
        const nodeIndex = index % sortedNodes.length;
        const nodeId = sortedNodes[nodeIndex].id;

        if (!assignment[nodeId]) {
            assignment[nodeId] = [];
        }

        assignment[nodeId].push({
            layer,
            strategy: 'none'
        });
    });

    return assignment;
}; 