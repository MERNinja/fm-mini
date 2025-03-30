/**
 * Test script for tensor parallelism
 * Deploys 3 headless clients and tests tensor parallelism with a sample prompt
 */
import { HeadlessClient } from './headless-client.js';
import { StrategyType } from '../../app/utils/parallelStrategy.js';
import fs from 'fs';
import path from 'path';

// Constants
const NUM_NODES = 3;
const SERVER_URL = 'http://localhost:8099';
const MODEL_ID = 'mock-test-model'; // Use mock model for testing
const MODEL_DIR = path.resolve('./models');
const TEST_PROMPT = 'Write me a short poem about distributed computing.';
const DEPLOYMENT_DELAY = 2000; // ms to wait between node deployments

// Check for available models and verify the specified model exists
function checkModelAvailability() {
  const mockModelDir = path.join(MODEL_DIR, 'mock-test-model');
  const mockModelExists = fs.existsSync(path.join(mockModelDir, 'mock-test-model.wasm'));
  
  const tinyModelDir = path.join(MODEL_DIR, 'tiny-test-model');
  const tinyModelExists = fs.existsSync(path.join(tinyModelDir, 'tiny-test-model.wasm'));
  
  const tinyllamaDir = path.join(MODEL_DIR, 'tinyllama-1.1b-chat-v1.0');
  const tinyllamaExists = fs.existsSync(path.join(tinyllamaDir, 'tinyllama-1.1b-chat-v1.0.wasm'));
  
  // Check models in order of preference: mock > tiny > tinyllama
  if (mockModelExists) {
    console.log('Using mock-test-model for testing');
    return 'mock-test-model';
  } else if (tinyModelExists) {
    console.log('Mock model not found, using tiny-test-model instead');
    return 'tiny-test-model';
  } else if (tinyllamaExists) {
    console.log('Mock and tiny models not found, using tinyllama-1.1b-chat-v1.0 instead');
    return 'tinyllama-1.1b-chat-v1.0';
  } else {
    console.error('No test models found!');
    console.error('Please run one of these commands to create/download a test model:');
    console.error('  npm run create:mock-model   # For a local mock model (fastest)');
    console.error('  npm run download:tiny-model # For a small test model');
    console.error('  npm run download:models     # For the complete tinyllama model');
    process.exit(1);
  }
}

// Ensure model directory exists
if (!fs.existsSync(MODEL_DIR)) {
  fs.mkdirSync(MODEL_DIR, { recursive: true });
  console.log(`Created model directory: ${MODEL_DIR}`);
}

// Check model availability
const activeModelId = checkModelAvailability();

/**
 * Store test results
 */
const testResults = {
  startTime: new Date().toISOString(),
  prompt: TEST_PROMPT,
  modelId: activeModelId,
  nodes: [],
  responses: [],
  parallelEnabled: false,
  errors: []
};

/**
 * Deploy a single headless client
 */
async function deployNode(nodeId, isTensorLeader = false) {
  try {
    console.log(`Deploying node ${nodeId}...`);
    
    const client = new HeadlessClient({
      id: `node_${nodeId}`,
      serverUrl: SERVER_URL,
      modelId: activeModelId,
      modelDir: MODEL_DIR
    });
    
    // Connect to server
    await client.connect();
    console.log(`Node ${nodeId} connected`);
    
    // Load model
    await client.loadModel();
    console.log(`Node ${nodeId} model loaded`);
    
    // Record node in test results
    testResults.nodes.push({
      id: client.config.id,
      status: 'ready'
    });
    
    return client;
  } catch (error) {
    console.error(`Error deploying node ${nodeId}:`, error);
    testResults.errors.push({
      node: `node_${nodeId}`,
      error: error.message,
      phase: 'deployment'
    });
    throw error;
  }
}

/**
 * Run test with tensor parallelism
 */
async function runTest() {
  let clients = [];
  
  try {
    console.log('=== Starting Tensor Parallelism Test ===');
    console.log(`Using model: ${activeModelId}`);
    console.log(`Deploying ${NUM_NODES} headless clients...`);
    
    // Deploy nodes sequentially
    for (let i = 0; i < NUM_NODES; i++) {
      const client = await deployNode(i + 1, i === 0); // First node is tensor leader
      clients.push(client);
      
      // Wait between deployments to allow registration
      if (i < NUM_NODES - 1) {
        await new Promise(resolve => setTimeout(resolve, DEPLOYMENT_DELAY));
      }
    }
    
    console.log('All nodes deployed successfully');
    
    // The first client will be designated as the leader
    const leaderClient = clients[0];
    
    // Enable tensor parallelism on the leader
    console.log('Enabling tensor parallelism on leader node...');
    const parallelEnabled = await leaderClient.enableTensorParallel();
    testResults.parallelEnabled = parallelEnabled;
    
    if (parallelEnabled) {
      console.log('Tensor parallelism enabled successfully');
      
      // Set strategy to TENSOR_PARALLEL (splitting attention heads)
      leaderClient.setParallelStrategy(StrategyType.TENSOR_PARALLEL);
      console.log('Using TENSOR_PARALLEL strategy');
      
      // Get tensor status
      const status = leaderClient.llmEngine.tensorParallel.getStatus();
      console.log('Tensor status:', status);
      testResults.tensorStatus = status;
      
      // Performance metrics
      const metrics = leaderClient.llmEngine.tensorParallel.getPerformanceMetrics();
      console.log('Performance metrics:', metrics);
      testResults.performanceMetrics = metrics;
    } else {
      console.log('Failed to enable tensor parallelism, continuing with single node');
    }
    
    // Send the test prompt from the leader
    console.log(`\nSending test prompt: "${TEST_PROMPT}"`);
    const startTime = Date.now();
    const result = await leaderClient.sendChat(TEST_PROMPT);
    const endTime = Date.now();
    
    console.log('\n=== Test Results ===');
    console.log(`Response (generated in ${endTime - startTime}ms):`);
    console.log(result.response);
    console.log('\nProcessing stats:', result.stats);
    
    // Record test results
    testResults.responses.push({
      node: leaderClient.config.id,
      response: result.response,
      stats: result.stats
    });
    
    // Save results to JSON file
    const testReport = JSON.stringify(testResults, null, 2);
    fs.writeFileSync('./tests/tensor_parallel/test-results.json', testReport);
    console.log('\nTest results saved to test-results.json');
    
  } catch (error) {
    console.error('Test error:', error);
    testResults.errors.push({
      phase: 'test',
      error: error.message
    });
  } finally {
    // Clean up: disconnect all clients
    console.log('\nCleaning up...');
    for (const client of clients) {
      if (client) {
        client.disconnect();
      }
    }
    console.log('All nodes disconnected');
    
    // Save final results even if there were errors
    if (testResults.errors.length > 0) {
      const testReport = JSON.stringify(testResults, null, 2);
      fs.writeFileSync('./tests/tensor_parallel/test-results-with-errors.json', testReport);
      console.log('Error details saved to test-results-with-errors.json');
    }
    
    console.log('=== Test Completed ===');
  }
}

// Run the test
runTest(); 