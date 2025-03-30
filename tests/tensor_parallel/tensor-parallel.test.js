/**
 * Jest tests for tensor parallelism
 */
import { HeadlessClient } from './headless-client.js';
import { StrategyType } from '../../app/utils/parallelStrategy.js';
import path from 'path';
import fs from 'fs';

// Constants
const SERVER_URL = 'http://localhost:8099';
const MODEL_ID = 'mock-test-model'; // Use mock model for testing
const MODEL_DIR = path.resolve('./models');
const TEST_PROMPTS = [
  'Write me a short poem about tensor parallelism.',
  'Explain how distributed computing works in 3 sentences.',
  'Complete this phrase: Tensor parallelism helps accelerate...'
];

// Check for available models and use the one that's available
function getAvailableModel() {
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
    console.log('Using tiny-test-model for tensor parallelism tests');
    return 'tiny-test-model';
  } else if (tinyllamaExists) {
    console.log('Using tinyllama-1.1b-chat-v1.0 model for tensor parallelism tests');
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

// Global variables for clients
let clients = [];
let leaderClient = null;
let activeModelId = null;

// Setup function to be run before all tests
beforeAll(async () => {
  // Ensure model directory exists
  if (!fs.existsSync(MODEL_DIR)) {
    fs.mkdirSync(MODEL_DIR, { recursive: true });
    console.log(`Created model directory: ${MODEL_DIR}`);
  }
  
  // Get available model
  activeModelId = getAvailableModel();
  
  // Deploy 3 clients
  console.log('Deploying 3 headless clients...');
  
  try {
    // Create clients
    for (let i = 0; i < 3; i++) {
      const client = new HeadlessClient({
        id: `test_node_${i + 1}`,
        serverUrl: SERVER_URL,
        modelId: activeModelId,
        modelDir: MODEL_DIR
      });
      
      clients.push(client);
    }
    
    // Connect and load models in parallel
    await Promise.all(clients.map(async (client) => {
      await client.connect();
      await client.loadModel();
      console.log(`Client ${client.config.id} connected and model loaded`);
    }));
    
    // Set the first client as the leader
    leaderClient = clients[0];
    
  } catch (error) {
    console.error('Setup error:', error);
    throw error;
  }
}, 300000); // 5 minute timeout for setup

// Cleanup function to be run after all tests
afterAll(async () => {
  // Disconnect all clients
  console.log('Cleaning up clients...');
  for (const client of clients) {
    if (client) {
      client.disconnect();
    }
  }
  
  clients = [];
  leaderClient = null;
}, 10000);

describe('Tensor Parallelism', () => {
  
  test('All clients should be connected and ready', () => {
    // Check if all clients are initialized
    for (const client of clients) {
      expect(client.socket).not.toBeNull();
      expect(client.llmEngine).not.toBeNull();
      expect(client.modelInfo.loaded).toBe(true);
      expect(client.modelInfo.status).toBe('ready');
    }
  });
  
  test('Leader should be able to enable tensor parallelism', async () => {
    // Enable tensor parallelism on the leader
    const parallelEnabled = await leaderClient.enableTensorParallel();
    
    // Should be enabled since we have multiple nodes
    expect(parallelEnabled).toBe(true);
    expect(leaderClient.isParallelEnabled).toBe(true);
    
    // Check tensor status
    const status = leaderClient.llmEngine.tensorParallel.getStatus();
    expect(status.enabled).toBe(true);
    expect(status.connectedPeers.length).toBeGreaterThan(0);
    
    // Record status for reporting
    console.log('Tensor status:', status);
  }, 30000);
  
  test('Leader should be able to set TENSOR_PARALLEL strategy', () => {
    // Set strategy to TENSOR_PARALLEL (splitting attention heads)
    const result = leaderClient.setParallelStrategy(StrategyType.TENSOR_PARALLEL);
    
    expect(result).toBe(true);
    
    // Verify strategy was set
    const status = leaderClient.llmEngine.tensorParallel.getStatus();
    expect(status.strategy).toBe(StrategyType.TENSOR_PARALLEL);
  });
  
  test('Leader should get performance metrics', () => {
    // Get performance metrics
    const metrics = leaderClient.llmEngine.tensorParallel.getPerformanceMetrics();
    
    // Should have valid metrics
    expect(metrics).toBeDefined();
    expect(metrics.error).toBeUndefined();
    
    // Record metrics for reporting
    console.log('Performance metrics:', metrics);
  });
  
  test('Leader should process prompts with tensor parallelism', async () => {
    // Process a short prompt
    const prompt = TEST_PROMPTS[0];
    const result = await leaderClient.sendChat(prompt);
    
    // Should have a valid response
    expect(result).toBeDefined();
    expect(result.response).toBeDefined();
    expect(result.response.length).toBeGreaterThan(0);
    
    // Should have valid stats
    expect(result.stats).toBeDefined();
    expect(result.stats.tokens).toBeDefined();
    expect(result.stats.tokens.total).toBeGreaterThan(0);
    
    // Should indicate parallelism was used
    expect(result.stats.parallelism.enabled).toBe(true);
    expect(result.stats.parallelism.nodes).toBeGreaterThan(0);
    
    // Record response for reporting
    console.log('Prompt:', prompt);
    console.log('Response:', result.response);
    console.log('Stats:', result.stats);
  }, 60000);
  
  test('Leader should be able to process multiple prompts', async () => {
    // Skip the first prompt as we already tested it
    const prompts = TEST_PROMPTS.slice(1);
    
    // Process each prompt
    for (const prompt of prompts) {
      const result = await leaderClient.sendChat(prompt);
      
      // Should have a valid response
      expect(result).toBeDefined();
      expect(result.response).toBeDefined();
      expect(result.response.length).toBeGreaterThan(0);
      
      // Record response for reporting
      console.log('Prompt:', prompt);
      console.log('Response:', result.response);
    }
  }, 120000);
  
  test('Leader should be able to disable tensor parallelism', () => {
    // Disable tensor parallelism
    const result = leaderClient.llmEngine.tensorParallel.disable();
    
    expect(result).toBe(true);
    expect(leaderClient.isParallelEnabled).toBe(false);
    
    // Verify it was disabled
    const status = leaderClient.llmEngine.tensorParallel.getStatus();
    expect(status.enabled).toBe(false);
  });
}); 