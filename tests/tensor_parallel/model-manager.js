/**
 * Model Manager for WebLLM testing
 * Downloads and caches small models for testing tensor parallelism
 */
import fs from 'fs';
import path from 'path';
import { pipeline } from 'stream/promises';
import { createWriteStream } from 'fs';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';

// Get __dirname in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Model information - Using public models from WebLLM
const TEST_MODELS = {
  'mock-test-model': {
    url: 'local-only',
    configUrl: 'local-only',
    size: '~1MB'
  },
  'tinyllama-1.1b-chat-v1.0': {
    url: 'https://huggingface.co/mlc-ai/mlc-chat-TinyLlama-1.1B-Chat-v1.0-q4f16_1/resolve/main/TinyLlama-1.1B-Chat-v1.0-q4f16_1.wasm',
    configUrl: 'https://huggingface.co/mlc-ai/mlc-chat-TinyLlama-1.1B-Chat-v1.0-q4f16_1/resolve/main/TinyLlama-1.1B-Chat-v1.0-q4f16_1.json',
    size: '~560MB'
  },
  'redpajama-3b': {
    url: 'https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_1/resolve/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1.wasm',
    configUrl: 'https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_1/resolve/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1.json',
    size: '~820MB'
  }
};

// User agent for requests
const USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36';

/**
 * ModelManager class for handling WebLLM models
 */
export class ModelManager {
  constructor(options = {}) {
    this.modelDir = options.modelDir || path.resolve(__dirname, '../../models');
    this.defaultModel = options.defaultModel || 'tinyllama-1.1b-chat-v1.0';
  }
  
  /**
   * Initialize the model directory
   */
  async init() {
    try {
      // Create model directory if it doesn't exist
      if (!fs.existsSync(this.modelDir)) {
        fs.mkdirSync(this.modelDir, { recursive: true });
        console.log(`Created model directory: ${this.modelDir}`);
      }
      
      return this;
    } catch (error) {
      console.error(`Error initializing model directory: ${error.message}`);
      throw error;
    }
  }
  
  /**
   * Get available models
   */
  getAvailableModels() {
    return Object.keys(TEST_MODELS);
  }
  
  /**
   * Check if a model is already downloaded
   * @param {string} modelId The model ID
   */
  isModelDownloaded(modelId) {
    if (!TEST_MODELS[modelId]) {
      throw new Error(`Model ${modelId} not found in TEST_MODELS`);
    }
    
    const modelPath = path.join(this.modelDir, modelId);
    const wasmFile = path.join(modelPath, `${modelId}.wasm`);
    const configFile = path.join(modelPath, `${modelId}.json`);
    
    return fs.existsSync(wasmFile) && fs.existsSync(configFile);
  }
  
  /**
   * Download a file using fetch
   * @param {string} url URL to download from
   * @param {string} filePath Path to save the file
   */
  async downloadFile(url, filePath) {
    console.log(`Downloading from ${url}...`);
    
    try {
      const response = await fetch(url, {
        headers: {
          'User-Agent': USER_AGENT
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to download ${url}: ${response.statusText} (${response.status})`);
      }
      
      const fileStream = createWriteStream(filePath);
      await new Promise((resolve, reject) => {
        response.body.pipe(fileStream);
        response.body.on('error', (error) => {
          console.error('Stream error:', error);
          reject(error);
        });
        fileStream.on('error', (error) => {
          console.error('File write error:', error);
          reject(error);
        });
        fileStream.on('finish', function() {
          resolve();
        });
      });
      
      console.log(`Successfully downloaded: ${filePath}`);
      return true;
    } catch (error) {
      console.error(`Error downloading ${url}:`, error);
      
      // Remove partial file if it exists
      if (fs.existsSync(filePath)) {
        try {
          fs.unlinkSync(filePath);
          console.log(`Removed incomplete download: ${filePath}`);
        } catch (unlinkError) {
          console.error(`Failed to remove incomplete file: ${unlinkError.message}`);
        }
      }
      
      throw error;
    }
  }
  
  /**
   * Try to download a file with retries
   * @param {string} url URL to download from
   * @param {string} filePath Path to save the file
   * @param {number} retries Number of retries
   */
  async downloadWithRetry(url, filePath, retries = 3) {
    for (let attempt = 1; attempt <= retries; attempt++) {
      console.log(`Download attempt ${attempt}/${retries}`);
      try {
        return await this.downloadFile(url, filePath);
      } catch (error) {
        console.error(`Attempt ${attempt} failed:`, error);
        
        if (attempt < retries) {
          const delay = attempt * 3000; // Exponential backoff
          console.log(`Retry ${attempt}/${retries} in ${delay/1000} seconds...`);
          await new Promise(resolve => setTimeout(resolve, delay));
        } else {
          throw error;
        }
      }
    }
  }
  
  /**
   * Download a model
   * @param {string} modelId The model ID
   */
  async downloadModel(modelId) {
    if (!TEST_MODELS[modelId]) {
      throw new Error(`Model ${modelId} not found in TEST_MODELS`);
    }
    
    if (this.isModelDownloaded(modelId)) {
      console.log(`Model ${modelId} is already downloaded`);
      return true;
    }
    
    // Skip downloading for mock models
    if (modelId === 'mock-test-model') {
      console.log(`Mock model ${modelId} is locally generated, no download needed`);
      return true;
    }
    
    const modelInfo = TEST_MODELS[modelId];
    const modelPath = path.join(this.modelDir, modelId);
    const wasmFile = path.join(modelPath, `${modelId}.wasm`);
    const configFile = path.join(modelPath, `${modelId}.json`);
    
    // Create model directory
    try {
      if (!fs.existsSync(modelPath)) {
        fs.mkdirSync(modelPath, { recursive: true });
        console.log(`Created directory for model ${modelId}: ${modelPath}`);
      }
    } catch (error) {
      console.error(`Error creating model directory: ${error.message}`);
      throw error;
    }
    
    console.log(`Downloading model ${modelId} (${modelInfo.size})...`);
    
    try {
      // Download model WASM
      console.log(`Downloading ${modelId} WASM file...`);
      await this.downloadWithRetry(modelInfo.url, wasmFile);
      
      // Download model config
      console.log(`Downloading ${modelId} config file...`);
      await this.downloadWithRetry(modelInfo.configUrl, configFile);
      
      if (fs.existsSync(wasmFile) && fs.existsSync(configFile)) {
        const wasmSize = (fs.statSync(wasmFile).size / (1024 * 1024)).toFixed(2);
        const configSize = (fs.statSync(configFile).size / 1024).toFixed(2);
        console.log(`Model ${modelId} downloaded successfully`);
        console.log(`- WASM file: ${wasmSize} MB`);
        console.log(`- Config file: ${configSize} KB`);
        return true;
      } else {
        throw new Error(`Failed to download model ${modelId} - files not found after download`);
      }
    } catch (error) {
      console.error(`Error downloading model ${modelId}:`, error);
      return false;
    }
  }
  
  /**
   * Download the default model
   */
  async downloadDefaultModel() {
    return this.downloadModel(this.defaultModel);
  }
  
  /**
   * Get model directory path
   * @param {string} modelId The model ID
   */
  getModelPath(modelId) {
    return path.join(this.modelDir, modelId);
  }
}

/**
 * Create and initialize a model manager
 */
export async function createModelManager(options = {}) {
  const manager = new ModelManager(options);
  await manager.init();
  return manager;
}

// If this script is executed directly, download the default model
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const modelId = process.argv[2] || 'tinyllama-1.1b-chat-v1.0';
  console.log(`Starting download for model ${modelId}...`);
  
  createModelManager()
    .then(manager => manager.downloadModel(modelId))
    .then(success => {
      if (success) {
        console.log(`Model ${modelId} is ready for testing`);
        process.exit(0);
      } else {
        console.error(`Failed to download model ${modelId}`);
        process.exit(1);
      }
    })
    .catch(error => {
      console.error('Error:', error);
      process.exit(1);
    });
}