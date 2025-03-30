#!/usr/bin/env node
/**
 * Tiny Model Downloader for testing
 * Downloads a very small test model for debugging tensor parallelism
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';

// Get __dirname in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Small model for debugging - Using WebLLM-Redpajama (smaller and public)
const TINY_TEST_MODEL = {
  id: 'tiny-test-model',
  name: 'RedPajama Debug Model',
  wasmUrl: 'https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_1/resolve/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1.wasm',
  configUrl: 'https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_1/resolve/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1.json'
};

// Create model directory
const MODEL_DIR = path.resolve(__dirname, '../../models/tiny-test-model');
if (!fs.existsSync(MODEL_DIR)) {
  fs.mkdirSync(MODEL_DIR, { recursive: true });
  console.log(`Created directory: ${MODEL_DIR}`);
}

/**
 * Download a file with a User-Agent header
 */
async function downloadFile(url, filePath) {
  console.log(`Downloading from ${url}...`);
  try {
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}, Message: ${response.statusText}`);
    }
    
    const fileStream = fs.createWriteStream(filePath);
    
    await new Promise((resolve, reject) => {
      response.body.pipe(fileStream);
      response.body.on('error', (err) => {
        console.error('Stream error:', err);
        reject(err);
      });
      fileStream.on('error', (err) => {
        console.error('File write error:', err);
        reject(err);
      });
      fileStream.on('finish', () => resolve());
    });
    
    console.log(`Downloaded to ${filePath}`);
    return true;
  } catch (error) {
    console.error(`Error downloading file: ${error.message}`);
    if (error.cause) {
      console.error('Cause:', error.cause);
    }
    return false;
  }
}

/**
 * Try to download with retries
 */
async function downloadWithRetry(url, filePath, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    console.log(`Download attempt ${attempt}/${maxRetries}`);
    try {
      const success = await downloadFile(url, filePath);
      if (success) return true;
    } catch (error) {
      console.error(`Attempt ${attempt} failed:`, error);
    }
    
    if (attempt < maxRetries) {
      const delay = attempt * 2000;
      console.log(`Retrying in ${delay/1000} seconds...`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  return false;
}

/**
 * Main function
 */
async function main() {
  try {
    console.log('Downloading tiny test model for debugging...');
    
    // Download WASM file
    const wasmPath = path.join(MODEL_DIR, 'tiny-test-model.wasm');
    const wasmSuccess = await downloadWithRetry(TINY_TEST_MODEL.wasmUrl, wasmPath);
    
    // Download config file
    const configPath = path.join(MODEL_DIR, 'tiny-test-model.json');
    const configSuccess = await downloadWithRetry(TINY_TEST_MODEL.configUrl, configPath);
    
    if (wasmSuccess && configSuccess) {
      const wasmSize = (fs.statSync(wasmPath).size / (1024 * 1024)).toFixed(2);
      const configSize = (fs.statSync(configPath).size / 1024).toFixed(2);
      
      console.log('\nDownload completed successfully!');
      console.log(`WASM file size: ${wasmSize} MB`);
      console.log(`Config file size: ${configSize} KB`);
      console.log(`\nFiles saved to: ${MODEL_DIR}`);
      console.log('\nTo use this model, update the MODEL_ID in your test scripts to "tiny-test-model"');
      
      return true;
    } else {
      console.error('Failed to download one or more files');
      return false;
    }
    
  } catch (error) {
    console.error('Error:', error);
    return false;
  }
}

// Run the script
main()
  .then(success => {
    process.exit(success ? 0 : 1);
  })
  .catch(error => {
    console.error('Unhandled error:', error);
    process.exit(1);
  }); 