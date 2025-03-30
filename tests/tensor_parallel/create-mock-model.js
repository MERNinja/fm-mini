#!/usr/bin/env node
/**
 * Create Mock Model for Testing
 * 
 * This script creates small mock model files for testing tensor parallelism
 * without needing to download real models.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get __dirname in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Create mock model directory
const MODEL_ID = 'mock-test-model';
const MODEL_DIR = path.resolve(__dirname, '../../models', MODEL_ID);

// Create directory if it doesn't exist
if (!fs.existsSync(MODEL_DIR)) {
  fs.mkdirSync(MODEL_DIR, { recursive: true });
  console.log(`Created directory: ${MODEL_DIR}`);
}

// Mock model config with minimal structure needed for testing
const mockModelConfig = {
  "model_info": {
    "model_id": "mock-test-model",
    "model_name": "Mock Test Model",
    "max_sequence_length": 2048,
    "vocab_size": 32000,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12
  },
  "tokenizer_files": [
    "tokenizer.json"
  ],
  "context_window_size": 2048,
  "sliding_window_size": 0,
  "prepended_space_in_first_token": false,
  "add_bos_token": true,
  "add_eos_token": true,
  "chat_template": "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[USER]: ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '[ASSISTANT]: ' + message['content'] }}{% elif message['role'] == 'system' %}{{ '[SYSTEM]: ' + message['content'] }}{% endif %}{% endfor %}",
  "generate_chat_response_prompt": "{{ '[ASSISTANT]: ' }}"
};

// Create a buffer of the desired size (1MB is enough for testing)
function createMockModelFile(size = 1024 * 1024) {
  const buffer = Buffer.alloc(size);
  // Fill with random data
  for (let i = 0; i < size; i += 4) {
    buffer.writeUInt32LE(Math.floor(Math.random() * 0xFFFFFFFF), i);
  }
  return buffer;
}

// Write files
function writeMockFiles() {
  try {
    // Write mock WASM file (1MB)
    const wasmPath = path.join(MODEL_DIR, `${MODEL_ID}.wasm`);
    fs.writeFileSync(wasmPath, createMockModelFile());
    
    // Write mock JSON config
    const configPath = path.join(MODEL_DIR, `${MODEL_ID}.json`);
    fs.writeFileSync(configPath, JSON.stringify(mockModelConfig, null, 2));
    
    // Write mock tokenizer file
    const tokenizerPath = path.join(MODEL_DIR, 'tokenizer.json');
    fs.writeFileSync(tokenizerPath, JSON.stringify({
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": null,
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "BPE",
        "vocab": { "0": "a", "1": "b" },
        "merges": ["a b"]
      }
    }, null, 2));
    
    // Get file sizes
    const wasmSize = (fs.statSync(wasmPath).size / (1024 * 1024)).toFixed(2);
    const configSize = (fs.statSync(configPath).size / 1024).toFixed(2);
    const tokenizerSize = (fs.statSync(tokenizerPath).size / 1024).toFixed(2);
    
    console.log('Mock model files created successfully:');
    console.log(`- WASM file: ${wasmSize} MB`);
    console.log(`- Config file: ${configSize} KB`);
    console.log(`- Tokenizer file: ${tokenizerSize} KB`);
    console.log(`\nModel location: ${MODEL_DIR}`);
    console.log('\nTo use this model for testing:');
    console.log('1. Update your test scripts to use MODEL_ID = "mock-test-model"');
    console.log('2. Run tests with: npm run test:tensor');
    
    return true;
  } catch (error) {
    console.error('Error creating mock files:', error);
    return false;
  }
}

// Run the script
writeMockFiles();
console.log('\nNote: This mock model is for TESTING ONLY and will not produce real results.');
console.log('It should only be used to test the tensor parallelism infrastructure without needing to download real models.'); 