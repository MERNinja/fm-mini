#!/usr/bin/env node
/**
 * Run script for tensor parallelism tests
 * This script starts the server and runs the tests
 * 
 * Usage:
 *   node run-tests.js           - Run the full test with mock WebLLM
 *   node run-tests.js --simple  - Run a simplified test that doesn't use WebLLM
 */
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import net from 'net';

// Get __dirname in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '../..');

// Parse command line arguments
const args = process.argv.slice(2);
const SIMPLE_MODE = args.includes('--simple');

// Constants
const BASE_PORT = 8099;
const MAX_PORT_ATTEMPTS = 10; // Try up to 10 different ports
const TEST_TIMEOUT = 300000; // 5 minutes

// Find first available port
const findAvailablePort = async (basePort) => {
  for (let attempt = 0; attempt < MAX_PORT_ATTEMPTS; attempt++) {
    const port = basePort + attempt;
    const isPortAvailable = await checkPort(port);
    if (isPortAvailable) {
      return port;
    }
    console.log(`Port ${port} is in use, trying next port...`);
  }
  throw new Error(`Could not find an available port after ${MAX_PORT_ATTEMPTS} attempts`);
};

// Check if a port is available
const checkPort = (port) => {
  return new Promise((resolve) => {
    const tester = net.createServer()
      .once('error', () => {
        // Port is in use
        resolve(false);
      })
      .once('listening', () => {
        // Port is available
        tester.close();
        resolve(true);
      })
      .listen(port);
  });
};

// Start server
const startServer = async () => {
  console.log('Finding available port...');
  const port = await findAvailablePort(BASE_PORT);
  console.log(`Starting server on port ${port}...`);
  
  // Update port in test files
  updatePortInTestFiles(port);
  
  return new Promise((resolve) => {
    const server = spawn('node', ['server.js'], {
      cwd: ROOT_DIR,
      stdio: 'pipe',
      env: {
        ...process.env,
        PORT: port.toString(),
        NODE_ENV: 'test'
      }
    });
    
    let started = false;
    
    server.stdout.on('data', (data) => {
      const output = data.toString();
      console.log(`[SERVER]: ${output.trim()}`);
      
      // Check if server is ready
      if (output.includes('Server running') && !started) {
        started = true;
        console.log(`Server started successfully on port ${port}`);
        
        // Wait a moment for server initialization
        setTimeout(() => resolve({ server, port }), 1000);
      }
    });
    
    server.stderr.on('data', (data) => {
      console.error(`[SERVER ERROR]: ${data.toString().trim()}`);
    });
    
    server.on('error', (error) => {
      console.error('Failed to start server:', error);
      process.exit(1);
    });
    
    // Set a timeout in case server doesn't start
    setTimeout(() => {
      if (!started) {
        console.error('Server failed to start within timeout');
        server.kill();
        process.exit(1);
      }
    }, 10000);
  });
};

// Update port in test files
const updatePortInTestFiles = (port) => {
  try {
    // Update test files to use the correct port
    const testFiles = [
      'test-tensor-parallel.js',
      'tensor-parallel.test.js',
      'simple-test.js'
    ];
    
    for (const filename of testFiles) {
      const filePath = path.join(__dirname, filename);
      if (fs.existsSync(filePath)) {
        let content = fs.readFileSync(filePath, 'utf8');
        content = content.replace(/serverUrl: 'http:\/\/localhost:\d+'/g, `serverUrl: 'http://localhost:${port}'`);
        content = content.replace(/const SERVER_URL = 'http:\/\/localhost:\d+'/g, `const SERVER_URL = 'http://localhost:${port}'`);
        fs.writeFileSync(filePath, content);
      }
    }
    
    console.log(`Updated test files to use port ${port}`);
  } catch (error) {
    console.error('Error updating port in test files:', error);
  }
};

// Run the full tensor parallel test
const runTensorTest = async () => {
  console.log('Running tensor parallel test...');
  
  return new Promise((resolve, reject) => {
    const test = spawn('node', [path.join(__dirname, 'test-tensor-parallel.js')], {
      cwd: ROOT_DIR,
      stdio: 'inherit'
    });
    
    test.on('close', (code) => {
      if (code === 0) {
        console.log('Tensor test completed successfully');
        resolve();
      } else {
        console.error(`Tensor test failed with code ${code}`);
        reject(new Error(`Tensor test failed with code ${code}`));
      }
    });
    
    test.on('error', (error) => {
      console.error('Failed to run tensor test:', error);
      reject(error);
    });
  });
};

// Run the simple test that doesn't use WebLLM
const runSimpleTest = async () => {
  console.log('Running simple connection test (no WebLLM)...');
  
  return new Promise((resolve, reject) => {
    const test = spawn('node', [path.join(__dirname, 'simple-test.js')], {
      cwd: ROOT_DIR,
      stdio: 'inherit'
    });
    
    test.on('close', (code) => {
      if (code === 0) {
        console.log('Simple test completed successfully');
        resolve();
      } else {
        console.error(`Simple test failed with code ${code}`);
        reject(new Error(`Simple test failed with code ${code}`));
      }
    });
    
    test.on('error', (error) => {
      console.error('Failed to run simple test:', error);
      reject(error);
    });
  });
};

// Main function
const main = async () => {
  let serverInfo = null;
  
  try {
    console.log('=== Starting Tensor Parallelism Tests ===');
    console.log(SIMPLE_MODE ? 'Mode: Simple (no WebLLM)' : 'Mode: Full (with mock WebLLM)');
    
    if (!SIMPLE_MODE) {
      // First, ensure we have a mock model for testing
      const mockModelDir = path.join(ROOT_DIR, 'models', 'mock-test-model');
      const mockModelFile = path.join(mockModelDir, 'mock-test-model.wasm');
      
      if (!fs.existsSync(mockModelFile)) {
        console.log('Mock model not found, creating it first...');
        
        // Run the create-mock-model script
        const createModel = spawn('node', [path.join(__dirname, 'create-mock-model.js')], {
          cwd: ROOT_DIR,
          stdio: 'inherit'
        });
        
        await new Promise((resolve, reject) => {
          createModel.on('close', (code) => {
            if (code === 0) {
              console.log('Mock model created successfully');
              resolve();
            } else {
              console.error(`Failed to create mock model with code ${code}`);
              reject(new Error('Failed to create mock model'));
            }
          });
        });
      }
    }
    
    // Start server
    serverInfo = await startServer();
    
    // Run appropriate test
    if (SIMPLE_MODE) {
      await runSimpleTest();
    } else {
      await runTensorTest();
    }
    
    console.log('\n=== All tests completed successfully ===');
    
  } catch (error) {
    console.error('Error running tests:', error);
    process.exit(1);
  } finally {
    // Cleanup
    if (serverInfo && serverInfo.server) {
      console.log('Stopping server...');
      serverInfo.server.kill();
      console.log('Server stopped');
    }
  }
};

// Run the script
main(); 