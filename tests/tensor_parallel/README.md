# Tensor Parallelism Tests

This directory contains tests to verify the tensor parallelism functionality in the WebLLM implementation.

## Overview

The tensor parallelism tests deploy multiple headless clients that act as tensor nodes, each loading a small LLM locally. The tests verify that:
1. Clients can successfully connect to each other
2. Tensor parallelism can be enabled between nodes
3. Inference is distributed across multiple nodes
4. Different parallelism strategies can be used

## Test Setup

The test environment consists of:
- A WebSocket signaling server (`server.js` in the root directory)
- Three headless client nodes, each loading the same LLM locally
- A test runner that verifies the behavior of the system

## Test Models

To avoid downloading large models every time, the tests can use one of the following options:

### Mock Model (Fastest, Recommended)

A locally created mock model (about 1MB) designed for quick testing:

```
npm run create:mock-model
```

This is the fastest option and doesn't require any downloads. The mock model is generated locally with minimal random data sufficient for testing the tensor parallel infrastructure.

### Tiny Test Model

A very small version of TinyLlama (around 270MB):

```
npm run download:tiny-model
```

This model is suitable when you want actual language model functionality in a smaller package.

### Standard Test Model

The standard test model (`tinyllama-1.1b-chat-v1.0`) is around 560MB in size:

```
npm run download:models
```

All models will be stored locally in the `models` directory. The tests automatically detect which model is available and use it, with a preference order of: mock model > tiny model > standard model.

## Running the Tests

### Prerequisites

Make sure you have the required dependencies:

```
npm install
```

### Create or Download a Test Model

Before running tests, create or download at least one of the test models:

```
npm run create:mock-model     # For a local mock model (fastest, recommended)
```

or

```
npm run download:tiny-model   # For a small test model
```

or

```
npm run download:models       # For the full TinyLlama model
```

### Running the Test Suite

To run the full test suite:

```
npm run test:tensor
```

This will:
1. Start the server
2. Deploy three headless clients
3. Run the tensor parallelism tests
4. Clean up and stop the server

### Individual Tests

You can also run the individual test scripts:

```
node tests/tensor_parallel/test-tensor-parallel.js
```

Or using Jest:

```
npm test -- tests/tensor_parallel/tensor-parallel.test.js
```

## Monitoring Test Results

The tests output detailed logs about the tensor parallelism process, including:
- Node registration
- Model loading progress
- Tensor partitioning information
- Performance metrics
- Inference results

Test results are also saved to JSON files in the `tests/tensor_parallel` directory:
- `test-results.json`: Contains successful test results
- `test-results-with-errors.json`: Contains information about any errors encountered

## Troubleshooting

If you encounter issues:

1. Check that the server is running on port 8080
2. Verify that the model was created/downloaded correctly (check the `models` directory)
3. Ensure there are no firewall or network restrictions blocking WebSocket connections
4. Check the console logs for detailed error messages

### Download Issues

If you're experiencing problems downloading the real models:

1. Use the mock model option with `npm run create:mock-model` (doesn't require downloads)
2. Check your network connection and ensure you can access Hugging Face
3. If you have proxy/firewall issues, consider downloading the model files manually and placing them in the correct directory:
   - For mock-test-model: `models/mock-test-model/mock-test-model.wasm` and `models/mock-test-model/mock-test-model.json`
   - For tiny-test-model: `models/tiny-test-model/tiny-test-model.wasm` and `models/tiny-test-model/tiny-test-model.json`
   - For tinyllama: `models/tinyllama-1.1b-chat-v1.0/tinyllama-1.1b-chat-v1.0.wasm` and `models/tinyllama-1.1b-chat-v1.0/tinyllama-1.1b-chat-v1.0.json`

### 401 Unauthorized Errors

If you get 401 errors when downloading from Hugging Face:
1. Use the mock model option instead (`npm run create:mock-model`)
2. The real models may require authentication or have access restrictions

## Extending the Tests

To add new tests:
1. Add test cases to `tensor-parallel.test.js`
2. Add new prompts to the `TEST_PROMPTS` array
3. Create new test functions in the Jest test suite

To test different parallelism strategies:
1. Modify the strategy type in the tests (e.g., `StrategyType.LAYER_PARALLEL`)
2. Compare performance metrics between different strategies 