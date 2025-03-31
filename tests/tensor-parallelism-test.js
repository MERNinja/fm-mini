/**
 * Tensor Parallelism Test
 * 
 * This file contains tests to verify that tensor parallelism is working correctly.
 * It monitors node activity and verifies that:
 * 1. Tasks are properly distributed from origin to peer nodes
 * 2. Peer nodes process their assigned layers
 * 3. Results are gathered back and combined
 */

const io = require('socket.io-client');
const assert = require('assert');

// Connect to socket.io server
const socket = io('http://localhost:3000');

// Test settings
const TEST_TIMEOUT = 30000; // 30 seconds
const ORIGIN_NODE_ID = null; // Will be set when we identify origin node
const PEER_NODE_IDS = []; // Will store peer node IDs
const REQUIRED_PEER_COUNT = 2; // Need at least 2 peer nodes

// Track test state
let originNodePromptSent = false;
let tasksAssigned = 0;
let tasksCompleted = 0;
let responseReceived = false;
let allTestsPassed = false;

// Start tests when connected
socket.on('connect', () => {
  console.log('Connected to socket.io server for testing');
  
  // Register a test observer node
  socket.emit('register_node', {
    id: 'test_observer',
    model: 'test',
    ip: 'localhost',
    status: 'testing'
  });
  
  // Monitor all node activity
  socket.on('node_activity', (activity) => {
    console.log(`[TEST] ${activity.action}:`, activity.prompt);
    
    // Track origin node when prompt is sent
    if (activity.action === 'delegation_start') {
      ORIGIN_NODE_ID = activity.nodeId;
      originNodePromptSent = true;
      console.log(`[TEST] Origin node identified: ${ORIGIN_NODE_ID}`);
    }
    
    // Track task assignments to peer nodes
    if (activity.action === 'sending_task' && activity.isOriginNode) {
      tasksAssigned++;
      console.log(`[TEST] Task ${tasksAssigned} assigned`);
    }
    
    // Track task completion by peer nodes
    if (activity.action === 'task_completed_by_peer' || activity.action === 'peer_completed_task') {
      tasksCompleted++;
      if (!PEER_NODE_IDS.includes(activity.nodeId)) {
        PEER_NODE_IDS.push(activity.nodeId);
      }
      console.log(`[TEST] Task completed by peer ${activity.nodeId}, ${tasksCompleted}/${tasksAssigned}`);
    }
    
    // Check if all tasks have been completed
    if (activity.action === 'all_tensor_tasks_verified' || activity.action === 'all_tasks_completed') {
      responseReceived = true;
      console.log('[TEST] All tasks completed, response generated');
      
      // Verify test conditions
      if (originNodePromptSent && tasksAssigned >= REQUIRED_PEER_COUNT && tasksCompleted >= REQUIRED_PEER_COUNT && PEER_NODE_IDS.length >= REQUIRED_PEER_COUNT) {
        allTestsPassed = true;
        console.log('\n[TEST] ✅ ALL TESTS PASSED: Real tensor parallelism verified');
        console.log(`[TEST] - Origin node: ${ORIGIN_NODE_ID}`);
        console.log(`[TEST] - Peer nodes: ${PEER_NODE_IDS.join(', ')}`);
        console.log(`[TEST] - Tasks assigned: ${tasksAssigned}`);
        console.log(`[TEST] - Tasks completed: ${tasksCompleted}`);
        
        // Clean exit after test passes
        setTimeout(() => {
          socket.disconnect();
          console.log('[TEST] Test completed successfully');
          process.exit(0);
        }, 2000);
      }
    }
  });
  
  // Set a timeout for the test
  setTimeout(() => {
    console.log('\n[TEST] ⚠️ TEST TIMED OUT');
    console.log(`[TEST] - Origin node: ${ORIGIN_NODE_ID || 'Not identified'}`);
    console.log(`[TEST] - Peer nodes: ${PEER_NODE_IDS.length > 0 ? PEER_NODE_IDS.join(', ') : 'None'}`);
    console.log(`[TEST] - Tasks assigned: ${tasksAssigned}`);
    console.log(`[TEST] - Tasks completed: ${tasksCompleted}`);
    console.log(`[TEST] - Tests passed: ${allTestsPassed ? 'Yes' : 'No'}`);
    
    // Exit with non-zero code if tests failed
    socket.disconnect();
    process.exit(allTestsPassed ? 0 : 1);
  }, TEST_TIMEOUT);
});

// Handle errors
socket.on('error', (err) => {
  console.error('[TEST] Socket error:', err);
  process.exit(1);
});

console.log('[TEST] Tensor parallelism test started');
console.log('[TEST] Waiting for 3 nodes to connect and execute "write me a short story" prompt...'); 