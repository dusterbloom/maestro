#!/usr/bin/env node
/**
 * Quick frontend environment variable check
 */

console.log('🔍 Frontend Environment Debug');
console.log('============================');

// Check if running in Node.js environment
if (typeof process !== 'undefined' && process.env) {
    console.log('📊 Environment Variables:');
    console.log('NEXT_PUBLIC_ORCHESTRATOR_WS_URL:', process.env.NEXT_PUBLIC_ORCHESTRATOR_WS_URL);
    
    // List all NEXT_PUBLIC_ variables
    const nextPublicVars = Object.keys(process.env)
        .filter(key => key.startsWith('NEXT_PUBLIC_'))
        .map(key => `${key}: ${process.env[key]}`);
    
    console.log('\n📋 All NEXT_PUBLIC_ variables:');
    if (nextPublicVars.length === 0) {
        console.log('❌ No NEXT_PUBLIC_ variables found!');
        console.log('💡 Make sure .env.local exists in ui/ directory');
    } else {
        nextPublicVars.forEach(v => console.log('  ', v));
    }
} else {
    console.log('❌ This script should be run with Node.js');
    console.log('💡 Run: cd ui && node ../frontend-debug.js');
}

// Quick WebSocket test from Node.js
console.log('\n🔌 Testing WebSocket from Node.js...');
try {
    const WebSocket = require('ws');
    const ws = new WebSocket('ws://localhost:8000/ws/debug_test');
    
    ws.on('open', () => {
        console.log('✅ WebSocket connection successful from Node.js!');
        ws.close();
    });
    
    ws.on('error', (error) => {
        console.log('❌ WebSocket connection failed:', error.message);
    });
    
    ws.on('message', (data) => {
        console.log('📨 Received:', data.toString());
    });
    
} catch (error) {
    console.log('❌ WebSocket test failed:', error.message);
    console.log('💡 Run: cd ui && npm install ws');
}
