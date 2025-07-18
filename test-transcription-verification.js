#!/usr/bin/env node

/**
 * Transcription Verification Test
 * 
 * This script demonstrates that the transcription system is working by:
 * 1. Testing the WebSocket connection to the orchestrator
 * 2. Sending test audio data
 * 3. Capturing and displaying actual transcriptions
 * 4. Providing verification logs
 */

const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

// Configuration
const ORCHESTRATOR_URL = 'ws://localhost:8000/ws/voice';
const TEST_AUDIO_FILE = './audio_en.wav';
const SESSION_ID = `test_${Date.now()}`;

// Enhanced logging
const log = {
    info: (msg, data = '') => console.log(`\x1b[36m[INFO]\x1b[0m ${msg}`, data),
    success: (msg, data = '') => console.log(`\x1b[32m[SUCCESS]\x1b[0m ${msg}`, data),
    error: (msg, data = '') => console.log(`\x1b[31m[ERROR]\x1b[0m ${msg}`, data),
    debug: (msg, data = '') => console.log(`\x1b[33m[DEBUG]\x1b[0m ${msg}`, data),
    transcript: (msg) => console.log(`\x1b[35m[TRANSCRIPT]\x1b[0m ${msg}`)
};

class TranscriptionTester {
    constructor() {
        this.ws = null;
        this.transcripts = [];
        this.startTime = Date.now();
    }

    async run() {
        log.info('🎯 Starting Transcription Verification Test');
        log.info(`Session ID: ${SESSION_ID}`);
        log.info(`Orchestrator URL: ${ORCHESTRATOR_URL}`);

        try {
            await this.testConnection();
            await this.sendTestAudio();
            await this.waitForTranscripts();
            await this.displayResults();
        } catch (error) {
            log.error('Test failed:', error.message);
            process.exit(1);
        }
    }

    async testConnection() {
        return new Promise((resolve, reject) => {
            log.info('🔌 Testing WebSocket connection...');
            
            this.ws = new WebSocket(ORCHESTRATOR_URL);

            this.ws.on('open', () => {
                log.success('✅ WebSocket connected successfully');
                resolve();
            });

            this.ws.on('error', (error) => {
                log.error('❌ WebSocket connection failed:', error.message);
                reject(error);
            });

            this.ws.on('message', (data) => {
                try {
                    const message = JSON.parse(data.toString());
                    this.handleMessage(message);
                } catch (error) {
                    log.debug('Raw message received:', data.toString());
                }
            });

            this.ws.on('close', (code, reason) => {
                log.info(`WebSocket closed: ${code} ${reason}`);
            });
        });
    }

    handleMessage(message) {
        log.debug('Received message:', JSON.stringify(message, null, 2));

        if (message.type === 'transcript') {
            const transcript = {
                text: message.text,
                timestamp: new Date().toISOString(),
                latency: Date.now() - this.startTime
            };
            
            this.transcripts.push(transcript);
            log.transcript(`"${message.text}" (latency: ${transcript.latency}ms)`);
        }

        if (message.type === 'status') {
            log.info(`Status: ${message.status}`);
        }

        if (message.type === 'error') {
            log.error(`Error: ${message.error}`);
        }
    }

    async sendTestAudio() {
        log.info('🎤 Sending test audio data...');

        // Check if test audio file exists
        if (!fs.existsSync(TEST_AUDIO_FILE)) {
            log.info('Test audio file not found, using synthetic audio data...');
            await this.sendSyntheticAudio();
            return;
        }

        // Read and send actual audio file
        const audioBuffer = fs.readFileSync(TEST_AUDIO_FILE);
        log.info(`Sending ${audioBuffer.length} bytes of audio data`);

        // Send audio in chunks to simulate real-time streaming
        const chunkSize = 5460; // 16kHz * 0.34125s * 4 bytes
        for (let i = 0; i < audioBuffer.length; i += chunkSize) {
            const chunk = audioBuffer.slice(i, i + chunkSize);
            this.ws.send(chunk);
            await this.sleep(100); // Simulate real-time streaming
        }

        log.success('✅ Test audio data sent successfully');
    }

    async sendSyntheticAudio() {
        log.info('🎵 Generating synthetic audio data...');
        
        // Generate 3 seconds of synthetic audio (16kHz, Float32Array)
        const sampleRate = 16000;
        const duration = 3;
        const samples = sampleRate * duration;
        const audioData = new Float32Array(samples);

        // Generate a simple sine wave
        const frequency = 440; // A4 note
        for (let i = 0; i < samples; i++) {
            audioData[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * 0.3;
        }

        // Convert to bytes
        const buffer = Buffer.from(audioData.buffer);
        log.info(`Sending ${buffer.length} bytes of synthetic audio`);

        // Send in chunks
        const chunkSize = 5460;
        for (let i = 0; i < buffer.length; i += chunkSize) {
            const chunk = buffer.slice(i, i + chunkSize);
            this.ws.send(chunk);
            await this.sleep(100);
        }
    }

    async waitForTranscripts() {
        log.info('⏳ Waiting for transcriptions...');
        
        return new Promise((resolve) => {
            const timeout = setTimeout(() => {
                log.info('⏰ Timeout reached, checking results...');
                resolve();
            }, 10000);

            // Check every 500ms for new transcripts
            const checkInterval = setInterval(() => {
                if (this.transcripts.length > 0) {
                    clearTimeout(timeout);
                    clearInterval(checkInterval);
                    log.success(`🎉 Received ${this.transcripts.length} transcriptions`);
                    resolve();
                }
            }, 500);
        });
    }

    async displayResults() {
        console.log('\n' + '='.repeat(60));
        console.log('📊 TRANSCRIPTION VERIFICATION RESULTS');
        console.log('='.repeat(60));

        if (this.transcripts.length === 0) {
            log.error('❌ No transcriptions received');
            console.log('\nPossible issues:');
            console.log('1. WhisperLive service not running');
            console.log('2. Audio format incompatible');
            console.log('3. Network connectivity issues');
            console.log('4. Service configuration problems');
            return;
        }

        log.success(`✅ Successfully received ${this.transcripts.length} transcriptions`);
        
        this.transcripts.forEach((transcript, index) => {
            console.log(`\n${index + 1}. "${transcript.text}"`);
            console.log(`   Latency: ${transcript.latency}ms`);
            console.log(`   Timestamp: ${transcript.timestamp}`);
        });

        console.log('\n✅ TRANSCRIPTION SYSTEM VERIFICATION COMPLETE');
        console.log('The system is successfully processing audio and returning transcriptions!');
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Run the test
async function main() {
    const tester = new TranscriptionTester();
    await tester.run();
}

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\n🛑 Test interrupted by user');
    process.exit(0);
});

if (require.main === module) {
    main().catch(console.error);
}

module.exports = TranscriptionTester;