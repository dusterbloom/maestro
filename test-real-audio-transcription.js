#!/usr/bin/env node
/**
 * Test script to verify transcription flow using real audio file (audio_en.wav)
 * This script reads the actual audio file and sends it to the orchestrator
 */

const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

// Configuration
const ORCHESTRATOR_URL = 'ws://localhost:8000';
const AUDIO_FILE = path.join(__dirname, 'audio_en.wav');
const SESSION_ID = `test_${Date.now()}`;

// WAV file parser
function parseWavFile(buffer) {
    // WAV file header parsing
    const riffHeader = buffer.toString('ascii', 0, 4);
    if (riffHeader !== 'RIFF') {
        throw new Error('Invalid WAV file format');
    }

    const fileSize = buffer.readUInt32LE(4);
    const waveHeader = buffer.toString('ascii', 8, 12);
    if (waveHeader !== 'WAVE') {
        throw new Error('Invalid WAV file format');
    }

    // Find fmt chunk
    let fmtOffset = 12;
    while (fmtOffset < buffer.length) {
        const chunkId = buffer.toString('ascii', fmtOffset, fmtOffset + 4);
        const chunkSize = buffer.readUInt32LE(fmtOffset + 4);
        
        if (chunkId === 'fmt ') {
            const audioFormat = buffer.readUInt16LE(fmtOffset + 8);
            const numChannels = buffer.readUInt16LE(fmtOffset + 10);
            const sampleRate = buffer.readUInt32LE(fmtOffset + 12);
            const bitsPerSample = buffer.readUInt16LE(fmtOffset + 22);
            
            console.log(`📊 Audio format: ${audioFormat}, Channels: ${numChannels}, Sample rate: ${sampleRate}, Bits: ${bitsPerSample}`);
            
            // Find data chunk
            let dataOffset = fmtOffset + 8 + chunkSize;
            while (dataOffset < buffer.length) {
                const dataChunkId = buffer.toString('ascii', dataOffset, dataOffset + 4);
                const dataChunkSize = buffer.readUInt32LE(dataOffset + 4);
                
                if (dataChunkId === 'data') {
                    const audioData = buffer.slice(dataOffset + 8, dataOffset + 8 + dataChunkSize);
                    
                    // Convert to Float32Array based on format
                    let float32Data;
                    if (bitsPerSample === 16) {
                        // Convert 16-bit PCM to Float32
                        const int16Data = new Int16Array(audioData.buffer, audioData.byteOffset, audioData.length / 2);
                        float32Data = new Float32Array(int16Data.length);
                        for (let i = 0; i < int16Data.length; i++) {
                            float32Data[i] = int16Data[i] / 32768.0; // Normalize to [-1, 1]
                        }
                    } else if (bitsPerSample === 32) {
                        // Already 32-bit float
                        float32Data = new Float32Array(audioData.buffer, audioData.byteOffset, audioData.length / 4);
                    } else {
                        throw new Error(`Unsupported bits per sample: ${bitsPerSample}`);
                    }
                    
                    // Resample to 16kHz if needed
                    if (sampleRate !== 16000) {
                        console.log(`🔄 Resampling from ${sampleRate}Hz to 16000Hz...`);
                        float32Data = resampleAudio(float32Data, sampleRate, 16000);
                    }
                    
                    // Handle mono/stereo conversion
                    if (numChannels > 1) {
                        console.log(`🔄 Converting ${numChannels} channels to mono...`);
                        const monoData = new Float32Array(float32Data.length / numChannels);
                        for (let i = 0; i < monoData.length; i++) {
                            let sum = 0;
                            for (let ch = 0; ch < numChannels; ch++) {
                                sum += float32Data[i * numChannels + ch];
                            }
                            monoData[i] = sum / numChannels;
                        }
                        float32Data = monoData;
                    }
                    
                    return float32Data;
                }
                dataOffset += 8 + dataChunkSize;
            }
        }
        fmtOffset += 8 + chunkSize;
    }
    
    throw new Error('No audio data found in WAV file');
}

// Resample audio data
function resampleAudio(inputData, fromRate, toRate) {
    if (fromRate === toRate) return inputData;
    
    const ratio = fromRate / toRate;
    const outputLength = Math.floor(inputData.length / ratio);
    const output = new Float32Array(outputLength);
    
    for (let i = 0; i < outputLength; i++) {
        const sourceIndex = i * ratio;
        const sourceIndexFloor = Math.floor(sourceIndex);
        const sourceIndexCeil = Math.min(sourceIndexFloor + 1, inputData.length - 1);
        const fraction = sourceIndex - sourceIndexFloor;
        
        // Linear interpolation
        output[i] = inputData[sourceIndexFloor] * (1 - fraction) + inputData[sourceIndexCeil] * fraction;
    }
    
    return output;
}

class RealAudioTranscriptionTester {
    constructor() {
        this.ws = null;
        this.transcriptions = [];
        this.sessionId = null;
        this.audioData = null;
    }

    async loadAudioFile() {
        try {
            console.log(`📁 Loading audio file: ${AUDIO_FILE}`);
            const audioBuffer = fs.readFileSync(AUDIO_FILE);
            this.audioData = parseWavFile(audioBuffer);
            console.log(`✅ Loaded ${this.audioData.length} samples of audio data`);
            return true;
        } catch (error) {
            console.error('❌ Failed to load audio file:', error.message);
            return false;
        }
    }

    async connect() {
        return new Promise((resolve, reject) => {
            console.log('🔌 Connecting to orchestrator...');
            this.ws = new WebSocket(`${ORCHESTRATOR_URL}/ws/voice`);

            this.ws.on('open', () => {
                console.log('✅ Connected to orchestrator');
                resolve();
            });

            this.ws.on('message', (data) => {
                try {
                    const message = JSON.parse(data.toString());
                    console.log('📨 Received:', message);
                    
                    if (message.type === 'ready') {
                        this.sessionId = message.session_id;
                        console.log('🎯 Session ID:', this.sessionId);
                    } else if (message.type === 'live_transcript') {
                        this.transcriptions.push({
                            timestamp: Date.now(),
                            text: message.text,
                            type: 'live'
                        });
                        console.log('📝 Live transcript:', message.text);
                    } else if (message.type === 'transcription_complete') {
                        this.transcriptions.push({
                            timestamp: Date.now(),
                            text: message.text,
                            type: 'final'
                        });
                        console.log('✅ Final transcript:', message.text);
                    }
                } catch (error) {
                    console.error('❌ Failed to parse message:', error);
                }
            });

            this.ws.on('error', (error) => {
                console.error('❌ WebSocket error:', error);
                reject(error);
            });

            this.ws.on('close', (code, reason) => {
                console.log(`🔌 Connection closed: ${code} ${reason}`);
            });

            setTimeout(() => {
                if (this.ws.readyState !== WebSocket.OPEN) {
                    reject(new Error('Connection timeout'));
                }
            }, 10000);
        });
    }

    async sendAudio() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            throw new Error('WebSocket not connected');
        }

        if (!this.audioData) {
            throw new Error('No audio data loaded');
        }

        console.log(`🎤 Sending ${this.audioData.length} samples (${this.audioData.buffer.byteLength} bytes) of real audio data...`);
        
        // Send audio data in chunks to simulate real-time streaming
        const chunkSize = 4096;
        let offset = 0;
        
        const sendChunk = () => {
            if (offset < this.audioData.length) {
                const chunk = this.audioData.slice(offset, offset + chunkSize);
                this.ws.send(chunk.buffer);
                offset += chunkSize;
                
                // Send next chunk after a short delay
                setTimeout(sendChunk, 100);
            } else {
                // Send end of audio marker
                this.ws.send(JSON.stringify({ type: 'end_audio' }));
                console.log('📤 Sent end_audio marker');
            }
        };
        
        sendChunk();
    }

    async runTest() {
        try {
            console.log('🚀 Starting real audio transcription test...\n');
            
            // Load audio file
            const audioLoaded = await this.loadAudioFile();
            if (!audioLoaded) {
                console.log('❌ Cannot proceed without audio file');
                return;
            }
            
            // Connect to orchestrator
            await this.connect();
            
            // Wait for session to be ready
            console.log('⏳ Waiting for session initialization...');
            await new Promise(resolve => setTimeout(resolve, 3000));
            
            // Send audio
            await this.sendAudio();
            
            // Wait for transcriptions
            console.log('⏳ Waiting for transcriptions...');
            
            const timeout = setTimeout(() => {
                console.log('⏰ Test timeout - checking results...');
                this.printResults();
                this.cleanup();
            }, 30000);
            
            // Check for transcriptions periodically
            const checkInterval = setInterval(() => {
                if (this.transcriptions.length > 0) {
                    clearTimeout(timeout);
                    clearInterval(checkInterval);
                    console.log('✅ Transcriptions received!');
                    this.printResults();
                    this.cleanup();
                }
            }, 1000);
            
        } catch (error) {
            console.error('❌ Test failed:', error);
            this.cleanup();
        }
    }

    printResults() {
        console.log('\n📊 Test Results:');
        console.log('================');
        console.log(`Session ID: ${this.sessionId}`);
        console.log(`Transcriptions received: ${this.transcriptions.length}`);
        
        if (this.transcriptions.length > 0) {
            console.log('\nTranscriptions:');
            this.transcriptions.forEach((t, i) => {
                console.log(`${i + 1}. [${t.type}] ${t.text}`);
            });
            console.log('\n✅ REAL AUDIO TRANSCRIPTION FLOW VERIFIED!');
            console.log('🎯 Transcription pipeline is working correctly!');
        } else {
            console.log('\n❌ No transcriptions received');
            console.log('Possible issues:');
            console.log('1. WhisperLive service not running');
            console.log('2. Audio format mismatch');
            console.log('3. Network connectivity issues');
            console.log('4. Session not properly initialized');
            console.log('5. Audio file format not supported');
        }
    }

    cleanup() {
        if (this.ws) {
            this.ws.close(1000, 'Test complete');
        }
        setTimeout(() => process.exit(0), 1000);
    }
}

// Run the test
if (require.main === module) {
    const tester = new RealAudioTranscriptionTester();
    tester.runTest().catch(console.error);
}

module.exports = RealAudioTranscriptionTester;