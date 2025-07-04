export class AudioRecorder {
  private audioContext: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private mediaStreamSource: MediaStreamAudioSourceNode | null = null;
  private isRecording = false;
  private onAudioDataCallback?: (audioData: Float32Array) => void;
  private onAudioLevelCallback?: (level: number) => void;
  
  // Voice activity detection for barge-in
  private currentAudioLevel = 0;
  private voiceActivityThreshold = 0.02; // Threshold for detecting speech
  private silenceDuration = 0;
  private maxSilenceMs = 1000; // 1 second of silence before stopping
  private lastVoiceActivityTime = 0;
  
  async initialize(): Promise<boolean> {
    try {
      // Request microphone access with specific constraints
      this.stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      });
      
      // Create audio context (will use default sample rate)
      this.audioContext = new AudioContext();
      
      // Load AudioWorklet processor
      await this.audioContext.audioWorklet.addModule('/audio-processor.js');
      
      // Create media stream source
      this.mediaStreamSource = this.audioContext.createMediaStreamSource(this.stream);
      
      // Create AudioWorkletNode
      this.workletNode = new AudioWorkletNode(this.audioContext, 'audio-processor');
      
      // Handle messages from worklet
      this.workletNode.port.onmessage = (event) => {
        const { type, data } = event.data;
        
        switch (type) {
          case 'audioLevel':
            this.currentAudioLevel = data.level;
            
            // Update voice activity detection
            const now = Date.now();
            if (data.isActive) {
              this.lastVoiceActivityTime = now;
              this.silenceDuration = 0;
            } else {
              this.silenceDuration = now - this.lastVoiceActivityTime;
            }
            
            // Notify about audio level for barge-in detection
            this.onAudioLevelCallback?.(this.currentAudioLevel);
            break;
            
          case 'audioData':
            if (this.isRecording && this.onAudioDataCallback) {
              this.onAudioDataCallback(data.audioData);
            }
            break;
        }
      };
      
      // Connect the audio processing chain
      this.mediaStreamSource.connect(this.workletNode);
      this.workletNode.connect(this.audioContext.destination);
      
      return true;
    } catch (error) {
      console.error('Failed to initialize audio recorder:', error);
      return false;
    }
  }
  
  start(onAudioData: (audioData: Float32Array) => void) {
    if (!this.audioContext || !this.workletNode) {
      throw new Error('Audio recorder not initialized');
    }
    
    this.onAudioDataCallback = onAudioData;
    this.isRecording = true;
    
    // Resume audio context if suspended
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
    
    // Send start message to worklet
    this.workletNode.port.postMessage({ type: 'start' });
  }
  
  stop() {
    this.isRecording = false;
    this.onAudioDataCallback = undefined;
    
    // Send stop message to worklet
    if (this.workletNode) {
      this.workletNode.port.postMessage({ type: 'stop' });
    }
  }
  
  /**
   * Resamples audio data to 16kHz (like WhisperLive Chrome Extension)
   */
  private resampleTo16kHz(audioData: Float32Array, origSampleRate: number = 44100): Float32Array {
    // Exact implementation from WhisperLive Chrome Extension
    const targetSampleRate = 16000;
    const targetLength = Math.round(audioData.length * (targetSampleRate / origSampleRate));
    const resampledData = new Float32Array(targetLength);
    
    if (targetLength === 0) return resampledData;
    
    const springFactor = (audioData.length - 1) / (targetLength - 1);
    resampledData[0] = audioData[0];
    resampledData[targetLength - 1] = audioData[audioData.length - 1];
    
    for (let i = 1; i < targetLength - 1; i++) {
      const index = i * springFactor;
      const leftIndex = Math.floor(index);
      const rightIndex = Math.ceil(index);
      const fraction = index - leftIndex;
      resampledData[i] = audioData[leftIndex] + (audioData[rightIndex] - audioData[leftIndex]) * fraction;
    }
    
    return resampledData;
  }
  
  getAudioLevel(): number {
    return this.currentAudioLevel;
  }
  
  isVoiceActive(): boolean {
    return this.currentAudioLevel > this.voiceActivityThreshold;
  }
  
  getSilenceDuration(): number {
    return this.silenceDuration;
  }
  
  setVoiceActivityThreshold(threshold: number) {
    this.voiceActivityThreshold = threshold;
  }
  
  onAudioLevel(callback: (level: number) => void) {
    this.onAudioLevelCallback = callback;
  }
  
  cleanup() {
    this.stop();
    
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    
    if (this.mediaStreamSource) {
      this.mediaStreamSource.disconnect();
      this.mediaStreamSource = null;
    }
    
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}

export class AudioPlayer {
  private audioContext: AudioContext;
  private gainNode: GainNode;
  private isStreaming: boolean = false;
  private streamingSource: AudioBufferSourceNode | null = null;
  private activeSources: AudioBufferSourceNode[] = [];
  private onPlaybackStartCallback?: () => void;
  private onPlaybackEndCallback?: () => void;
  
  constructor() {
    this.audioContext = new AudioContext();
    this.gainNode = this.audioContext.createGain();
    this.gainNode.connect(this.audioContext.destination);
  }
  
  async play(audioData: ArrayBuffer): Promise<void> {
    try {
      // Resume audio context if suspended (browser autoplay policy)
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }
      
      const audioBuffer = await this.audioContext.decodeAudioData(audioData);
      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.gainNode);
      
      // Track active sources for interruption capability
      this.activeSources.push(source);
      
      // Notify playback start
      this.onPlaybackStartCallback?.();
      
      source.start();
      
      // Return promise that resolves when audio finishes playing
      return new Promise((resolve) => {
        source.onended = () => {
          // Remove from active sources
          const index = this.activeSources.indexOf(source);
          if (index > -1) {
            this.activeSources.splice(index, 1);
          }
          
          // Notify playback end if no more active sources
          if (this.activeSources.length === 0) {
            this.onPlaybackEndCallback?.();
          }
          
          resolve();
        };
      });
    } catch (error) {
      console.error('Failed to play audio:', error);
      throw error;
    }
  }
  
  async playPCMChunk(pcmData: ArrayBuffer): Promise<void> {
    try {
      // Resume audio context if suspended
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }
      
      // PCM data from Kokoro: 16-bit signed integers at 24000 Hz
      const pcmInt16 = new Int16Array(pcmData);
      const sampleRate = 24000;
      const channels = 1;
      
      // Create audio buffer from PCM data
      const audioBuffer = this.audioContext.createBuffer(channels, pcmInt16.length, sampleRate);
      const outputData = audioBuffer.getChannelData(0);
      
      // Convert Int16 to Float32 (normalize to [-1, 1])
      for (let i = 0; i < pcmInt16.length; i++) {
        outputData[i] = pcmInt16[i] / 32768.0;
      }
      
      // Play the audio chunk
      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.gainNode);
      
      // Track active sources for interruption capability
      this.activeSources.push(source);
      
      // Start playback notification on first chunk
      if (this.activeSources.length === 1) {
        this.onPlaybackStartCallback?.();
      }
      
      source.start();
      
      // Return promise that resolves when audio finishes playing
      return new Promise((resolve) => {
        source.onended = () => {
          // Remove from active sources
          const index = this.activeSources.indexOf(source);
          if (index > -1) {
            this.activeSources.splice(index, 1);
          }
          
          // Notify playback end if no more active sources
          if (this.activeSources.length === 0) {
            this.onPlaybackEndCallback?.();
          }
          
          resolve();
        };
      });
    } catch (error) {
      console.error('Failed to play PCM chunk:', error);
      throw error;
    }
  }
  
  setVolume(volume: number) {
    // Volume should be between 0 and 1
    this.gainNode.gain.value = Math.max(0, Math.min(1, volume));
  }
  
  /**
   * Stop all currently playing audio immediately (for barge-in)
   */
  stopAll() {
    console.log(`Stopping ${this.activeSources.length} active audio sources`);
    this.activeSources.forEach(source => {
      try {
        source.stop();
      } catch (e) {
        // Source might already be stopped
        console.warn('Failed to stop audio source:', e);
      }
    });
    this.activeSources = [];
    this.onPlaybackEndCallback?.();
  }
  
  /**
   * Check if audio is currently playing
   */
  isPlaying(): boolean {
    return this.activeSources.length > 0;
  }
  
  /**
   * Set callback for when playback starts
   */
  onPlaybackStart(callback: () => void) {
    this.onPlaybackStartCallback = callback;
  }
  
  /**
   * Set callback for when all playback ends
   */
  onPlaybackEnd(callback: () => void) {
    this.onPlaybackEndCallback = callback;
  }
  
  cleanup() {
    this.stopAll();
    this.audioContext.close();
  }
}