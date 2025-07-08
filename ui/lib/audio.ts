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
  private voiceActivityThreshold = 0.1; // Threshold for detecting speech (increased from 0.02 to match professional VAD)
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
    
    // Send threshold update to worklet
    if (this.workletNode) {
      this.workletNode.port.postMessage({ 
        type: 'setThreshold', 
        data: { threshold } 
      });
    }
  }
  
  onAudioLevel(callback: (level: number) => void) {
    this.onAudioLevelCallback = callback;
  }
  
  getVoiceActivityThreshold(): number {
    return this.voiceActivityThreshold;
  }
  
  cleanup() {
    this.stop();
    
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
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

  // --- Streaming queue state ---
  private chunkQueue: ArrayBuffer[] = [];
  private isPlayingQueue: boolean = false;
  private endOfSentence: boolean = false;

  constructor() {
    this.audioContext = new AudioContext();
    this.gainNode = this.audioContext.createGain();
    this.gainNode.connect(this.audioContext.destination);
  }

  /**
   * Enqueue an audio chunk for sequential playback.
   */
  enqueueChunk(audioData: ArrayBuffer) {
    this.chunkQueue.push(audioData);
    if (!this.isPlayingQueue) {
      this.playNextChunk();
    }
  }

  /**
   * Signal the end of the current sentence's audio stream.
   */
  endAudioStream() {
    this.endOfSentence = true;
    // If nothing is playing and queue is empty, fire playback end callback
    if (!this.isPlayingQueue && this.chunkQueue.length === 0) {
      this.onPlaybackEndCallback?.();
    }
  }

  /**
   * Play the next chunk in the queue, if any.
   */
  private async playNextChunk() {
    if (this.chunkQueue.length === 0) {
      this.isPlayingQueue = false;
      // If end of sentence was signaled, fire playback end callback
      if (this.endOfSentence) {
        this.onPlaybackEndCallback?.();
        this.endOfSentence = false;
      }
      return;
    }
    this.isPlayingQueue = true;
    const audioData = this.chunkQueue.shift()!;
    try {
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }
      const audioBuffer = await this.audioContext.decodeAudioData(audioData);
      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.gainNode);

      this.activeSources.push(source);
      if (this.activeSources.length === 1) {
        this.onPlaybackStartCallback?.();
      }

      source.start();

      source.onended = () => {
        const index = this.activeSources.indexOf(source);
        if (index > -1) {
          this.activeSources.splice(index, 1);
        }
        // Play next chunk in the queue
        this.playNextChunk();
      };
    } catch (error) {
      console.error('Failed to play audio chunk:', error);
      // Try to play next chunk even if this one fails
      this.playNextChunk();
    }
  }
  
  // Optionally, you can keep playPCMChunk for PCM support, but it should also use the queue if needed.
  async playPCMChunk(pcmData: ArrayBuffer): Promise<void> {
    // For now, just enqueue as a normal chunk (assuming WAV/PCM is handled by decodeAudioData)
    this.enqueueChunk(pcmData);
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
    this.chunkQueue = [];
    this.isPlayingQueue = false;
    this.endOfSentence = false;
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