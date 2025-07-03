export class AudioRecorder {
  private audioContext: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private mediaStreamSource: MediaStreamAudioSourceNode | null = null;
  private isRecording = false;
  private onAudioDataCallback?: (audioData: Float32Array) => void;
  
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
      
      // Create media stream source
      this.mediaStreamSource = this.audioContext.createMediaStreamSource(this.stream);
      
      // Create script processor for raw audio data (4096 buffer size)
      this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
      
      // Handle audio processing like WhisperLive Chrome Extension
      this.processor.onaudioprocess = (event) => {
        if (!this.isRecording || !this.onAudioDataCallback) return;
        
        const inputData = event.inputBuffer.getChannelData(0);
        
        // Resample to 16kHz like Chrome Extension
        const resampledData = this.resampleTo16kHz(inputData, this.audioContext!.sampleRate);
        
        // Send resampled data
        this.onAudioDataCallback(resampledData);
      };
      
      // Connect the audio processing chain
      this.mediaStreamSource.connect(this.processor);
      this.processor.connect(this.audioContext.destination);
      
      return true;
    } catch (error) {
      console.error('Failed to initialize audio recorder:', error);
      return false;
    }
  }
  
  start(onAudioData: (audioData: Float32Array) => void) {
    if (!this.audioContext || !this.processor) {
      throw new Error('Audio recorder not initialized');
    }
    
    this.onAudioDataCallback = onAudioData;
    this.isRecording = true;
    
    // Resume audio context if suspended
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
  }
  
  stop() {
    this.isRecording = false;
    this.onAudioDataCallback = undefined;
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
    // Simplified audio level detection
    // In a real implementation, you'd analyze the audio stream
    return Math.random() * 0.5 + 0.1;
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
      source.start();
      
      // Return promise that resolves when audio finishes playing
      return new Promise((resolve) => {
        source.onended = () => resolve();
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
      source.start();
      
      // Return promise that resolves when audio finishes playing
      return new Promise((resolve) => {
        source.onended = () => resolve();
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
  
  cleanup() {
    this.audioContext.close();
  }
}