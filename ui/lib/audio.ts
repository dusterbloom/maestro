export class AudioRecorder {
  private mediaRecorder: MediaRecorder | null = null;
  private audioContext: AudioContext | null = null;
  private chunks: Blob[] = [];
  private stream: MediaStream | null = null;
  
  async initialize(): Promise<boolean> {
    try {
      // Request microphone access
      this.stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        } 
      });
      
      // Create audio context for processing
      this.audioContext = new AudioContext({ sampleRate: 16000 });
      
      // Create media recorder with optimal settings
      const options = { 
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 16000 
      };
      
      if (MediaRecorder.isTypeSupported(options.mimeType)) {
        this.mediaRecorder = new MediaRecorder(this.stream, options);
      } else {
        // Fallback to default format
        this.mediaRecorder = new MediaRecorder(this.stream);
      }
      
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.chunks.push(event.data);
        }
      };
      
      return true;
    } catch (error) {
      console.error('Failed to initialize audio recorder:', error);
      return false;
    }
  }
  
  start() {
    if (!this.mediaRecorder) {
      throw new Error('Audio recorder not initialized');
    }
    
    this.chunks = [];
    // Record in small chunks for real-time processing
    this.mediaRecorder.start(100); // 100ms chunks
  }
  
  async stop(): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
      if (!this.mediaRecorder) {
        reject(new Error('Audio recorder not initialized'));
        return;
      }
      
      this.mediaRecorder.onstop = async () => {
        try {
          const blob = new Blob(this.chunks, { type: 'audio/webm' });
          const arrayBuffer = await blob.arrayBuffer();
          
          // Convert to the format expected by the backend (if needed)
          const processedBuffer = await this.processAudioBuffer(arrayBuffer);
          resolve(processedBuffer);
        } catch (error) {
          reject(error);
        }
      };
      
      this.mediaRecorder.stop();
    });
  }
  
  private async processAudioBuffer(buffer: ArrayBuffer): Promise<ArrayBuffer> {
    // For now, return the buffer as-is
    // In a production system, you might want to:
    // 1. Convert to the exact format expected by WhisperLive
    // 2. Apply audio preprocessing (noise reduction, normalization)
    // 3. Resample to 16kHz if needed
    return buffer;
  }
  
  getAudioLevel(): number {
    // Simplified audio level detection
    // In a real implementation, you'd analyze the audio stream
    return Math.random() * 0.5 + 0.1;
  }
  
  cleanup() {
    if (this.mediaRecorder) {
      this.mediaRecorder.stop();
      this.mediaRecorder = null;
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
  
  setVolume(volume: number) {
    // Volume should be between 0 and 1
    this.gainNode.gain.value = Math.max(0, Math.min(1, volume));
  }
  
  cleanup() {
    this.audioContext.close();
  }
}