export class AudioProcessor {
  private audioContext: AudioContext | null = null
  private stream: MediaStream | null = null
  private source: MediaStreamAudioSourceNode | null = null
  private processor: ScriptProcessorNode | null = null
  private isProcessing = false
  private currentCallback: ((data: Float32Array) => void) | null = null
  
  constructor() {
    // AudioContext will be created when needed
  }
  
  async initialize(): Promise<void> {
    // Request microphone access
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    })
    
    // Create audio context
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
    
    // Create source from stream
    this.source = this.audioContext.createMediaStreamSource(this.stream)
    
    // Create processor for raw audio access
    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1)
  }
  
  startProcessing(onAudioData: (data: Float32Array) => void): void {
    if (!this.audioContext || !this.source || !this.processor) {
      throw new Error('AudioProcessor not initialized')
    }
    
    // Stop any existing processing first
    this.stopProcessing()
    
    this.isProcessing = true
    this.currentCallback = onAudioData
    
    this.processor.onaudioprocess = (event) => {
      if (this.isProcessing && this.currentCallback) {
        const inputData = event.inputBuffer.getChannelData(0)
        
        // Resample to 16kHz if needed
        const audioData16kHz = this.resampleTo16kHz(
          inputData, 
          this.audioContext!.sampleRate
        )
        
        this.currentCallback(audioData16kHz)
      }
    }
    
    // Connect the audio graph
    this.source.connect(this.processor)
    this.processor.connect(this.audioContext.destination)
    
    console.log('🎤 AudioProcessor: Started processing')
  }
  
  stopProcessing(): void {
    console.log('🛑 AudioProcessor: Stopping processing')
    this.isProcessing = false
    this.currentCallback = null
    
    if (this.processor && this.source) {
      try {
        this.source.disconnect()
        this.processor.disconnect()
      } catch (error) {
        console.warn('Error disconnecting audio nodes:', error)
      }
      
      // Clear the audio process handler to prevent any lingering callbacks
      this.processor.onaudioprocess = null
    }
    
    console.log('✅ AudioProcessor: Stopped processing')
  }
  
  private resampleTo16kHz(audioData: Float32Array, origSampleRate: number): Float32Array {
    const targetSampleRate = 16000
    
    if (origSampleRate === targetSampleRate) {
      return audioData
    }
    
    const targetLength = Math.round(audioData.length * (targetSampleRate / origSampleRate))
    const resampledData = new Float32Array(targetLength)
    
    const springFactor = (audioData.length - 1) / (targetLength - 1)
    resampledData[0] = audioData[0]
    resampledData[targetLength - 1] = audioData[audioData.length - 1]
    
    for (let i = 1; i < targetLength - 1; i++) {
      const index = i * springFactor
      const leftIndex = Math.floor(index)
      const rightIndex = Math.ceil(index)
      const fraction = index - leftIndex
      resampledData[i] = audioData[leftIndex] + (audioData[rightIndex] - audioData[leftIndex]) * fraction
    }
    
    return resampledData
  }
  
  getAudioLevel(audioData: Float32Array): number {
    let sum = 0
    for (let i = 0; i < audioData.length; i++) {
      sum += Math.abs(audioData[i])
    }
    return sum / audioData.length
  }
  
  async cleanup(): Promise<void> {
    this.stopProcessing()
    
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop())
      this.stream = null
    }
    
    if (this.audioContext && this.audioContext.state !== 'closed') {
      await this.audioContext.close()
      this.audioContext = null
    }
    
    this.source = null
    this.processor = null
  }
}