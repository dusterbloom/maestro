// AudioWorkletProcessor for audio recording and processing
class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.isRecording = false;
    this.voiceActivityThreshold = 0.02;
    this.port.onmessage = this.handleMessage.bind(this);
  }

  handleMessage(event) {
    const { type, data } = event.data;
    
    switch (type) {
      case 'start':
        this.isRecording = true;
        break;
      case 'stop':
        this.isRecording = false;
        break;
      case 'setThreshold':
        this.voiceActivityThreshold = data.threshold;
        break;
    }
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    
    if (input && input.length > 0) {
      const inputData = input[0]; // Get first channel
      
      // Calculate current audio level (RMS)
      let sum = 0;
      for (let i = 0; i < inputData.length; i++) {
        sum += inputData[i] * inputData[i];
      }
      const audioLevel = Math.sqrt(sum / inputData.length);
      
      // Send audio level for voice activity detection
      this.port.postMessage({
        type: 'audioLevel',
        data: { level: audioLevel, isActive: audioLevel > this.voiceActivityThreshold }
      });
      
      if (this.isRecording) {
        // Resample to 16kHz
        const resampledData = this.resampleTo16kHz(inputData, sampleRate);
        
        // Send resampled audio data
        this.port.postMessage({
          type: 'audioData',
          data: { audioData: resampledData }
        });
      }
    }
    
    return true; // Keep processor alive
  }

  resampleTo16kHz(audioData, origSampleRate = 44100) {
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
}

registerProcessor('audio-processor', AudioProcessor);