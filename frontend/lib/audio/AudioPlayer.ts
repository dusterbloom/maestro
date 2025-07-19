interface QueuedAudio {
  sequence: number
  text: string
  audioData: string // base64
  audioUrl?: string
  audio?: HTMLAudioElement
}

export class AudioPlayer {
  private queue: QueuedAudio[] = []
  private isPlaying = false
  private currentAudio: HTMLAudioElement | null = null
  private abortController: AbortController | null = null
  
  async playAudio(audioData: string, sequence: number, text: string): Promise<void> {
    const queueItem: QueuedAudio = {
      sequence,
      text,
      audioData
    }
    
    // Add to queue
    this.queue.push(queueItem)
    this.queue.sort((a, b) => a.sequence - b.sequence)
    
    // Start processing if not already playing
    if (!this.isPlaying) {
      this.processQueue()
    }
  }


  
  private async processQueue(): Promise<void> {
    if (this.queue.length === 0 || this.isPlaying) {
      return
    }
    
    this.isPlaying = true
    
    while (this.queue.length > 0 && !this.abortController?.signal.aborted) {
      const item = this.queue.shift()!
      
      try {
        await this.playQueuedItem(item)
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
          console.log('Audio playback aborted')
          break
        }
        console.error('Error playing audio:', error)
      }
    }
    
    this.isPlaying = false
  }
  
  private async playQueuedItem(item: QueuedAudio): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // Convert base64 to blob
        const audioData = atob(item.audioData)
        const arrayBuffer = new ArrayBuffer(audioData.length)
        const uint8Array = new Uint8Array(arrayBuffer)
        
        for (let i = 0; i < audioData.length; i++) {
          uint8Array[i] = audioData.charCodeAt(i)
        }
        
        const audioBlob = new Blob([uint8Array], { type: 'audio/wav' })
        item.audioUrl = URL.createObjectURL(audioBlob)
        
        // Create audio element
        const audio = new Audio(item.audioUrl)
        this.currentAudio = audio
        item.audio = audio
        
        // Set up event handlers
        audio.onended = () => {
          this.cleanup(item)
          resolve()
        }
        
        audio.onerror = (error) => {
          this.cleanup(item)
          reject(error)
        }
        
        // Check for abort before playing
        if (this.abortController?.signal.aborted) {
          this.cleanup(item)
          reject(new Error('AbortError'))
          return
        }
        
        // Play the audio
        audio.play().catch(error => {
          this.cleanup(item)
          reject(error)
        })
        
      } catch (error) {
        reject(error)
      }
    })
  }
  
  private cleanup(item: QueuedAudio): void {
    if (item.audioUrl) {
      URL.revokeObjectURL(item.audioUrl)
    }
    if (item.audio === this.currentAudio) {
      this.currentAudio = null
    }
  }
  
    // Inside AudioPlayer class
  public pause(): void {
      if (this.currentAudio && !this.currentAudio.paused) {
          this.currentAudio.pause();
          console.log('Audio playback paused.');
      }
  }

  public resume(): void {
      if (this.currentAudio && this.currentAudio.paused) {
          this.currentAudio.play().catch(error => console.error("Resume failed", error));
          console.log('Audio playback resumed.');
      }
  }
  
  interrupt(): void {
    // Create new abort controller to signal interruption
    this.abortController = new AbortController()
    this.abortController.abort()
    
    // Stop current audio
    if (this.currentAudio) {
      this.currentAudio.pause()
      this.currentAudio = null
    }
    
    // Clear the queue
    this.queue.forEach(item => {
      if (item.audioUrl) {
        URL.revokeObjectURL(item.audioUrl)
      }
    })
    this.queue = []
    
    this.isPlaying = false
  }
  
  get queueLength(): number {
    return this.queue.length
  }
  
  get playing(): boolean {
    return this.isPlaying
  }
}