import { jest, describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { VoiceWebSocket } from '../lib/websocket';
import { AudioRecorder, AudioPlayer } from '../lib/audio';

// Mock fetch globally
global.fetch = jest.fn() as jest.MockedFunction<typeof fetch>;

// Mock AudioContext and related Web Audio APIs
const mockAudioContext = {
  createGain: jest.fn(() => ({
    gain: { value: 1 },
    connect: jest.fn(),
  })),
  createBufferSource: jest.fn(() => ({
    buffer: null,
    connect: jest.fn(),
    start: jest.fn(),
    stop: jest.fn(),
    onended: null,
  })),
  createBuffer: jest.fn(() => ({
    getChannelData: jest.fn(() => new Float32Array(1024)),
  })),
  decodeAudioData: jest.fn(() => Promise.resolve({
    getChannelData: jest.fn(() => new Float32Array(1024)),
  })),
  resume: jest.fn(() => Promise.resolve()),
  close: jest.fn(() => Promise.resolve()),
  state: 'running',
  destination: {},
};

// Mock AudioWorklet
const mockAudioWorklet = {
  addModule: jest.fn(() => Promise.resolve()),
};

// Mock MediaDevices
const mockMediaDevices = {
  getUserMedia: jest.fn(() => Promise.resolve({
    getTracks: jest.fn(() => [
      { stop: jest.fn() }
    ]),
  })),
};

// Setup global mocks
(global as any).AudioContext = jest.fn(() => mockAudioContext);
(global as any).AudioWorkletNode = jest.fn(() => ({
  port: {
    onmessage: null,
    postMessage: jest.fn(),
  },
  connect: jest.fn(),
  disconnect: jest.fn(),
}));
(global as any).navigator = {
  mediaDevices: mockMediaDevices,
};
(global as any).WebSocket = jest.fn();

describe('TTS Interruption System', () => {
  let voiceWebSocket: VoiceWebSocket;
  let audioPlayer: AudioPlayer;
  let mockFetch: jest.MockedFunction<typeof fetch>;

  beforeEach(() => {
    mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
    mockFetch.mockClear();
    voiceWebSocket = new VoiceWebSocket('ws://localhost:9090');
    audioPlayer = new AudioPlayer();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('VoiceWebSocket TTS Interruption', () => {
    it('should send interrupt request successfully', async () => {
      // Mock successful API response
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: 'interrupted',
          session_id: 'test_session',
          interrupt_time_ms: 15.5,
          message: 'TTS generation interrupted successfully'
        }),
      } as Response);

      const result = await voiceWebSocket.sendInterruptTts('test_session');

      expect(mockFetch).toHaveBeenCalledWith('/api/interrupt-tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: 'test_session'
        })
      });

      expect(result.success).toBe(true);
      expect(result.message).toBe('TTS generation interrupted successfully');
    });

    it('should handle interrupt request failure', async () => {
      // Mock failed API response
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const result = await voiceWebSocket.sendInterruptTts('test_session');

      expect(result.success).toBe(false);
      expect(result.message).toBe('Network error');
    });

    it('should handle HTTP error responses', async () => {
      // Mock HTTP error response
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      } as Response);

      const result = await voiceWebSocket.sendInterruptTts('test_session');

      expect(result.success).toBe(false);
      expect(result.message).toBe('HTTP 500');
    });

    it('should handle no active session response', async () => {
      // Mock no active session response
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: 'no_active_session',
          session_id: 'test_session',
          interrupt_time_ms: 5.2,
          message: 'No active TTS session to interrupt'
        }),
      } as Response);

      const result = await voiceWebSocket.sendInterruptTts('test_session');

      expect(result.success).toBe(false);
      expect(result.message).toBe('No active TTS session to interrupt');
    });

    it('should call interruption acknowledgment callback', async () => {
      const mockCallback = jest.fn();
      voiceWebSocket.onInterruptionAck(mockCallback);

      // Mock successful response
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: 'interrupted',
          session_id: 'test_session',
          message: 'TTS generation interrupted successfully'
        }),
      } as Response);

      await voiceWebSocket.sendInterruptTts('test_session');

      expect(mockCallback).toHaveBeenCalledWith(true, 'TTS generation interrupted successfully');
    });

    it('should call interruption acknowledgment callback on error', async () => {
      const mockCallback = jest.fn();
      voiceWebSocket.onInterruptionAck(mockCallback);

      // Mock error
      mockFetch.mockRejectedValueOnce(new Error('Network failure'));

      await voiceWebSocket.sendInterruptTts('test_session');

      expect(mockCallback).toHaveBeenCalledWith(false, 'Network failure');
    });
  });

  describe('AudioPlayer Interruption', () => {
    it('should stop all active audio sources', () => {
      // Mock active audio sources
      const mockSource1 = { stop: jest.fn() };
      const mockSource2 = { stop: jest.fn() };
      
      // Simulate active sources (we'd need to access private property for testing)
      (audioPlayer as any).activeSources = [mockSource1, mockSource2];

      audioPlayer.stopAll();

      expect(mockSource1.stop).toHaveBeenCalled();
      expect(mockSource2.stop).toHaveBeenCalled();
      expect((audioPlayer as any).activeSources).toHaveLength(0);
    });

    it('should handle stop errors gracefully', () => {
      // Mock source that throws error on stop
      const mockSource = { 
        stop: jest.fn(() => { throw new Error('Already stopped'); })
      };
      
      (audioPlayer as any).activeSources = [mockSource];

      // Should not throw
      expect(() => audioPlayer.stopAll()).not.toThrow();
      expect((audioPlayer as any).activeSources).toHaveLength(0);
    });

    it('should report playing state correctly', () => {
      // Initially not playing
      expect(audioPlayer.isPlaying()).toBe(false);

      // Mock active sources
      (audioPlayer as any).activeSources = [{ stop: jest.fn() }];
      expect(audioPlayer.isPlaying()).toBe(true);

      // After stopping all
      audioPlayer.stopAll();
      expect(audioPlayer.isPlaying()).toBe(false);
    });

    it('should call playback end callback when stopping all', () => {
      const mockCallback = jest.fn();
      audioPlayer.onPlaybackEnd(mockCallback);

      // Mock active sources
      (audioPlayer as any).activeSources = [{ stop: jest.fn() }];

      audioPlayer.stopAll();

      expect(mockCallback).toHaveBeenCalled();
    });
  });

  describe('AudioRecorder Voice Activity Detection', () => {
    let audioRecorder: AudioRecorder;

    beforeEach(async () => {
      audioRecorder = new AudioRecorder();
      // Mock successful initialization
      await audioRecorder.initialize();
    });

    afterEach(() => {
      audioRecorder.cleanup();
    });

    it('should detect voice activity above threshold', () => {
      // Set a low threshold
      audioRecorder.setVoiceActivityThreshold(0.01);
      
      // Simulate high audio level
      (audioRecorder as any).currentAudioLevel = 0.05;

      expect(audioRecorder.isVoiceActive()).toBe(true);
    });

    it('should not detect voice activity below threshold', () => {
      // Set a higher threshold
      audioRecorder.setVoiceActivityThreshold(0.05);
      
      // Simulate low audio level
      (audioRecorder as any).currentAudioLevel = 0.01;

      expect(audioRecorder.isVoiceActive()).toBe(false);
    });

    it('should track silence duration', () => {
      // Simulate voice activity timing
      const now = Date.now();
      (audioRecorder as any).lastVoiceActivityTime = now - 1000; // 1 second ago
      (audioRecorder as any).silenceDuration = 1000;

      expect(audioRecorder.getSilenceDuration()).toBe(1000);
    });

    it('should call audio level callback', () => {
      const mockCallback = jest.fn();
      audioRecorder.onAudioLevel(mockCallback);

      // Simulate audio level update
      const testLevel = 0.03;
      (audioRecorder as any).currentAudioLevel = testLevel;
      (audioRecorder as any).onAudioLevelCallback?.(testLevel);

      expect(mockCallback).toHaveBeenCalledWith(testLevel);
    });
  });

  describe('Interruption Integration Scenarios', () => {
    it('should handle rapid successive interruptions', async () => {
      const sessionId = 'rapid_test_session';
      
      // Mock successful responses
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          status: 'interrupted',
          session_id: sessionId,
          message: 'TTS generation interrupted successfully'
        }),
      } as Response);

      // Send multiple rapid interruptions
      const promises = Array(5).fill(0).map(() => 
        voiceWebSocket.sendInterruptTts(sessionId)
      );

      const results = await Promise.all(promises);

      // All should succeed
      results.forEach(result => {
        expect(result.success).toBe(true);
      });

      // Should have made 5 API calls
      expect(mockFetch).toHaveBeenCalledTimes(5);
    });

    it('should handle interruption during audio playback', () => {
      // Mock playing state
      (audioPlayer as any).activeSources = [
        { stop: jest.fn() },
        { stop: jest.fn() }
      ];

      expect(audioPlayer.isPlaying()).toBe(true);

      // Interrupt
      audioPlayer.stopAll();

      expect(audioPlayer.isPlaying()).toBe(false);
    });

    it('should handle interruption with voice activity detection', () => {
      const mockCallback = jest.fn();
      
      // Set up voice activity monitoring
      audioRecorder.onAudioLevel((level) => {
        if (audioRecorder.isVoiceActive() && audioPlayer.isPlaying()) {
          // Simulate barge-in detection
          mockCallback('barge-in-detected');
          audioPlayer.stopAll();
        }
      });

      // Simulate TTS playing
      (audioPlayer as any).activeSources = [{ stop: jest.fn() }];

      // Simulate voice activity above threshold
      audioRecorder.setVoiceActivityThreshold(0.02);
      (audioRecorder as any).currentAudioLevel = 0.05;
      (audioRecorder as any).onAudioLevelCallback?.(0.05);

      expect(mockCallback).toHaveBeenCalledWith('barge-in-detected');
      expect(audioPlayer.isPlaying()).toBe(false);
    });
  });
});