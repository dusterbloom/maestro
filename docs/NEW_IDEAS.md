# Maestro Improvement Plan & Implementation

## Phase 1: Configuration review and improvement

### 1.1 Review and improve Centralized Configuration System

**File: `orchestrator/src/config.py`**
```python
from pydantic import BaseSettings, Field
from typing import Optional, List
from enum import Enum

class STTModel(str, Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class Language(str, Enum):
    AUTO = "auto"
    EN = "en"
    IT = "it"
    ES = "es"
    FR = "fr"
    DE = "de"

class TTSEngine(str, Enum):
    KOKORO = "kokoro"
    COQUI = "coqui"
    EDGE = "edge"

class AppConfig(BaseSettings):
    # Service URLs
    whisper_url: str = Field(default="http://whisper-live:9090")
    ollama_url: str = Field(default="http://host.docker.internal:11434")
    tts_url: str = Field(default="http://kokoro:8880/v1")
    
    # STT Configuration
    stt_model: STTModel = Field(default=STTModel.BASE)
    stt_language: Language = Field(default=Language.AUTO)
    stt_silence_threshold: float = Field(default=0.5, ge=0.1, le=2.0)
    
    # LLM Configuration
    llm_model: str = Field(default="gemma2:9b")
    llm_max_tokens: int = Field(default=2048, ge=100, le=8192)
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_system_prompt: str = Field(default="You are a helpful AI assistant. Keep responses concise but informative.")
    
    # TTS Configuration
    tts_engine: TTSEngine = Field(default=TTSEngine.KOKORO)
    tts_voice: str = Field(default="af_bella")
    tts_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    tts_language: Language = Field(default=Language.EN)
    
    # Performance Settings
    target_latency_ms: int = Field(default=500, ge=200, le=2000)
    audio_chunk_size: int = Field(default=1024)
    
    # UI Settings
    show_chat_history: bool = Field(default=True)
    auto_scroll_chat: bool = Field(default=True)
    theme: str = Field(default="dark")
    
    class Config:
        env_file = ".env"
        env_prefix = "MAESTRO_"

# Global config instance
config = AppConfig()

# Voice mapping based on language
VOICE_MAPPING = {
    Language.EN: ["af_bella", "af_sarah", "af_joe"],
    Language.IT: ["it_male", "it_female"],
    Language.ES: ["es_male", "es_female"],
    Language.FR: ["fr_male", "fr_female"],
}

def get_voices_for_language(language: Language) -> List[str]:
    return VOICE_MAPPING.get(language, VOICE_MAPPING[Language.EN])
```

### 1.2 Settings Management API

**File: `orchestrator/src/settings_manager.py`**
```python
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import aioredis
import json
from .config import config, AppConfig, STTModel, Language, TTSEngine, get_voices_for_language
from .ollama_client import get_available_models

router = APIRouter(prefix="/api/settings", tags=["settings"])

class SettingsManager:
    def __init__(self):
        self.redis = None
        self.current_config = config
    
    async def init_redis(self):
        try:
            self.redis = await aioredis.from_url("redis://redis:6379")
        except:
            self.redis = None  # Fallback to in-memory
    
    async def get_settings(self) -> Dict[str, Any]:
        """Get current settings"""
        return {
            "stt": {
                "model": self.current_config.stt_model,
                "language": self.current_config.stt_language,
                "silence_threshold": self.current_config.stt_silence_threshold,
                "available_models": [model.value for model in STTModel],
                "available_languages": [lang.value for lang in Language]
            },
            "llm": {
                "model": self.current_config.llm_model,
                "max_tokens": self.current_config.llm_max_tokens,
                "temperature": self.current_config.llm_temperature,
                "system_prompt": self.current_config.llm_system_prompt,
                "available_models": await get_available_models()
            },
            "tts": {
                "engine": self.current_config.tts_engine,
                "voice": self.current_config.tts_voice,
                "speed": self.current_config.tts_speed,
                "language": self.current_config.tts_language,
                "available_engines": [engine.value for engine in TTSEngine],
                "available_voices": get_voices_for_language(self.current_config.tts_language)
            },
            "performance": {
                "target_latency_ms": self.current_config.target_latency_ms,
                "audio_chunk_size": self.current_config.audio_chunk_size
            },
            "ui": {
                "show_chat_history": self.current_config.show_chat_history,
                "auto_scroll_chat": self.current_config.auto_scroll_chat,
                "theme": self.current_config.theme
            }
        }
    
    async def update_settings(self, settings: Dict[str, Any]):
        """Update settings and persist them"""
        # Validate and update config
        try:
            # Update the config object
            for category, values in settings.items():
                for key, value in values.items():
                    if hasattr(self.current_config, f"{category}_{key}"):
                        setattr(self.current_config, f"{category}_{key}", value)
                    elif hasattr(self.current_config, key):
                        setattr(self.current_config, key, value)
            
            # Persist to Redis if available
            if self.redis:
                await self.redis.set("maestro:settings", json.dumps(settings))
            
            return {"status": "success", "message": "Settings updated"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid settings: {str(e)}")
    
    async def reset_settings(self):
        """Reset to default settings"""
        self.current_config = AppConfig()
        if self.redis:
            await self.redis.delete("maestro:settings")
        return {"status": "success", "message": "Settings reset to defaults"}

settings_manager = SettingsManager()

@router.get("/")
async def get_settings():
    return await settings_manager.get_settings()

@router.post("/update")
async def update_settings(settings: Dict[str, Any]):
    return await settings_manager.update_settings(settings)

@router.post("/reset")
async def reset_settings():
    return await settings_manager.reset_settings()

@router.get("/voices/{language}")
async def get_voices_for_lang(language: str):
    try:
        lang_enum = Language(language)
        return {"voices": get_voices_for_language(lang_enum)}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid language")
```

## Phase 2: Enhanced Chat UI

### 2.1 Chat History Manager

**File: `orchestrator/src/chat_manager.py`**
```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import uuid

class ChatMessage(BaseModel):
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    audio_url: Optional[str] = None
    metadata: Dict[str, Any] = {}

class ChatSession(BaseModel):
    id: str
    messages: List[ChatMessage] = []
    created_at: datetime
    last_activity: datetime

class ChatManager:
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.current_session_id: Optional[str] = None
    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        session = ChatSession(
            id=session_id,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        self.sessions[session_id] = session
        self.current_session_id = session_id
        return session_id
    
    def add_message(self, role: str, content: str, audio_url: Optional[str] = None, metadata: Dict[str, Any] = None) -> ChatMessage:
        if not self.current_session_id:
            self.create_session()
        
        message = ChatMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            audio_url=audio_url,
            metadata=metadata or {}
        )
        
        session = self.sessions[self.current_session_id]
        session.messages.append(message)
        session.last_activity = datetime.now()
        
        return message
    
    def get_session_messages(self, session_id: Optional[str] = None) -> List[ChatMessage]:
        session_id = session_id or self.current_session_id
        if session_id and session_id in self.sessions:
            return self.sessions[session_id].messages
        return []
    
    def clear_session(self, session_id: Optional[str] = None):
        session_id = session_id or self.current_session_id
        if session_id and session_id in self.sessions:
            self.sessions[session_id].messages = []
    
    def get_context_for_llm(self, max_messages: int = 10) -> str:
        messages = self.get_session_messages()
        recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
        
        context = []
        for msg in recent_messages:
            context.append(f"{msg.role}: {msg.content}")
        
        return "\n".join(context)

chat_manager = ChatManager()
```

### 2.2 Enhanced Frontend Chat Component

**File: `ui/src/components/ChatInterface.tsx`**
```typescript
import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage, ChatSession } from '../types/chat';
import { Settings } from '../types/settings';

interface ChatInterfaceProps {
  messages: ChatMessage[];
  isRecording: boolean;
  isProcessing: boolean;
  settings: Settings;
  onSettingsChange: (settings: Partial<Settings>) => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  isRecording,
  isProcessing,
  settings,
  onSettingsChange
}) => {
  const [showSettings, setShowSettings] = useState(false);
  const [showChat, setShowChat] = useState(settings.ui.show_chat_history);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (settings.ui.auto_scroll_chat) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="flex justify-between items-center p-4 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center space-x-4">
          <h1 className="text-xl font-bold">Maestro AI</h1>
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-red-500 animate-pulse' : 'bg-gray-500'}`} />
            <span className="text-sm text-gray-300">
              {isRecording ? 'Recording' : isProcessing ? 'Processing' : 'Ready'}
            </span>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowChat(!showChat)}
            className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
          >
            {showChat ? 'Hide Chat' : 'Show Chat'}
          </button>
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 hover:bg-gray-700 rounded transition-colors"
          >
            ⚙️
          </button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Chat Messages */}
        {showChat && (
          <div className="flex-1 flex flex-col">
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.length === 0 ? (
                <div className="text-center text-gray-400 mt-8">
                  <p>No messages yet. Start a conversation!</p>
                </div>
              ) : (
                messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                        message.role === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-700 text-gray-100'
                      }`}
                    >
                      <p className="text-sm">{message.content}</p>
                      <div className="flex justify-between items-center mt-1">
                        <span className="text-xs opacity-75">
                          {formatTime(message.timestamp)}
                        </span>
                        {message.audio_url && (
                          <button
                            onClick={() => {
                              const audio = new Audio(message.audio_url);
                              audio.play();
                            }}
                            className="text-xs hover:underline opacity-75"
                          >
                            🔊
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>
        )}

        {/* Settings Panel */}
        {showSettings && (
          <SettingsPanel
            settings={settings}
            onSettingsChange={onSettingsChange}
            onClose={() => setShowSettings(false)}
          />
        )}
      </div>
    </div>
  );
};
```

## Phase 3: Dynamic Settings Interface

### 3.1 Settings Panel Component

**File: `ui/src/components/SettingsPanel.tsx`**
```typescript
import React, { useState, useEffect } from 'react';
import { Settings, STTModel, Language, TTSEngine } from '../types/settings';

interface SettingsPanelProps {
  settings: Settings;
  onSettingsChange: (settings: Partial<Settings>) => void;
  onClose: () => void;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  settings,
  onSettingsChange,
  onClose
}) => {
  const [localSettings, setLocalSettings] = useState(settings);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [availableVoices, setAvailableVoices] = useState<string[]>([]);

  useEffect(() => {
    // Fetch available models from Ollama
    fetchAvailableModels();
  }, []);

  useEffect(() => {
    // Fetch available voices when language changes
    fetchAvailableVoices(localSettings.tts.language);
  }, [localSettings.tts.language]);

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch('/api/ollama/models');
      const data = await response.json();
      setAvailableModels(data.models || []);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  const fetchAvailableVoices = async (language: string) => {
    try {
      const response = await fetch(`/api/settings/voices/${language}`);
      const data = await response.json();
      setAvailableVoices(data.voices || []);
    } catch (error) {
      console.error('Failed to fetch voices:', error);
    }
  };

  const handleSettingChange = (category: string, key: string, value: any) => {
    const newSettings = {
      ...localSettings,
      [category]: {
        ...localSettings[category],
        [key]: value
      }
    };
    setLocalSettings(newSettings);
  };

  const applySettings = () => {
    onSettingsChange(localSettings);
    onClose();
  };

  const resetSettings = async () => {
    try {
      await fetch('/api/settings/reset', { method: 'POST' });
      window.location.reload(); // Reload to get fresh settings
    } catch (error) {
      console.error('Failed to reset settings:', error);
    }
  };

  return (
    <div className="w-80 bg-gray-800 border-l border-gray-700 p-4 overflow-y-auto">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-lg font-semibold">Settings</h2>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-white transition-colors"
        >
          ✕
        </button>
      </div>

      {/* STT Settings */}
      <div className="mb-6">
        <h3 className="text-md font-medium mb-3 text-blue-300">Speech-to-Text</h3>
        
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium mb-1">Model</label>
            <select
              value={localSettings.stt.model}
              onChange={(e) => handleSettingChange('stt', 'model', e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
            >
              {Object.values(STTModel).map(model => (
                <option key={model} value={model}>{model.toUpperCase()}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Language</label>
            <select
              value={localSettings.stt.language}
              onChange={(e) => handleSettingChange('stt', 'language', e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
            >
              {Object.values(Language).map(lang => (
                <option key={lang} value={lang}>
                  {lang === 'auto' ? 'Auto-detect' : lang.toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Silence Threshold: {localSettings.stt.silence_threshold}s
            </label>
            <input
              type="range"
              min="0.1"
              max="2.0"
              step="0.1"
              value={localSettings.stt.silence_threshold}
              onChange={(e) => handleSettingChange('stt', 'silence_threshold', parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* LLM Settings */}
      <div className="mb-6">
        <h3 className="text-md font-medium mb-3 text-green-300">Language Model</h3>
        
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium mb-1">Model</label>
            <select
              value={localSettings.llm.model}
              onChange={(e) => handleSettingChange('llm', 'model', e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
            >
              {availableModels.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Max Tokens: {localSettings.llm.max_tokens}
            </label>
            <input
              type="range"
              min="100"
              max="8192"
              step="100"
              value={localSettings.llm.max_tokens}
              onChange={(e) => handleSettingChange('llm', 'max_tokens', parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Temperature: {localSettings.llm.temperature}
            </label>
            <input
              type="range"
              min="0.0"
              max="2.0"
              step="0.1"
              value={localSettings.llm.temperature}
              onChange={(e) => handleSettingChange('llm', 'temperature', parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">System Prompt</label>
            <textarea
              value={localSettings.llm.system_prompt}
              onChange={(e) => handleSettingChange('llm', 'system_prompt', e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm h-20 resize-none"
              placeholder="Enter system prompt..."
            />
          </div>
        </div>
      </div>

      {/* TTS Settings */}
      <div className="mb-6">
        <h3 className="text-md font-medium mb-3 text-purple-300">Text-to-Speech</h3>
        
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium mb-1">Language</label>
            <select
              value={localSettings.tts.language}
              onChange={(e) => handleSettingChange('tts', 'language', e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
            >
              {Object.values(Language).filter(lang => lang !== 'auto').map(lang => (
                <option key={lang} value={lang}>{lang.toUpperCase()}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Voice</label>
            <select
              value={localSettings.tts.voice}
              onChange={(e) => handleSettingChange('tts', 'voice', e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
            >
              {availableVoices.map(voice => (
                <option key={voice} value={voice}>{voice}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Speed: {localSettings.tts.speed}x
            </label>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.1"
              value={localSettings.tts.speed}
              onChange={(e) => handleSettingChange('tts', 'speed', parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="space-y-2">
        <button
          onClick={applySettings}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded transition-colors"
        >
          Apply Settings
        </button>
        <button
          onClick={resetSettings}
          className="w-full bg-gray-600 hover:bg-gray-700 text-white py-2 px-4 rounded transition-colors"
        >
          Reset to Defaults
        </button>
      </div>
    </div>
  );
};
```

## Phase 4: Implementation Priority

### Immediate Actions (Week 1):
1. **Create configuration system** - Implement `config.py` and settings manager
2. **Fix LLM token budget** - Update max_tokens to 2048+ tokens
3. **Remove hardcoded values** - Replace with config references

### Short-term (Week 2):
1. **Implement chat manager** - Proper message handling and history
2. **Create settings API endpoints** - Real-time configuration updates
3. **Build basic settings UI** - Essential controls for STT/LLM/TTS

### Medium-term (Week 3-4):
1. **Enhanced chat interface** - Proper UI with message bubbles
2. **Advanced settings panel** - Complete configuration interface
3. **Auto voice selection** - Language-based voice switching

### Code Cleanup Checklist:
- [ ] Replace hardcoded URLs with config references
- [ ] Remove unused imports and dead code
- [ ] Extract magic numbers into named constants
- [ ] Add proper error handling for service failures
- [ ] Implement graceful degradation when services are unavailable
- [ ] Add logging for debugging and monitoring

This plan addresses all your concerns while maintaining the ultra-low latency focus of Maestro. The modular approach allows you to implement features incrementally without breaking existing functionality.