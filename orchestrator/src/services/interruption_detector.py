# services/interruption_detector.py
"""
Intelligent interruption detection for voice conversations.
Analyzes user input to detect interruption signals and adapt response generation.
"""

import re
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger

@dataclass
class InterruptionAnalysis:
    is_interruption: bool
    interruption_confidence: float
    wants_brevity: bool
    should_stop_current: bool
    suggested_max_tokens: int
    suggested_response_type: str  # "acknowledge", "brief", "normal", "clarifying"

class InterruptionDetector:
    """
    Detects user interruption intent and provides response adaptation recommendations.
    """
    
    # Interruption signal phrases (ordered by strength)
    STRONG_INTERRUPTION_PHRASES = [
        "stop", "enough", "okay thanks", "that's good", "got it",
        "never mind", "forget it", "no no", "wait wait"
    ]
    
    MEDIUM_INTERRUPTION_PHRASES = [
        "please", "hold on", "wait", "okay", "alright", 
        "no", "yes", "sure", "thanks"
    ]
    
    WEAK_INTERRUPTION_PHRASES = [
        "um", "uh", "well", "so", "actually", "but"
    ]
    
    # Brevity request indicators
    BREVITY_TRIGGERS = [
        "briefly", "quick", "short", "summarize", "tldr", "tl;dr",
        "in summary", "bottom line", "just tell me", "simple answer",
        "yes or no", "one word", "short version"
    ]
    
    # Continuation signals (reduces interruption likelihood)
    CONTINUATION_SIGNALS = [
        "tell me more", "go on", "continue", "keep going", 
        "what else", "and then", "more details", "explain"
    ]
    
    def __init__(self):
        self.user_interruption_history = {}  # Track per-user patterns
        
    def analyze_input(self, 
                     transcript: str, 
                     user_id: str, 
                     conversation_context: Dict,
                     audio_metadata: Optional[Dict] = None) -> InterruptionAnalysis:
        """
        Analyze user input for interruption signals and response adaptation needs.
        
        Args:
            transcript: User's spoken input
            user_id: Unique user identifier
            conversation_context: Recent conversation history
            audio_metadata: Optional audio characteristics (volume, speed, etc.)
            
        Returns:
            InterruptionAnalysis with recommendations
        """
        
        transcript_lower = transcript.lower().strip()
        words = transcript_lower.split()
        
        # Initialize analysis scores
        interruption_score = 0.0
        brevity_score = 0.0
        
        # 1. Phrase-based analysis
        phrase_score = self._analyze_phrases(transcript_lower)
        interruption_score += phrase_score
        
        # 2. Length-based analysis (short inputs often indicate interruptions)
        length_score = self._analyze_length(words)
        interruption_score += length_score
        
        # 3. Repetition analysis (emphasis often indicates interruption)
        repetition_score = self._analyze_repetition(words)
        interruption_score += repetition_score
        
        # 4. Timing analysis (if available)
        timing_score = self._analyze_timing(conversation_context, audio_metadata)
        interruption_score += timing_score
        
        # 5. Brevity request detection
        brevity_score = self._detect_brevity_requests(transcript_lower)
        
        # 6. User pattern analysis
        pattern_adjustment = self._analyze_user_patterns(user_id, transcript_lower)
        interruption_score += pattern_adjustment
        
        # 7. Context-based adjustments
        context_adjustment = self._analyze_context(conversation_context)
        interruption_score += context_adjustment
        
        # Normalize scores
        interruption_score = max(0.0, min(1.0, interruption_score))
        brevity_score = max(0.0, min(1.0, brevity_score))
        
        # Determine interruption classification
        is_interruption = interruption_score > 0.6
        should_stop_current = interruption_score > 0.8
        wants_brevity = brevity_score > 0.4 or interruption_score > 0.7
        
        # Suggest response parameters
        suggested_max_tokens, response_type = self._suggest_response_params(
            interruption_score, brevity_score, conversation_context
        )
        
        # Update user patterns
        self._update_user_patterns(user_id, interruption_score, transcript_lower)
        
        # Log analysis for debugging
        logger.debug(f"Interruption analysis for '{transcript[:30]}...': "
                    f"score={interruption_score:.2f}, brevity={brevity_score:.2f}, "
                    f"type={response_type}")
        
        return InterruptionAnalysis(
            is_interruption=is_interruption,
            interruption_confidence=interruption_score,
            wants_brevity=wants_brevity,
            should_stop_current=should_stop_current,
            suggested_max_tokens=suggested_max_tokens,
            suggested_response_type=response_type
        )
    
    def _analyze_phrases(self, transcript_lower: str) -> float:
        """Analyze for interruption phrases."""
        score = 0.0
        
        # Check for strong interruption phrases
        for phrase in self.STRONG_INTERRUPTION_PHRASES:
            if phrase in transcript_lower:
                score += 0.4
                
        # Check for medium interruption phrases
        for phrase in self.MEDIUM_INTERRUPTION_PHRASES:
            if phrase in transcript_lower:
                score += 0.25
                
        # Check for weak interruption phrases
        for phrase in self.WEAK_INTERRUPTION_PHRASES:
            if phrase in transcript_lower:
                score += 0.1
                
        # Reduce score for continuation signals
        for phrase in self.CONTINUATION_SIGNALS:
            if phrase in transcript_lower:
                score -= 0.3
                
        return score
    
    def _analyze_length(self, words: List[str]) -> float:
        """Analyze input length for interruption likelihood."""
        word_count = len(words)
        
        if word_count <= 1:
            return 0.4  # Very short, likely interruption
        elif word_count <= 3:
            return 0.3  # Short, probably interruption
        elif word_count <= 6:
            return 0.1  # Medium, might be interruption
        else:
            return 0.0  # Long, unlikely interruption
    
    def _analyze_repetition(self, words: List[str]) -> float:
        """Analyze word repetition for emphasis."""
        if len(words) < 2:
            return 0.0
            
        unique_words = set(words)
        repetition_ratio = 1.0 - (len(unique_words) / len(words))
        
        # High repetition suggests emphasis/interruption
        return repetition_ratio * 0.3
    
    def _analyze_timing(self, conversation_context: Dict, audio_metadata: Optional[Dict]) -> float:
        """Analyze timing context for interruption likelihood."""
        score = 0.0
        
        # Check if response was interrupted mid-generation
        last_response_time = conversation_context.get('last_response_time', 0)
        current_time = time.time()
        
        if last_response_time and (current_time - last_response_time) < 3.0:
            score += 0.3  # Quick response suggests interruption
            
        # Audio metadata analysis (if available)
        if audio_metadata:
            volume = audio_metadata.get('average_volume', 0)
            if volume > 0.8:  # Loud input
                score += 0.1
                
            speech_rate = audio_metadata.get('speech_rate', 1.0)
            if speech_rate > 1.5:  # Fast speech
                score += 0.1
                
        return score
    
    def _detect_brevity_requests(self, transcript_lower: str) -> float:
        """Detect requests for brief responses."""
        score = 0.0
        
        for trigger in self.BREVITY_TRIGGERS:
            if trigger in transcript_lower:
                score += 0.4
                
        # Question word patterns that suggest brief answers
        brief_patterns = [
            r'\b(is|are|can|will|do|does|did)\b.*\?',  # Yes/no questions
            r'\bhow (much|many|long)\b',  # Quantity questions
            r'\bwhat.*(time|date|year)\b',  # Time questions
        ]
        
        for pattern in brief_patterns:
            if re.search(pattern, transcript_lower):
                score += 0.2
                
        return score
    
    def _analyze_user_patterns(self, user_id: str, transcript_lower: str) -> float:
        """Analyze user-specific interruption patterns."""
        if user_id not in self.user_interruption_history:
            return 0.0
            
        user_data = self.user_interruption_history[user_id]
        
        # Check if user frequently interrupts
        interruption_frequency = user_data.get('interruption_rate', 0.0)
        if interruption_frequency > 0.7:
            return 0.1  # User tends to interrupt
        elif interruption_frequency < 0.3:
            return -0.1  # User rarely interrupts
            
        # Check for user-specific interruption phrases
        user_phrases = user_data.get('common_interruption_phrases', [])
        for phrase in user_phrases:
            if phrase in transcript_lower:
                return 0.2
                
        return 0.0
    
    def _analyze_context(self, conversation_context: Dict) -> float:
        """Analyze conversation context for interruption likelihood."""
        score = 0.0
        
        # Recent interruption attempts
        recent_interruptions = conversation_context.get('recent_interruption_count', 0)
        if recent_interruptions > 2:
            score += 0.2  # User is getting impatient
            
        # Long AI responses recently
        recent_response_lengths = conversation_context.get('recent_response_lengths', [])
        if recent_response_lengths:
            avg_length = sum(recent_response_lengths) / len(recent_response_lengths)
            if avg_length > 100:  # Long responses
                score += 0.15
                
        return score
    
    def _suggest_response_params(self, 
                                interruption_score: float, 
                                brevity_score: float, 
                                conversation_context: Dict) -> tuple[int, str]:
        """Suggest response parameters based on analysis."""
        
        if interruption_score > 0.8:
            # Strong interruption - acknowledge briefly
            return 20, "acknowledge"
        elif interruption_score > 0.6 or brevity_score > 0.6:
            # Medium interruption or brevity request
            return 50, "brief"
        elif brevity_score > 0.4:
            # Some brevity preference
            return 80, "concise"
        elif interruption_score > 0.7:
            # Ask for clarification
            return 30, "clarifying"
        else:
            # Normal response
            return 150, "normal"
    
    def _update_user_patterns(self, user_id: str, interruption_score: float, transcript: str):
        """Update user-specific interruption patterns."""
        if user_id not in self.user_interruption_history:
            self.user_interruption_history[user_id] = {
                'interruption_rate': 0.0,
                'total_inputs': 0,
                'interruption_count': 0,
                'common_interruption_phrases': []
            }
            
        user_data = self.user_interruption_history[user_id]
        user_data['total_inputs'] += 1
        
        if interruption_score > 0.6:
            user_data['interruption_count'] += 1
            
            # Track common interruption phrases
            words = transcript.split()[:3]  # First few words
            phrase = ' '.join(words)
            if phrase not in user_data['common_interruption_phrases']:
                if len(user_data['common_interruption_phrases']) < 10:
                    user_data['common_interruption_phrases'].append(phrase)
        
        # Update interruption rate
        user_data['interruption_rate'] = (
            user_data['interruption_count'] / user_data['total_inputs']
        )
        
        # Keep only recent history (last 50 interactions)
        if user_data['total_inputs'] > 50:
            user_data['interruption_count'] = int(user_data['interruption_count'] * 0.9)
            user_data['total_inputs'] = 45
    
    def get_user_interruption_stats(self, user_id: str) -> Dict:
        """Get interruption statistics for a user."""
        return self.user_interruption_history.get(user_id, {
            'interruption_rate': 0.0,
            'total_inputs': 0,
            'interruption_count': 0,
            'common_interruption_phrases': []
        })
