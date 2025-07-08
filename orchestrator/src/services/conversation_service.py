import logging
import ollama
from config import config
from services.base_service import BaseService, ServiceResult
from core.state_machine import Session

logger = logging.getLogger(__name__)

class ConversationService(BaseService):
    def __init__(self):
        self.ollama_url = config.OLLAMA_URL
        self.system_context = """You are a helpful voice assistant. You have speaker recognition capabilities and can remember users by their voice. Be conversational, helpful, and personable. Keep responses concise for voice interaction."""

    async def process(self, text: str, context: dict) -> ServiceResult:
        """
        Generate a response using the LLM, incorporating conversation history
        and agentic speaker greetings.
        """
        session: Session = context.get("session")
        if not session:
            return ServiceResult(success=False, error="No session context provided.")

        # Agentic greeting logic
        greeting = self._get_agentic_greeting(session)
        
        history = session.conversation_history
        prompt = self._build_prompt(text, history, greeting)
        
        try:
            client = ollama.AsyncClient(host=self.ollama_url)
            response = await client.generate(
                model=config.LLM_MODEL,
                prompt=prompt,
                stream=True,  # Enable streaming for ultra-low latency
                options={
                    "num_predict": config.LLM_MAX_TOKENS,
                    "temperature": config.LLM_TEMPERATURE,
                }
            )
            
            # Collect streaming response with early token logging
            full_response = ""
            token_count = 0
            start_time = time.time()
            
            async for chunk in response:
                if 'response' in chunk:
                    token_count += 1
                    full_response += chunk['response']
                    
                    # Log first token arrival for ultra-low latency tracking
                    if token_count == 1:
                        first_token_time = time.time() - start_time
                        logger.info(f"🚀 First LLM token arrived in {first_token_time:.3f}s")
                    
                    # TODO: Start TTS processing on first sentence boundary
                    # This would achieve true streaming pipeline
            
            full_response = full_response.strip()
            if not full_response:
                return ServiceResult(success=False, error="LLM generated an empty response.")
            
            # Add exchange to history
            session.conversation_history.append({"role": "user", "content": text})
            session.conversation_history.append({"role": "assistant", "content": full_response})
            
            return ServiceResult(success=True, data=full_response)
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return ServiceResult(success=False, error=f"LLM generation failed: {e}")

    def _get_agentic_greeting(self, session: Session) -> str:
        """Generate a personalized greeting based on speaker state."""
        if session.is_new_speaker:
            session.is_new_speaker = False # Ensure greeting is only used once
            return "Hello! I don't think we've met before. I've registered your voice. What should I call you?"
        
        # This part can be expanded with more context from memory
        return f"Welcome back, {session.speaker_name}!"

    def _build_prompt(self, text: str, history: list, greeting: str) -> str:
        # Prepend the greeting to the history for the LLM
        messages = [{"role": "assistant", "content": greeting}] if greeting else []
        for item in history:
            messages.append(item)
        messages.append({"role": "user", "content": text})
        
        # This is a simplified prompt builder. A more advanced version would use a proper chat template.
        prompt = self.system_context
        for msg in messages:
            prompt += f"\n{msg['role'].capitalize()}: {msg['content']}"
        prompt += "\nAssistant:"
        return prompt

    async def health_check(self) -> bool:
        return True

    def get_metrics(self) -> dict:
        return {}