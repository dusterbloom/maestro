import logging
import ollama
from orchestrator.src.config import config
from orchestrator.src.services.base_service import BaseService, ServiceResult
from orchestrator.src.core.state_machine import Session

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
                stream=False, # For simplicity in this refactoring phase
                options={
                    "num_predict": config.LLM_MAX_TOKENS,
                    "temperature": config.LLM_TEMPERATURE,
                }
            )
            full_response = response.get('response', '').strip()
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