
import logging
import ollama
from orchestrator.src.config import config
from orchestrator.src.services.base_service import BaseService, ServiceResult

logger = logging.getLogger(__name__)

class ConversationService(BaseService):
    def __init__(self):
        self.ollama_url = config.OLLAMA_URL
        self.system_context = """You are a helpful voice assistant. You have speaker recognition capabilities and can remember users by their voice.
- When you recognize a returning user, greet them warmly by name.
- When you meet someone new, ask for their name and remember it.
- Be conversational, helpful, and personable.
- Keep responses concise for voice interaction."""

    async def process(self, text: str, context: dict = None) -> ServiceResult:
        """Generate a response using the LLM."""
        history = context.get("conversation_history", [])
        prompt = self._build_prompt(text, history)
        
        try:
            client = ollama.AsyncClient(host=self.ollama_url)
            response = await client.generate(
                model=config.LLM_MODEL,
                prompt=prompt,
                stream=False,
                options={
                    "num_predict": config.LLM_MAX_TOKENS,
                    "temperature": config.LLM_TEMPERATURE,
                    "top_p": 0.8,
                    "num_ctx": 2048,
                }
            )
            full_response = response.get('response', '').strip()
            if not full_response:
                return ServiceResult(success=False, error="LLM generated an empty response.")
            
            return ServiceResult(success=True, data=full_response)
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return ServiceResult(success=False, error=f"LLM generation failed: {e}")

    def _build_prompt(self, text: str, history: list) -> str:
        # This can be made more sophisticated
        context_str = "\n".join([f"User: {item['user']}\nAssistant: {item['assistant']}" for item in history])
        return f"{self.system_context}\n\n{context_str}\n\nUser: {text}\nAssistant:"

    async def health_check(self) -> bool:
        # In a real scenario, you'd ping the Ollama service
        return True

    def get_metrics(self) -> dict:
        return {}
