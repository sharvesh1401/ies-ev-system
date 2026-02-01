import httpx
from typing import Optional, Dict, Any
from fastapi import HTTPException
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class AIService:
    """
    Service for interacting with DeepSeek AI API.
    """
    
    def __init__(self):
        self.api_key = settings.DEEPSEEK_API_KEY
        self.base_url = "https://api.deepseek.com/v1"  # Assumed standard v1, check if needed
        # DeepSeek often uses OpenAI compatible endpoints
        self.model = settings.DEEPSEEK_MODEL

    async def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a response from DeepSeek AI.
        
        Args:
            prompt: The user input prompt
            
        Returns:
            Dict containing the model response
            
        Raises:
            HTTPException: If the API call fails or API key is missing
        """
        if not self.api_key:
            # For Phase 0, we might want to fail gracefully or return a mock if not configured?
            # Prompt says "DeepSeek AI integration works (when API key is provided)"
            raise HTTPException(
                status_code=503, 
                detail="DeepSeek API key not configured. Please set DEEPSEEK_API_KEY in .env"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for the IES_EV (Intelligent Energy System for Electric Vehicles) project."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Using chat/completions endpoint (OpenAI compatible)
                response = await client.post(
                    f"https://api.deepseek.com/chat/completions",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code != 200:
                    logger.error(f"DeepSeek API Error: {response.text}")
                    raise HTTPException(
                        status_code=response.status_code, 
                        detail=f"AI Provider Error: {response.text}"
                    )
                
                return response.json()
                
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise HTTPException(status_code=503, detail=f"AI Service execution failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal AI Service Error")

ai_service = AIService()
