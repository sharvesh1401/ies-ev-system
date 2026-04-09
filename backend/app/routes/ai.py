from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from typing import Dict, Any
from app.services import ai_service
from app.limiter import limiter

router = APIRouter()

class ChatRequest(BaseModel):
    """Request model for AI chat.

    message is capped at 2000 characters to prevent prompt-injection via
    extremely large payloads and to stay within typical LLM context budgets.
    """
    message: str = Field(..., min_length=1, max_length=2000)

    model_config = {"extra": "forbid"}

class ChatResponse(BaseModel):
    """
    Response model for AI chat
    """
    response: str
    timestamp: str = "now" # In real app use datetime
    
@router.post("/chat", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
@limiter.limit("20/minute")
async def chat_with_ai(request: Request, body: ChatRequest) -> Dict[str, Any]:
    """
    Send a message to the AI and get a response.
    
    Args:
        request: The chat request containing the message
        
    Returns:
        The AI response
        
    Raises:
        HTTPException: If the AI service fails
    """
    # Simply delegate to the service
    result = await ai_service.generate_response(body.message)
    
    # Extract the actual content from DeepSeek/OpenAI format
    # Response format: { "choices": [ { "message": { "content": "..." } } ] }
    try:
        content = result.get("choices", [])[0].get("message", {}).get("content", "")
    except (IndexError, AttributeError):
        content = "Error parsing AI response"
        
    return {
        "response": content,
        "raw": result
    }
