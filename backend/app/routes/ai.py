from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any
from app.services import ai_service

router = APIRouter()

class ChatRequest(BaseModel):
    """
    Request model for AI chat
    """
    message: str

class ChatResponse(BaseModel):
    """
    Response model for AI chat
    """
    response: str
    timestamp: str = "now" # In real app use datetime
    
@router.post("/chat", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def chat_with_ai(request: ChatRequest) -> Dict[str, Any]:
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
    result = await ai_service.generate_response(request.message)
    
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
