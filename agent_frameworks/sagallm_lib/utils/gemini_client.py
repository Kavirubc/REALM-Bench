"""
Gemini client wrapper that mimics OpenAI's interface for compatibility with SagaLLM.
"""
import os
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


class GeminiClientWrapper:
    """
    Wrapper around LangChain's ChatGoogleGenerativeAI to mimic OpenAI's interface.
    This allows SagaLLM's ReactAgent to use Gemini instead of OpenAI.
    """
    
    def __init__(self, model: str = "gemini-flash-latest", temperature: float = 0.3):
        self._is_gemini = True
        self.model_name = model
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature
        )
    
    class Completions:
        """Mimics OpenAI's chat.completions interface"""
        
        def __init__(self, parent_client):
            self.parent_client = parent_client
        
        def create(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.3, max_tokens: int = 3000) -> str:
            """
            Create a chat completion using Gemini.
            
            Args:
                messages: List of message dicts with 'role' and 'content'
                model: Model name (ignored, uses client's model)
                temperature: Temperature setting
                max_tokens: Max tokens (ignored for Gemini)
            
            Returns:
                str: The response content
            """
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'system':
                    langchain_messages.append(SystemMessage(content=content))
                elif role == 'assistant':
                    langchain_messages.append(AIMessage(content=content))
                else:  # user
                    langchain_messages.append(HumanMessage(content=content))
            
            # Invoke the LLM
            try:
                response = self.parent_client.llm.invoke(langchain_messages)
                
                # Extract content (handle both string and AIMessage)
                if hasattr(response, 'content'):
                    content = response.content
                    # Handle list of content blocks (Gemini can return multiple)
                    if isinstance(content, list):
                        # Join all text content
                        text_parts = []
                        for part in content:
                            if hasattr(part, 'text'):
                                text_parts.append(part.text)
                            elif isinstance(part, str):
                                text_parts.append(part)
                        return '\n'.join(text_parts) if text_parts else str(content)
                    return str(content)
                return str(response)
            except Exception as e:
                # If there's an error, return a string representation
                return f"Error: {str(e)}"
    
    class Chat:
        """Mimics OpenAI's chat interface"""
        def __init__(self, parent_client):
            self.parent_client = parent_client
            self.completions = parent_client.Completions(parent_client)
    
    @property
    def chat(self):
        return self.Chat(self)

