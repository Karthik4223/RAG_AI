from typing import List, Dict, Any, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from app.core.config import settings
from app.core.logging import logger
import uuid

class LLMService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=settings.GOOGLE_API_KEY,
            model=settings.GEMINI_MODEL,
            temperature=0,
            convert_system_message_to_human=True # Useful for some Gemini models
        )
        
    def generate_answer(self, query: str, context_chunks: List[Tuple[Any, float]]) -> Dict[str, Any]:
        """Generates a grounded answer based on the provided context."""
        
        # Build context string
        context_text = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" 
            for doc, score in context_chunks
        ])
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "Context:\n{context}\n\nQuestion: {query}")
        ])
        
        chain = prompt_template | self.llm
        
        logger.info(f"Generating answer for query: {query[:50]}...")
        response = chain.invoke({
            "context": context_text,
            "query": query
        })
        
        return {
            "answer": response.content,
            "query_id": str(uuid.uuid4())
        }

    def _get_system_prompt(self) -> str:
        return """
        You are a senior AI assistant for a professional organization. 
        Your task is to provide accurate, grounded answers based ONLY on the provided context.
        
        Guidelines:
        1. If the answer is not in the context, clearly state that you don't have enough information. Do not hallucinate.
        2. Always cite your sources briefly if multiple sources are provided.
        3. Be professional, concise, and helpful.
        4. If the context contains conflicting information, point it out.
        5. Use bullet points for lists to improve readability.
        """
