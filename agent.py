from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: Optional[str] = "K"

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = []
    end_of_conversation: bool = False

class SHLAgent:
    def __init__(self, catalog):
        self.catalog = catalog
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Works with Groq/Gemini too

    def _should_recommend(self, history: List[dict]) -> bool:
        user_msgs = [m["content"] for m in history if m["role"] == "user"]
        if len(user_msgs) < 2:
            return False
        # Simple heuristic + LLM can be used for refinement
        return any(word in " ".join(user_msgs[-3:]).lower() for word in ["role", "developer", "engineer", "hiring", "position"])

    def chat(self, req: ChatRequest) -> ChatResponse:
        history = [{"role": m.role, "content": m.content} for m in req.messages]
        last_user = history[-1]["content"] if history and history[-1]["role"] == "user" else ""

        # Retrieve
        results = self.catalog.search(last_user + " " + " ".join([m["content"] for m in history if m["role"] == "user"]), 12)

        # System prompt for grounded behavior
        system_prompt = """You are an SHL Assessment Expert. 
        Use only the provided catalog data. 
        Clarify if vague. Recommend 1-10 relevant Individual Test Solutions when enough context. 
        Support refinement and comparison. Never recommend outside catalog or give hiring advice."""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)

        # Add catalog context
        if results:
            context = "\nCatalog matches:\n" + "\n".join([f"- {a['name']}: {a.get('description','')[:150]}" for a in results[:8]])
            messages[0]["content"] += context

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=600
        )
        reply = response.choices[0].message.content.strip()

        # Build recommendations
        recommendations = []
        if self._should_recommend(history) and results:
            for ass in results[:8]:
                recommendations.append(Recommendation(
                    name=ass["name"],
                    url=ass.get("url", "https://www.shl.com/"),
                    test_type=ass.get("test_type", "K")
                ))

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=len(recommendations) > 0
        )
