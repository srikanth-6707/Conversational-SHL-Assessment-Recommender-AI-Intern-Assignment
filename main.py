from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from catalog import SHLCatalog
from agent import SHLAgent, ChatRequest, ChatResponse

app = FastAPI(title="SHL Conversational Assessment Recommender")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

catalog = SHLCatalog()
agent = SHLAgent(catalog)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    return agent.chat(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
