import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client

# === FastAPI app ===
app = FastAPI(title="RAGPsy API Wrapper", version="1.0.0")

# Add CORS middleware for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gradio client
try:
    client = Client("zeeshanali66/RagPsychologistapi")
    print("✅ Connected to Hugging Face Space")
except Exception as e:
    print(f"❌ Error connecting to Hugging Face Space: {e}")
    client = None

# === API Models ===
class ChatRequest(BaseModel):
    question: str

# === API Endpoints ===
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="Hugging Face Space not available")
    
    try:
        result = client.predict(
            message=request.question,
            api_name="/predict"
        )
        return JSONResponse(content={"answer": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
def root():
    return {"message": "RAGPsy API Wrapper is running"}

@app.get("/health")
def health_check():
    if client is None:
        return {"status": "error", "message": "Hugging Face Space not connected"}
    return {"status": "ok", "message": "Connected to Hugging Face Space"}

# === Run the app ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
