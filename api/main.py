# Main FastAPI application
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import route modules
from suggestions import app as suggestions_app
from plagiarism import app as plagiarism_app
from database import db_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db_manager.connect()
    print("Database connected")
    yield
    # Shutdown
    await db_manager.close()
    print("Database disconnected")

# Create main application
app = FastAPI(
    title="Grammarly Clone API",
    description="Advanced writing assistant with AI-powered suggestions and plagiarism detection",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount sub-applications
app.mount("/suggestions", suggestions_app)
app.mount("/plagiarism", plagiarism_app)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Grammarly Clone API is running"}

# Document management endpoints
@app.post("/api/documents")
async def create_document(title: str, content: str, user_id: str = "default"):
    """Create a new document"""
    try:
        document_id = await db_manager.save_document(user_id, title, content)
        return {"document_id": document_id, "message": "Document created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{document_id}")
async def get_document(document_id: str):
    """Get a document by ID"""
    try:
        document = await db_manager.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/documents/{document_id}")
async def update_document(document_id: str, title: str, content: str, user_id: str = "default"):
    """Update a document"""
    try:
        await db_manager.save_document(user_id, title, content, document_id)
        return {"message": "Document updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    try:
        await db_manager.delete_document(document_id)
        return {"message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/documents")
async def get_user_documents(user_id: str, limit: int = 10):
    """Get user's documents"""
    try:
        documents = await db_manager.get_user_documents(user_id, limit)
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/statistics")
async def get_user_statistics(user_id: str, days: int = 7):
    """Get user writing statistics"""
    try:
        stats = await db_manager.get_user_statistics(user_id, days)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/trends")
async def get_writing_trends(user_id: str, days: int = 30):
    """Get writing trends over time"""
    try:
        trends = await db_manager.get_writing_trends(user_id, days)
        return {"trends": trends}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )