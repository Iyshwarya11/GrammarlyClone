# Main FastAPI application (Simplified)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from datetime import datetime
from typing import List, Dict, Optional

# Simple in-memory storage (replace with MongoDB in production)
documents_db = {}
suggestions_db = {}
plagiarism_db = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("API Server starting up...")
    yield
    # Shutdown
    print("API Server shutting down...")

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

# Import and include routers
from suggestions import app as suggestions_app
from plagiarism import app as plagiarism_app

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
        document_id = f"doc_{len(documents_db) + 1}_{int(datetime.now().timestamp())}"
        document = {
            "id": document_id,
            "user_id": user_id,
            "title": title,
            "content": content,
            "word_count": len(content.split()),
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }
        documents_db[document_id] = document
        return {"document_id": document_id, "message": "Document created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{document_id}")
async def get_document(document_id: str):
    """Get a document by ID"""
    try:
        if document_id not in documents_db:
            raise HTTPException(status_code=404, detail="Document not found")
        return documents_db[document_id]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/documents/{document_id}")
async def update_document(document_id: str, title: str, content: str, user_id: str = "default"):
    """Update a document"""
    try:
        if document_id not in documents_db:
            raise HTTPException(status_code=404, detail="Document not found")
        
        documents_db[document_id].update({
            "title": title,
            "content": content,
            "word_count": len(content.split()),
            "last_modified": datetime.now().isoformat()
        })
        return {"message": "Document updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    try:
        if document_id not in documents_db:
            raise HTTPException(status_code=404, detail="Document not found")
        del documents_db[document_id]
        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/documents")
async def get_user_documents(user_id: str, limit: int = 10):
    """Get user's documents"""
    try:
        user_docs = [doc for doc in documents_db.values() if doc["user_id"] == user_id]
        user_docs.sort(key=lambda x: x["last_modified"], reverse=True)
        return {"documents": user_docs[:limit]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/statistics")
async def get_user_statistics(user_id: str, days: int = 7):
    """Get user writing statistics"""
    try:
        user_docs = [doc for doc in documents_db.values() if doc["user_id"] == user_id]
        
        total_words = sum(doc.get("word_count", 0) for doc in user_docs)
        total_documents = len(user_docs)
        
        return {
            "total_words": total_words,
            "total_documents": total_documents,
            "average_words_per_document": total_words // max(1, total_documents),
            "period_days": days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/trends")
async def get_writing_trends(user_id: str, days: int = 30):
    """Get writing trends over time"""
    try:
        # Mock trend data
        trends = [
            {"date": "2024-01-01", "words": 1200, "documents": 2},
            {"date": "2024-01-02", "words": 1500, "documents": 3},
            {"date": "2024-01-03", "words": 900, "documents": 1},
        ]
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