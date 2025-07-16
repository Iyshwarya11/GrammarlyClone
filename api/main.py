# Main FastAPI application with integrated suggestions and plagiarism APIs
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, Dict
import re
import random
import hashlib
import asyncio
from datetime import datetime

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

# Pydantic models for suggestions
class TextInput(BaseModel):
    content: str
    goal: Optional[str] = "clarity"

class Suggestion(BaseModel):
    id: str
    type: str
    text: str
    suggestion: str
    explanation: str
    position: dict

class SuggestionResponse(BaseModel):
    suggestions: List[Suggestion]
    stats: dict

# Pydantic models for plagiarism
class PlagiarismRequest(BaseModel):
    content: str

class PlagiarismResult(BaseModel):
    id: str
    source: str
    similarity: float
    matchedText: str
    url: str
    type: str

class PlagiarismResponse(BaseModel):
    overallScore: float
    results: List[PlagiarismResult]
    processingTime: float

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

# SUGGESTIONS API ENDPOINTS
@app.post("/api/suggestions", response_model=SuggestionResponse)
async def get_suggestions(text_input: TextInput):
    """
    Get writing suggestions using rule-based analysis (no external models required)
    """
    try:
        content = text_input.content
        goal = text_input.goal
        
        # Get suggestions using rule-based approach
        suggestions = []
        suggestions.extend(check_grammar_rules(content))
        suggestions.extend(check_clarity_issues(content))
        suggestions.extend(check_tone_issues(content))
        suggestions.extend(check_engagement_issues(content))
        
        # Calculate statistics
        stats = calculate_stats(content)
        
        return SuggestionResponse(
            suggestions=suggestions,
            stats=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def check_grammar_rules(text: str) -> List[Suggestion]:
    """Check grammar using rule-based approach"""
    suggestions = []
    
    # Common grammar patterns
    patterns = [
        {
            "pattern": r"\bthere\s+is\s+\w+\s+\w+s\b",
            "type": "grammar",
            "suggestion": "Use 'there are' with plural nouns",
            "explanation": "Subject-verb agreement error"
        },
        {
            "pattern": r"\bits\s+\w+ing\b",
            "type": "grammar", 
            "suggestion": "Consider 'it's' (it is) instead of 'its' (possessive)",
            "explanation": "Common its/it's confusion"
        },
        {
            "pattern": r"\byour\s+\w+ing\b",
            "type": "grammar",
            "suggestion": "Consider 'you're' (you are) instead of 'your' (possessive)",
            "explanation": "Common your/you're confusion"
        },
        {
            "pattern": r"\b(very|really|quite|extremely)\s+\w+",
            "type": "clarity",
            "suggestion": "Consider using a stronger adjective instead of intensifiers",
            "explanation": "Intensifiers can weaken your writing"
        }
    ]
    
    for i, pattern_info in enumerate(patterns):
        matches = re.finditer(pattern_info["pattern"], text, re.IGNORECASE)
        for match in matches:
            suggestions.append(Suggestion(
                id=f"grammar_{i}_{match.start()}",
                type=pattern_info["type"],
                text=match.group(),
                suggestion=pattern_info["suggestion"],
                explanation=pattern_info["explanation"],
                position={"start": match.start(), "end": match.end()}
            ))
    
    return suggestions

def check_clarity_issues(text: str) -> List[Suggestion]:
    """Check for clarity issues"""
    suggestions = []
    
    # Long sentences
    sentences = re.split(r'[.!?]+', text)
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        if len(words) > 25:
            start_pos = text.find(sentence.strip())
            if start_pos != -1:
                suggestions.append(Suggestion(
                    id=f"clarity_long_{i}",
                    type="clarity",
                    text=sentence.strip()[:50] + "...",
                    suggestion="Consider breaking this into shorter sentences",
                    explanation="Long sentences can be hard to follow",
                    position={"start": start_pos, "end": start_pos + len(sentence)}
                ))
    
    # Passive voice detection
    passive_patterns = [
        r"\b(was|were|is|are|been|being)\s+\w+ed\b",
        r"\b(was|were|is|are|been|being)\s+\w+en\b"
    ]
    
    for pattern in passive_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            suggestions.append(Suggestion(
                id=f"clarity_passive_{match.start()}",
                type="clarity",
                text=match.group(),
                suggestion="Consider using active voice",
                explanation="Active voice is more direct and engaging",
                position={"start": match.start(), "end": match.end()}
            ))
    
    return suggestions

def check_tone_issues(text: str) -> List[Suggestion]:
    """Check for tone issues"""
    suggestions = []
    
    # Weak words
    weak_words = ["maybe", "perhaps", "possibly", "might", "could", "sort of", "kind of"]
    
    for weak_word in weak_words:
        pattern = r'\b' + re.escape(weak_word) + r'\b'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            suggestions.append(Suggestion(
                id=f"tone_weak_{match.start()}",
                type="tone",
                text=match.group(),
                suggestion="Consider using more confident language",
                explanation="Weak words can undermine your authority",
                position={"start": match.start(), "end": match.end()}
            ))
    
    return suggestions

def check_engagement_issues(text: str) -> List[Suggestion]:
    """Check for engagement issues"""
    suggestions = []
    
    # Repetitive sentence starters
    sentences = re.split(r'[.!?]+', text)
    starters = {}
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            first_words = ' '.join(sentence.split()[:2]).lower()
            if first_words in starters:
                starters[first_words] += 1
            else:
                starters[first_words] = 1
    
    for starter, count in starters.items():
        if count > 2:
            suggestions.append(Suggestion(
                id=f"engagement_repetitive_{starter}",
                type="engagement",
                text=f"Sentences starting with '{starter}'",
                suggestion="Vary your sentence beginnings",
                explanation="Repetitive sentence starters can make writing monotonous",
                position={"start": 0, "end": 50}
            ))
    
    return suggestions

def calculate_stats(text: str) -> dict:
    """Calculate text statistics"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    paragraphs = text.split('\n\n')
    
    # Simple readability score based on sentence and word length
    avg_sentence_length = len(words) / max(1, len([s for s in sentences if s.strip()]))
    avg_word_length = sum(len(word) for word in words) / max(1, len(words))
    
    # Simple readability formula (higher is better)
    readability_score = max(0, min(100, 100 - (avg_sentence_length * 2) - (avg_word_length * 5)))
    
    return {
        "wordCount": len(words),
        "characters": len(text),
        "sentences": len([s for s in sentences if s.strip()]),
        "paragraphs": len([p for p in paragraphs if p.strip()]),
        "readingTime": max(1, len(words) // 250),
        "readabilityScore": int(readability_score)
    }

# PLAGIARISM API ENDPOINTS
@app.post("/api/plagiarism/check", response_model=PlagiarismResponse)
async def check_plagiarism(request: PlagiarismRequest):
    """
    Check content for plagiarism using simulated analysis
    """
    start_time = datetime.now()
    
    try:
        content = request.content
        
        # Simulate processing time
        await asyncio.sleep(2)
        
        # Generate mock results based on content analysis
        results = await generate_mock_results(content)
        
        # Calculate overall score
        total_similarity = sum(result.similarity for result in results)
        overall_score = max(0, 100 - total_similarity)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PlagiarismResponse(
            overallScore=overall_score,
            results=results,
            processingTime=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate_mock_results(content: str) -> List[PlagiarismResult]:
    """Generate realistic mock plagiarism results"""
    results = []
    
    # Split content into sentences for analysis
    sentences = re.split(r'[.!?]+', content)
    meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not meaningful_sentences:
        return results
    
    # Mock sources
    sources = [
        {"name": "Wikipedia", "type": "web", "domain": "wikipedia.org"},
        {"name": "Academic Paper - ResearchGate", "type": "academic", "domain": "researchgate.net"},
        {"name": "Journal Article - JSTOR", "type": "publication", "domain": "jstor.org"},
        {"name": "News Article - BBC", "type": "web", "domain": "bbc.com"},
        {"name": "Blog Post - Medium", "type": "web", "domain": "medium.com"},
        {"name": "IEEE Paper", "type": "academic", "domain": "ieee.org"},
        {"name": "Nature Journal", "type": "publication", "domain": "nature.com"},
        {"name": "Educational Resource", "type": "web", "domain": "edu"}
    ]
    
    # Generate 1-4 results based on content length
    num_results = min(4, max(1, len(meaningful_sentences) // 3))
    
    for i in range(num_results):
        source = random.choice(sources)
        sentence = random.choice(meaningful_sentences)
        
        # Generate similarity score (higher for common phrases)
        common_words = ["the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our"]
        words = sentence.lower().split()
        common_count = sum(1 for word in words if word in common_words)
        
        # Base similarity on common words and sentence length
        base_similarity = min(25, (common_count / len(words)) * 100) if words else 0
        similarity = max(3, base_similarity + random.uniform(-5, 10))
        
        results.append(PlagiarismResult(
            id=f"result_{i}",
            source=source["name"],
            similarity=round(similarity, 1),
            matchedText=sentence[:100] + "..." if len(sentence) > 100 else sentence,
            url=f"https://{source['domain']}/article/{i+1}",
            type=source["type"]
        ))
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )