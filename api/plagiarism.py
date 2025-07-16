# FastAPI Backend - Plagiarism Checker API (Simplified)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import hashlib
import re
from datetime import datetime
import asyncio
import random

app = FastAPI()

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
    uvicorn.run(app, host="0.0.0.0", port=8001)