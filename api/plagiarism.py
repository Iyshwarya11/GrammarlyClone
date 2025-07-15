# FastAPI Backend - Plagiarism Checker API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import hashlib
import re
from datetime import datetime
import asyncio
import aiohttp

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
    Check content for plagiarism using multiple sources
    """
    start_time = datetime.now()
    
    try:
        content = request.content
        
        # Perform multiple checks simultaneously
        results = await asyncio.gather(
            check_web_sources(content),
            check_academic_sources(content),
            check_publication_sources(content),
            return_exceptions=True
        )
        
        # Combine results
        all_results = []
        for result_list in results:
            if isinstance(result_list, list):
                all_results.extend(result_list)
        
        # Calculate overall score
        total_similarity = sum(result.similarity for result in all_results)
        overall_score = max(0, 100 - total_similarity)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PlagiarismResponse(
            overallScore=overall_score,
            results=all_results,
            processingTime=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def check_web_sources(content: str) -> List[PlagiarismResult]:
    """Check against web sources"""
    results = []
    
    # Split content into chunks for better matching
    chunks = split_into_chunks(content, 50)
    
    for i, chunk in enumerate(chunks):
        # Simulate web search (in real implementation, use search APIs)
        similarity = await simulate_web_search(chunk)
        
        if similarity > 5:  # Only include matches above threshold
            results.append(PlagiarismResult(
                id=f"web_{i}",
                source=f"Web Source {i+1}",
                similarity=similarity,
                matchedText=chunk,
                url=f"https://example.com/source{i+1}",
                type="web"
            ))
    
    return results

async def check_academic_sources(content: str) -> List[PlagiarismResult]:
    """Check against academic sources"""
    results = []
    
    # Simulate academic database search
    chunks = split_into_chunks(content, 100)
    
    for i, chunk in enumerate(chunks):
        similarity = await simulate_academic_search(chunk)
        
        if similarity > 3:
            results.append(PlagiarismResult(
                id=f"academic_{i}",
                source=f"Academic Paper {i+1}",
                similarity=similarity,
                matchedText=chunk,
                url=f"https://scholar.google.com/paper{i+1}",
                type="academic"
            ))
    
    return results

async def check_publication_sources(content: str) -> List[PlagiarismResult]:
    """Check against publication sources"""
    results = []
    
    # Simulate publication database search
    chunks = split_into_chunks(content, 75)
    
    for i, chunk in enumerate(chunks):
        similarity = await simulate_publication_search(chunk)
        
        if similarity > 4:
            results.append(PlagiarismResult(
                id=f"publication_{i}",
                source=f"Publication {i+1}",
                similarity=similarity,
                matchedText=chunk,
                url=f"https://publication.com/article{i+1}",
                type="publication"
            ))
    
    return results

def split_into_chunks(text: str, words_per_chunk: int) -> List[str]:
    """Split text into chunks of specified word count"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), words_per_chunk):
        chunk = ' '.join(words[i:i + words_per_chunk])
        if len(chunk.strip()) > 20:  # Only include meaningful chunks
            chunks.append(chunk)
    
    return chunks

async def simulate_web_search(text: str) -> float:
    """Simulate web search API call"""
    # In real implementation, this would call actual search APIs
    await asyncio.sleep(0.1)  # Simulate API delay
    
    # Simple similarity calculation based on common phrases
    common_phrases = ["the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "its", "may", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "man", "men", "run", "say", "she", "too", "use"]
    
    words = text.lower().split()
    common_count = sum(1 for word in words if word in common_phrases)
    
    # Return similarity percentage (0-20%)
    return min(20, (common_count / len(words)) * 100) if words else 0

async def simulate_academic_search(text: str) -> float:
    """Simulate academic database search"""
    await asyncio.sleep(0.15)  # Simulate API delay
    
    # Academic texts might have higher similarity for technical terms
    technical_terms = ["analysis", "research", "study", "method", "result", "conclusion", "data", "significant", "hypothesis", "theory", "model", "framework", "approach", "methodology", "findings", "evidence", "correlation", "statistical", "experimental", "validation"]
    
    words = text.lower().split()
    technical_count = sum(1 for word in words if word in technical_terms)
    
    return min(15, (technical_count / len(words)) * 100) if words else 0

async def simulate_publication_search(text: str) -> float:
    """Simulate publication database search"""
    await asyncio.sleep(0.12)  # Simulate API delay
    
    # Publications might have formal language patterns
    formal_terms = ["furthermore", "however", "therefore", "moreover", "consequently", "nevertheless", "subsequently", "accordingly", "specifically", "particularly", "essentially", "fundamentally", "demonstrates", "indicates", "suggests", "reveals", "establishes", "confirms", "validates", "supports"]
    
    words = text.lower().split()
    formal_count = sum(1 for word in words if word in formal_terms)
    
    return min(12, (formal_count / len(words)) * 100) if words else 0

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)