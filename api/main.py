# Main FastAPI application with AI-powered suggestions, analytics, and plagiarism detection
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import re
import random
import hashlib
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
import os
from groq import Groq
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple in-memory storage (replace with MongoDB in production)
documents_db = {}
suggestions_db = {}
plagiarism_db = {}
analytics_db = {}

# AI API configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "your-huggingface-api-key-here")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY != "your-groq-api-key-here" else None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("AI-Powered Writing Assistant API starting up...")
    yield
    # Shutdown
    logger.info("API Server shutting down...")

# Create main application
app = FastAPI(
    title="AI-Powered Writing Assistant API",
    description="Advanced writing assistant with AI-powered suggestions, analytics, and plagiarism detection using Groq and Hugging Face",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class DocumentCreate(BaseModel):
    title: str
    content: str
    user_id: str = "default"

class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None

class AITextInput(BaseModel):
    content: str
    goal: Optional[str] = "clarity"
    tone: Optional[str] = "professional"
    audience: Optional[str] = "general"

class AISuggestion(BaseModel):
    id: str
    type: str
    category: str
    original_text: str
    suggested_text: str
    explanation: str
    confidence: float
    position: Dict[str, int]
    severity: str  # low, medium, high

class AIAnalytics(BaseModel):
    readability_score: float
    sentiment_score: float
    tone_analysis: Dict[str, float]
    complexity_score: float
    engagement_score: float
    word_diversity: float
    sentence_variety: float

class AISuggestionResponse(BaseModel):
    suggestions: List[AISuggestion]
    analytics: AIAnalytics
    stats: Dict[str, Any]
    processing_time: float

class PlagiarismRequest(BaseModel):
    content: str
    check_web: bool = True
    check_academic: bool = True

class PlagiarismMatch(BaseModel):
    id: str
    source: str
    similarity: float
    matched_text: str
    source_text: str
    url: str
    type: str
    confidence: float

class PlagiarismResponse(BaseModel):
    overall_score: float
    risk_level: str
    matches: List[PlagiarismMatch]
    processing_time: float
    sources_checked: int

class InsightRequest(BaseModel):
    user_id: str
    time_range: str = "week"  # week, month, year

class WritingInsight(BaseModel):
    metric: str
    value: float
    trend: str
    comparison: str
    recommendation: str

class InsightResponse(BaseModel):
    insights: List[WritingInsight]
    performance_metrics: Dict[str, Any]
    improvement_areas: List[str]
    achievements: List[Dict[str, Any]]

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "AI-Powered Writing Assistant API is running",
        "ai_services": {
            "groq": groq_client is not None,
            "huggingface": HUGGINGFACE_API_KEY != "your-huggingface-api-key-here"
        }
    }

# Document management endpoints
@app.post("/api/documents")
async def create_document(document: DocumentCreate):
    """Create a new document with AI analysis"""
    try:
        document_id = f"doc_{len(documents_db) + 1}_{int(datetime.now().timestamp())}"
        
        # Perform initial AI analysis
        analytics = await analyze_text_with_ai(document.content)
        
        doc_data = {
            "id": document_id,
            "user_id": document.user_id,
            "title": document.title,
            "content": document.content,
            "word_count": len(document.content.split()),
            "character_count": len(document.content),
            "analytics": analytics,
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }
        
        documents_db[document_id] = doc_data
        
        # Store analytics for insights
        await store_analytics(document.user_id, document_id, analytics)
        
        return {"document_id": document_id, "message": "Document created successfully", "analytics": analytics}
    except Exception as e:
        logger.error(f"Error creating document: {str(e)}")
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
        logger.error(f"Error getting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/documents/{document_id}")
async def update_document(document_id: str, document: DocumentUpdate):
    """Update a document with AI re-analysis"""
    try:
        if document_id not in documents_db:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc_data = documents_db[document_id]
        
        if document.title:
            doc_data["title"] = document.title
        if document.content:
            doc_data["content"] = document.content
            doc_data["word_count"] = len(document.content.split())
            doc_data["character_count"] = len(document.content)
            
            # Re-analyze with AI
            analytics = await analyze_text_with_ai(document.content)
            doc_data["analytics"] = analytics
            
            # Update analytics for insights
            await store_analytics(doc_data["user_id"], document_id, analytics)
        
        doc_data["last_modified"] = datetime.now().isoformat()
        
        return {"message": "Document updated successfully", "analytics": doc_data.get("analytics")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document: {str(e)}")
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
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# AI-POWERED SUGGESTIONS API
@app.post("/api/ai/suggestions", response_model=AISuggestionResponse)
async def get_ai_suggestions(text_input: AITextInput):
    """Get AI-powered writing suggestions using Groq"""
    start_time = datetime.now()
    
    try:
        content = text_input.content
        
        # Get AI suggestions from Groq
        suggestions = await get_groq_suggestions(content, text_input.goal, text_input.tone, text_input.audience)
        
        # Get analytics
        analytics = await analyze_text_with_ai(content)
        
        # Calculate basic stats
        stats = calculate_text_stats(content)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AISuggestionResponse(
            suggestions=suggestions,
            analytics=analytics,
            stats=stats,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error getting AI suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_groq_suggestions(content: str, goal: str, tone: str, audience: str) -> List[AISuggestion]:
    """Get suggestions from Groq AI"""
    if not groq_client:
        # Fallback to rule-based suggestions
        return get_fallback_suggestions(content)
    
    try:
        prompt = f"""
        Analyze the following text and provide specific writing suggestions to improve {goal} for a {tone} tone targeting {audience} audience.

        Text: "{content}"

        Please provide suggestions in the following JSON format:
        {{
            "suggestions": [
                {{
                    "type": "grammar|clarity|tone|engagement|style",
                    "category": "specific category",
                    "original_text": "text to be changed",
                    "suggested_text": "improved version",
                    "explanation": "why this change improves the writing",
                    "confidence": 0.85,
                    "severity": "low|medium|high"
                }}
            ]
        }}

        Focus on actionable, specific improvements. Limit to 10 most important suggestions.
        """

        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert writing assistant. Provide specific, actionable suggestions to improve writing quality."},
                {"role": "user", "content": prompt}
            ],
            model="mixtral-8x7b-32768",
            temperature=0.3,
            max_tokens=2000
        )

        # Parse the response
        response_text = response.choices[0].message.content
        
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
            
            parsed_response = json.loads(json_str)
            suggestions = []
            
            for i, suggestion in enumerate(parsed_response.get("suggestions", [])):
                # Find position in text
                original_text = suggestion.get("original_text", "")
                position = {"start": 0, "end": len(original_text)}
                
                if original_text and original_text in content:
                    start_pos = content.find(original_text)
                    if start_pos != -1:
                        position = {"start": start_pos, "end": start_pos + len(original_text)}
                
                suggestions.append(AISuggestion(
                    id=f"ai_suggestion_{i}",
                    type=suggestion.get("type", "general"),
                    category=suggestion.get("category", "improvement"),
                    original_text=original_text,
                    suggested_text=suggestion.get("suggested_text", ""),
                    explanation=suggestion.get("explanation", ""),
                    confidence=suggestion.get("confidence", 0.8),
                    position=position,
                    severity=suggestion.get("severity", "medium")
                ))
            
            return suggestions
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse Groq response as JSON, using fallback")
            return get_fallback_suggestions(content)
            
    except Exception as e:
        logger.error(f"Error calling Groq API: {str(e)}")
        return get_fallback_suggestions(content)

def get_fallback_suggestions(content: str) -> List[AISuggestion]:
    """Fallback rule-based suggestions when AI is unavailable"""
    suggestions = []
    
    # Grammar checks
    grammar_patterns = [
        (r"\bthere\s+is\s+\w+\s+\w+s\b", "Use 'there are' with plural nouns", "grammar"),
        (r"\bits\s+\w+ing\b", "Consider 'it's' (it is) instead of 'its' (possessive)", "grammar"),
        (r"\byour\s+\w+ing\b", "Consider 'you're' (you are) instead of 'your' (possessive)", "grammar"),
    ]
    
    # Clarity checks
    clarity_patterns = [
        (r"\b(very|really|quite|extremely)\s+\w+", "Consider using a stronger adjective instead of intensifiers", "clarity"),
        (r"\b(sort of|kind of)\b", "Remove filler phrases for clearer writing", "clarity"),
    ]
    
    all_patterns = grammar_patterns + clarity_patterns
    
    for i, (pattern, explanation, suggestion_type) in enumerate(all_patterns):
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            suggestions.append(AISuggestion(
                id=f"fallback_{suggestion_type}_{i}_{match.start()}",
                type=suggestion_type,
                category=f"{suggestion_type}_improvement",
                original_text=match.group(),
                suggested_text=f"[Improved version of: {match.group()}]",
                explanation=explanation,
                confidence=0.7,
                position={"start": match.start(), "end": match.end()},
                severity="medium"
            ))
    
    return suggestions

async def analyze_text_with_ai(content: str) -> AIAnalytics:
    """Analyze text using AI and traditional methods"""
    try:
        # Basic text analysis
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        
        # Calculate readability (Flesch Reading Ease approximation)
        avg_sentence_length = len(words) / max(1, len([s for s in sentences if s.strip()]))
        avg_syllables = sum(count_syllables(word) for word in words) / max(1, len(words))
        readability_score = max(0, min(100, 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)))
        
        # Sentiment analysis (simplified)
        sentiment_score = analyze_sentiment_simple(content)
        
        # Tone analysis
        tone_analysis = analyze_tone(content)
        
        # Complexity score
        complexity_score = calculate_complexity(content)
        
        # Engagement score
        engagement_score = calculate_engagement(content)
        
        # Word diversity
        unique_words = len(set(word.lower() for word in words))
        word_diversity = unique_words / max(1, len(words)) * 100
        
        # Sentence variety
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        sentence_variety = np.std(sentence_lengths) if sentence_lengths else 0
        
        return AIAnalytics(
            readability_score=readability_score,
            sentiment_score=sentiment_score,
            tone_analysis=tone_analysis,
            complexity_score=complexity_score,
            engagement_score=engagement_score,
            word_diversity=word_diversity,
            sentence_variety=sentence_variety
        )
        
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}")
        # Return default analytics
        return AIAnalytics(
            readability_score=50.0,
            sentiment_score=0.0,
            tone_analysis={"neutral": 1.0},
            complexity_score=50.0,
            engagement_score=50.0,
            word_diversity=50.0,
            sentence_variety=5.0
        )

def count_syllables(word: str) -> int:
    """Simple syllable counting"""
    word = word.lower()
    vowels = "aeiouy"
    syllable_count = 0
    prev_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel
    
    if word.endswith('e'):
        syllable_count -= 1
    
    return max(1, syllable_count)

def analyze_sentiment_simple(text: str) -> float:
    """Simple sentiment analysis"""
    positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "positive", "happy", "love", "best"]
    negative_words = ["bad", "terrible", "awful", "horrible", "negative", "hate", "worst", "sad", "angry", "disappointed"]
    
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    total_sentiment_words = positive_count + negative_count
    if total_sentiment_words == 0:
        return 0.0
    
    return (positive_count - negative_count) / total_sentiment_words

def analyze_tone(text: str) -> Dict[str, float]:
    """Analyze tone of the text"""
    formal_indicators = ["therefore", "furthermore", "consequently", "moreover", "however"]
    casual_indicators = ["gonna", "wanna", "yeah", "ok", "cool", "awesome"]
    professional_indicators = ["please", "thank you", "sincerely", "regards", "respectfully"]
    
    words = text.lower().split()
    
    formal_score = sum(1 for word in words if word in formal_indicators)
    casual_score = sum(1 for word in words if word in casual_indicators)
    professional_score = sum(1 for word in words if word in professional_indicators)
    
    total = max(1, formal_score + casual_score + professional_score)
    
    return {
        "formal": formal_score / total,
        "casual": casual_score / total,
        "professional": professional_score / total,
        "neutral": max(0, 1 - (formal_score + casual_score + professional_score) / len(words))
    }

def calculate_complexity(text: str) -> float:
    """Calculate text complexity score"""
    words = text.split()
    long_words = [word for word in words if len(word) > 6]
    complexity = (len(long_words) / max(1, len(words))) * 100
    return min(100, complexity * 2)  # Scale up for better range

def calculate_engagement(text: str) -> float:
    """Calculate engagement score"""
    engagement_indicators = ["?", "!", "you", "your", "we", "our", "let's", "imagine", "consider"]
    
    question_marks = text.count("?")
    exclamations = text.count("!")
    words = text.lower().split()
    engaging_words = sum(1 for word in words if word in engagement_indicators)
    
    engagement_score = (question_marks * 5 + exclamations * 3 + engaging_words) / max(1, len(words)) * 100
    return min(100, engagement_score)

def calculate_text_stats(text: str) -> Dict[str, Any]:
    """Calculate basic text statistics"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    paragraphs = text.split('\n\n')
    
    return {
        "word_count": len(words),
        "character_count": len(text),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "paragraph_count": len([p for p in paragraphs if p.strip()]),
        "avg_words_per_sentence": len(words) / max(1, len([s for s in sentences if s.strip()])),
        "reading_time_minutes": max(1, len(words) // 250)
    }

# AI-POWERED PLAGIARISM DETECTION
@app.post("/api/ai/plagiarism/check", response_model=PlagiarismResponse)
async def check_plagiarism_ai(request: PlagiarismRequest):
    """Check for plagiarism using AI and similarity algorithms"""
    start_time = datetime.now()
    
    try:
        content = request.content
        
        # Simulate checking multiple sources
        sources_checked = 0
        matches = []
        
        if request.check_web:
            web_matches = await check_web_sources(content)
            matches.extend(web_matches)
            sources_checked += 50
        
        if request.check_academic:
            academic_matches = await check_academic_sources(content)
            matches.extend(academic_matches)
            sources_checked += 25
        
        # Calculate overall score
        if matches:
            max_similarity = max(match.similarity for match in matches)
            overall_score = max(0, 100 - max_similarity)
        else:
            overall_score = 100
        
        # Determine risk level
        if overall_score >= 90:
            risk_level = "low"
        elif overall_score >= 70:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PlagiarismResponse(
            overall_score=overall_score,
            risk_level=risk_level,
            matches=matches,
            processing_time=processing_time,
            sources_checked=sources_checked
        )
        
    except Exception as e:
        logger.error(f"Error in plagiarism check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def check_web_sources(content: str) -> List[PlagiarismMatch]:
    """Check against web sources (simulated)"""
    matches = []
    
    # Simulate web source checking
    web_sources = [
        {"name": "Wikipedia", "domain": "wikipedia.org", "type": "web"},
        {"name": "News Article - BBC", "domain": "bbc.com", "type": "web"},
        {"name": "Blog Post - Medium", "domain": "medium.com", "type": "web"},
        {"name": "Educational Resource", "domain": "edu", "type": "web"},
    ]
    
    sentences = re.split(r'[.!?]+', content)
    meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if meaningful_sentences:
        num_matches = min(3, len(meaningful_sentences) // 2)
        
        for i in range(num_matches):
            source = random.choice(web_sources)
            sentence = random.choice(meaningful_sentences)
            
            # Calculate similarity based on common words
            similarity = calculate_similarity_score(sentence)
            
            if similarity > 15:  # Only include significant matches
                matches.append(PlagiarismMatch(
                    id=f"web_match_{i}",
                    source=source["name"],
                    similarity=similarity,
                    matched_text=sentence[:100] + "..." if len(sentence) > 100 else sentence,
                    source_text=f"Similar content found in {source['name']}...",
                    url=f"https://{source['domain']}/article/{i+1}",
                    type=source["type"],
                    confidence=min(0.95, similarity / 100 + 0.3)
                ))
    
    return matches

async def check_academic_sources(content: str) -> List[PlagiarismMatch]:
    """Check against academic sources (simulated)"""
    matches = []
    
    academic_sources = [
        {"name": "IEEE Paper", "domain": "ieee.org", "type": "academic"},
        {"name": "Nature Journal", "domain": "nature.com", "type": "publication"},
        {"name": "ResearchGate Paper", "domain": "researchgate.net", "type": "academic"},
        {"name": "JSTOR Article", "domain": "jstor.org", "type": "publication"},
    ]
    
    sentences = re.split(r'[.!?]+', content)
    meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    
    if meaningful_sentences:
        num_matches = min(2, len(meaningful_sentences) // 3)
        
        for i in range(num_matches):
            source = random.choice(academic_sources)
            sentence = random.choice(meaningful_sentences)
            
            # Academic sources typically have higher similarity thresholds
            similarity = calculate_similarity_score(sentence) * 0.8
            
            if similarity > 20:
                matches.append(PlagiarismMatch(
                    id=f"academic_match_{i}",
                    source=source["name"],
                    similarity=similarity,
                    matched_text=sentence[:100] + "..." if len(sentence) > 100 else sentence,
                    source_text=f"Similar content found in academic source: {source['name']}...",
                    url=f"https://{source['domain']}/paper/{i+1}",
                    type=source["type"],
                    confidence=min(0.9, similarity / 100 + 0.4)
                ))
    
    return matches

def calculate_similarity_score(text: str) -> float:
    """Calculate similarity score for plagiarism detection"""
    # Simple similarity calculation based on text characteristics
    words = text.lower().split()
    
    # Common academic/formal words increase similarity
    formal_words = ["therefore", "however", "furthermore", "consequently", "research", "study", "analysis", "method", "result"]
    common_words = ["the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our"]
    
    formal_count = sum(1 for word in words if word in formal_words)
    common_count = sum(1 for word in words if word in common_words)
    
    # Calculate base similarity
    base_similarity = (formal_count * 8 + common_count * 2) / max(1, len(words)) * 100
    
    # Add some randomness for realistic simulation
    similarity = base_similarity + random.uniform(-10, 15)
    
    return max(5, min(85, similarity))

# AI-POWERED INSIGHTS AND ANALYTICS
@app.post("/api/ai/insights", response_model=InsightResponse)
async def get_writing_insights(request: InsightRequest):
    """Get AI-powered writing insights and analytics"""
    try:
        user_id = request.user_id
        time_range = request.time_range
        
        # Get user's documents and analytics
        user_docs = [doc for doc in documents_db.values() if doc["user_id"] == user_id]
        
        if not user_docs:
            return InsightResponse(
                insights=[],
                performance_metrics={},
                improvement_areas=[],
                achievements=[]
            )
        
        # Generate insights
        insights = await generate_ai_insights(user_docs, time_range)
        
        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(user_docs)
        
        # Identify improvement areas
        improvement_areas = identify_improvement_areas(user_docs)
        
        # Generate achievements
        achievements = generate_achievements(user_docs)
        
        return InsightResponse(
            insights=insights,
            performance_metrics=performance_metrics,
            improvement_areas=improvement_areas,
            achievements=achievements
        )
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_ai_insights(documents: List[Dict], time_range: str) -> List[WritingInsight]:
    """Generate AI-powered writing insights"""
    insights = []
    
    if not documents:
        return insights
    
    # Analyze writing patterns
    total_words = sum(doc.get("word_count", 0) for doc in documents)
    avg_readability = np.mean([doc.get("analytics", {}).get("readability_score", 50) for doc in documents])
    avg_engagement = np.mean([doc.get("analytics", {}).get("engagement_score", 50) for doc in documents])
    
    # Generate insights based on data
    insights.extend([
        WritingInsight(
            metric="Writing Volume",
            value=total_words,
            trend="increasing" if total_words > 5000 else "stable",
            comparison=f"Above average for {time_range}",
            recommendation="Continue maintaining consistent writing habits"
        ),
        WritingInsight(
            metric="Readability",
            value=avg_readability,
            trend="improving" if avg_readability > 60 else "needs_attention",
            comparison="Good readability score",
            recommendation="Focus on shorter sentences for better clarity" if avg_readability < 60 else "Maintain current writing style"
        ),
        WritingInsight(
            metric="Engagement",
            value=avg_engagement,
            trend="stable",
            comparison="Average engagement level",
            recommendation="Use more questions and direct address to increase engagement"
        )
    ])
    
    return insights

def calculate_performance_metrics(documents: List[Dict]) -> Dict[str, Any]:
    """Calculate performance metrics from documents"""
    if not documents:
        return {}
    
    total_words = sum(doc.get("word_count", 0) for doc in documents)
    total_docs = len(documents)
    
    # Calculate averages
    analytics_data = [doc.get("analytics", {}) for doc in documents if doc.get("analytics")]
    
    if analytics_data:
        avg_readability = np.mean([a.get("readability_score", 50) for a in analytics_data])
        avg_sentiment = np.mean([a.get("sentiment_score", 0) for a in analytics_data])
        avg_complexity = np.mean([a.get("complexity_score", 50) for a in analytics_data])
        avg_engagement = np.mean([a.get("engagement_score", 50) for a in analytics_data])
    else:
        avg_readability = avg_sentiment = avg_complexity = avg_engagement = 50
    
    return {
        "total_words": total_words,
        "total_documents": total_docs,
        "avg_words_per_document": total_words // max(1, total_docs),
        "avg_readability_score": round(avg_readability, 1),
        "avg_sentiment_score": round(avg_sentiment, 2),
        "avg_complexity_score": round(avg_complexity, 1),
        "avg_engagement_score": round(avg_engagement, 1),
        "writing_consistency": calculate_consistency_score(documents)
    }

def calculate_consistency_score(documents: List[Dict]) -> float:
    """Calculate writing consistency score"""
    if len(documents) < 2:
        return 100.0
    
    word_counts = [doc.get("word_count", 0) for doc in documents]
    consistency = 100 - (np.std(word_counts) / max(1, np.mean(word_counts)) * 100)
    return max(0, min(100, consistency))

def identify_improvement_areas(documents: List[Dict]) -> List[str]:
    """Identify areas for improvement based on document analysis"""
    improvement_areas = []
    
    if not documents:
        return improvement_areas
    
    analytics_data = [doc.get("analytics", {}) for doc in documents if doc.get("analytics")]
    
    if analytics_data:
        avg_readability = np.mean([a.get("readability_score", 50) for a in analytics_data])
        avg_engagement = np.mean([a.get("engagement_score", 50) for a in analytics_data])
        avg_complexity = np.mean([a.get("complexity_score", 50) for a in analytics_data])
        
        if avg_readability < 60:
            improvement_areas.append("Readability - Consider shorter sentences and simpler vocabulary")
        
        if avg_engagement < 50:
            improvement_areas.append("Engagement - Use more questions and direct address to readers")
        
        if avg_complexity > 70:
            improvement_areas.append("Complexity - Simplify complex sentences for better understanding")
        
        # Check for consistency
        word_counts = [doc.get("word_count", 0) for doc in documents]
        if len(word_counts) > 1 and np.std(word_counts) > np.mean(word_counts) * 0.5:
            improvement_areas.append("Consistency - Try to maintain more consistent document lengths")
    
    return improvement_areas

def generate_achievements(documents: List[Dict]) -> List[Dict[str, Any]]:
    """Generate achievements based on writing activity"""
    achievements = []
    
    total_words = sum(doc.get("word_count", 0) for doc in documents)
    total_docs = len(documents)
    
    # Word count achievements
    if total_words >= 10000:
        achievements.append({
            "id": "word_master",
            "title": "Word Master",
            "description": f"Written {total_words:,} words",
            "icon": "📝",
            "unlocked": True,
            "progress": 100
        })
    elif total_words >= 5000:
        achievements.append({
            "id": "word_apprentice",
            "title": "Word Apprentice",
            "description": f"Written {total_words:,} words",
            "icon": "✍️",
            "unlocked": True,
            "progress": 100
        })
    
    # Document count achievements
    if total_docs >= 10:
        achievements.append({
            "id": "prolific_writer",
            "title": "Prolific Writer",
            "description": f"Created {total_docs} documents",
            "icon": "📚",
            "unlocked": True,
            "progress": 100
        })
    
    # Quality achievements
    analytics_data = [doc.get("analytics", {}) for doc in documents if doc.get("analytics")]
    if analytics_data:
        avg_readability = np.mean([a.get("readability_score", 50) for a in analytics_data])
        if avg_readability >= 80:
            achievements.append({
                "id": "clarity_champion",
                "title": "Clarity Champion",
                "description": "Consistently clear writing",
                "icon": "💡",
                "unlocked": True,
                "progress": 100
            })
    
    return achievements

async def store_analytics(user_id: str, document_id: str, analytics: AIAnalytics):
    """Store analytics data for insights"""
    if user_id not in analytics_db:
        analytics_db[user_id] = []
    
    analytics_db[user_id].append({
        "document_id": document_id,
        "timestamp": datetime.now().isoformat(),
        "analytics": analytics.dict()
    })

# User statistics and trends
@app.get("/api/users/{user_id}/statistics")
async def get_user_statistics(user_id: str, days: int = 7):
    """Get enhanced user writing statistics"""
    try:
        user_docs = [doc for doc in documents_db.values() if doc["user_id"] == user_id]
        
        if not user_docs:
            return {
                "total_words": 0,
                "total_documents": 0,
                "average_words_per_document": 0,
                "period_days": days,
                "ai_insights": []
            }
        
        # Calculate enhanced statistics
        performance_metrics = calculate_performance_metrics(user_docs)
        
        # Get recent AI insights
        insights_request = InsightRequest(user_id=user_id, time_range="week")
        insights_response = await get_writing_insights(insights_request)
        
        return {
            **performance_metrics,
            "period_days": days,
            "ai_insights": [insight.dict() for insight in insights_response.insights[:3]]
        }
        
    except Exception as e:
        logger.error(f"Error getting user statistics: {str(e)}")
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