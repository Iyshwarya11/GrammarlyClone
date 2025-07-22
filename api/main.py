import os
import json
import re
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys from environment variables (do NOT hardcode keys here)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "your-huggingface-api-key-here")
# Tip: Set these in your environment or a .env file for security.
# Example (on Windows):
#   set GROQ_API_KEY=your-groq-key-here
#   set HUGGINGFACE_API_KEY=your-hf-key-here

# In-memory storage (replace with database in production)
documents_db = {}
analytics_db = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("API Server starting up...")
    initialize_sample_data()
    yield
    # Shutdown
    logger.info("API Server shutting down...")

def initialize_sample_data():
    """Initialize sample data for frontend testing"""
    sample_docs = [
        {
            "id": "1",
            "user_id": "default",
            "title": "Marketing Proposal Draft",
            "content": "This is a comprehensive marketing proposal for our new product launch. The strategy includes digital marketing, social media campaigns, and traditional advertising methods.",
            "word_count": 1250,
            "character_count": 7500,
            "score": 89,
            "status": "In Progress",
            "created_at": (datetime.now() - timedelta(hours=2)).isoformat(),
            "last_modified": (datetime.now() - timedelta(hours=2)).isoformat(),
            "analytics": {
                "readability_score": 75.0,
                "sentiment_score": 0.2,
                "tone_analysis": {"professional": 0.8, "formal": 0.6, "neutral": 0.4},
                "complexity_score": 65.0,
                "engagement_score": 70.0,
                "word_diversity": 85.0,
                "sentence_variety": 7.2
            }
        },
        {
            "id": "2",
            "user_id": "default",
            "title": "Research Paper - AI Ethics",
            "content": "Artificial intelligence ethics is a critical field that examines the moral implications of AI systems. This paper explores the key ethical considerations in AI development and deployment.",
            "word_count": 3500,
            "character_count": 21000,
            "score": 95,
            "status": "Completed",
            "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
            "last_modified": (datetime.now() - timedelta(days=1)).isoformat(),
            "analytics": {
                "readability_score": 82.0,
                "sentiment_score": 0.1,
                "tone_analysis": {"formal": 0.9, "professional": 0.8, "neutral": 0.7},
                "complexity_score": 78.0,
                "engagement_score": 65.0,
                "word_diversity": 92.0,
                "sentence_variety": 8.5
            }
        },
        {
            "id": "3",
            "user_id": "default",
            "title": "Email Campaign Copy",
            "content": "Subject: Exciting New Features Coming Soon! Dear valued customers, we're thrilled to announce some amazing updates to our platform that will enhance your experience.",
            "word_count": 800,
            "character_count": 4800,
            "score": 92,
            "status": "Reviewed",
            "created_at": (datetime.now() - timedelta(days=3)).isoformat(),
            "last_modified": (datetime.now() - timedelta(days=3)).isoformat(),
            "analytics": {
                "readability_score": 88.0,
                "sentiment_score": 0.6,
                "tone_analysis": {"casual": 0.7, "professional": 0.5, "neutral": 0.3},
                "complexity_score": 45.0,
                "engagement_score": 85.0,
                "word_diversity": 78.0,
                "sentence_variety": 6.8
            }
        }
    ]
    
    for doc in sample_docs:
        documents_db[doc["id"]] = doc
# Create main application
app = FastAPI(
    title="GrammarlyClone AI API",
    description="AI-powered writing assistant with real-time suggestions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class DocumentResponse(BaseModel):
    id: str
    title: str
    content: str
    word_count: int
    score: int
    status: str
    last_modified: str
    created_at: str

class DashboardStats(BaseModel):
    words_written: int
    documents_created: int
    average_score: int
    improvement_rate: int

class RecentDocument(BaseModel):
    id: str
    title: str
    last_modified: str
    word_count: int
    score: int
    status: str
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
    rule: Optional[str] = None  # Grammar rule or writing principle

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
    type: str
    title: str
    description: str
    impact: str
    recommendation: str

class InsightResponse(BaseModel):
    insights: List[WritingInsight]
    performance_metrics: Dict[str, Any]
    improvement_areas: List[str]
    achievements: List[Dict[str, Any]]

class RewriteRequest(BaseModel):
    content: str
    goal: str = "formal"

class RewriteResponse(BaseModel):
    rewritten_text: str

class SummarizeRequest(BaseModel):
    content: str

class SummarizeResponse(BaseModel):
    summary: str

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "groq_api": "available" if GROQ_API_KEY and GROQ_API_KEY != "your-groq-api-key-here" else "not_configured",
            "huggingface_api": "available" if HUGGINGFACE_API_KEY and HUGGINGFACE_API_KEY != "your-huggingface-api-key-here" else "not_configured"
        }
    }

# Document management endpoints
@app.post("/api/documents")
async def create_document(document: DocumentCreate):
    """Create a new document"""
    try:
        doc_id = f"doc_{len(documents_db) + 1}"
        word_count = len(document.content.split())
        
        # Perform initial AI analysis
        analytics = await analyze_text_with_ai(document.content)
        
        # Calculate score based on analytics
        score = calculate_document_score(analytics)
        
        doc_data = {
            "id": document_id,
            "user_id": document.user_id,
            "title": document.title,
            "content": document.content,
            "word_count": len(document.content.split()),
            "character_count": len(document.content),
            "score": score,
            "status": "In Progress",
            "analytics": analytics,
            "created_at": datetime.now().isoformat(),
            "user_id": document.user_id
        }
        
        documents_db[doc_id] = new_doc
        
        # Store analytics for insights
        await store_analytics(document.user_id, document_id, analytics)
        
        return {
            "document_id": document_id, 
            "message": "Document created successfully", 
            "score": score,
            "analytics": analytics.dict() if hasattr(analytics, 'dict') else analytics
        }
        return DocumentResponse(**new_doc)
    except Exception as e:
        logger.error(f"Error creating document: {e}")
        raise HTTPException(status_code=500, detail="Failed to create document")

@app.get("/api/documents")
async def get_documents(user_id: str = "default", limit: int = 10):
    """Get all documents for a user"""
    try:
        user_docs = [doc for doc in documents_db.values() if doc.get("user_id") == user_id]
        sorted_docs = sorted(user_docs, key=lambda x: x["last_modified"], reverse=True)
        return [DocumentResponse(**doc) for doc in sorted_docs[:limit]]
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch documents")

def calculate_document_score(analytics) -> int:
    """Calculate overall document score from analytics"""
    try:
        if hasattr(analytics, 'dict'):
            analytics_dict = analytics.dict()
        else:
            analytics_dict = analytics
            
        readability = analytics_dict.get("readability_score", 50)
        engagement = analytics_dict.get("engagement_score", 50)
        word_diversity = analytics_dict.get("word_diversity", 50)
        
        # Weighted average
        score = (readability * 0.4 + engagement * 0.3 + word_diversity * 0.3)
        return int(max(0, min(100, score)))
    except:
        return 75  # Default score
@app.get("/api/documents/{document_id}")
async def get_document(document_id: str):
    """Get a specific document and update last_modified to now for history tracking"""
    try:
        if document_id not in documents_db:
            raise HTTPException(status_code=404, detail="Document not found")
        # Update last_modified to now
        documents_db[document_id]["last_modified"] = datetime.now().isoformat()
        return DocumentResponse(**documents_db[document_id])
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch document")

@app.put("/api/documents/{document_id}")
async def update_document(document_id: str, document: DocumentUpdate):
    """Update a document"""
    try:
        if document_id not in documents_db:
            raise HTTPException(status_code=404, detail="Document not found")
        
        current_doc = documents_db[document_id]
        
        if document.title is not None:
            current_doc["title"] = document.title
        
        if document.content is not None:
            current_doc["content"] = document.content
            current_doc["word_count"] = len(document.content.split())
        
        current_doc["last_modified"] = datetime.now().isoformat()
        
        documents_db[document_id] = current_doc
        
        return DocumentResponse(**current_doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update document")

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
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

# AI Suggestions endpoint
@app.post("/api/ai/suggestions", response_model=AISuggestionResponse)
async def get_ai_suggestions(text_input: AITextInput):
    print("DEBUG: /api/ai/suggestions called with:", text_input.content)
    start_time = datetime.now()
    try:
        # Get AI suggestions
        ai_suggestions = await try_huggingface_api(
            text_input.content,
            text_input.goal,
            text_input.tone,
            text_input.audience
        )
        # Get fallback suggestions
        fallback_suggestions = get_fallback_suggestions(text_input.content)
        # Combine both, filtering duplicates
        all_suggestions = ai_suggestions + [
            s for s in fallback_suggestions
            if not any(
                s.original_text == ai_s.original_text and s.suggested_text == ai_s.suggested_text
                for ai_s in ai_suggestions
            )
        ]
        analytics = await analyze_text_with_ai(text_input.content)
        stats = calculate_text_stats(text_input.content)
        processing_time = (datetime.now() - start_time).total_seconds()
        return AISuggestionResponse(
            suggestions=all_suggestions,
            analytics=analytics,
            stats=stats,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error getting AI suggestions: {e}")
        # On error, use fallback only
        suggestions = get_fallback_suggestions(text_input.content)
        analytics = await analyze_text_with_ai(text_input.content)
        stats = calculate_text_stats(text_input.content)
        processing_time = (datetime.now() - start_time).total_seconds()
        return AISuggestionResponse(
            suggestions=suggestions,
            analytics=analytics,
            stats=stats,
            processing_time=processing_time
        )

async def try_groq_api(content: str, goal: str, tone: str, audience: str) -> List[AISuggestion]:
    """Try to get suggestions using Groq API"""
    try:
        logger.info(f"GROQ_API_KEY is set: {GROQ_API_KEY[:8]}..." if GROQ_API_KEY and GROQ_API_KEY != "your-groq-api-key-here" else "GROQ_API_KEY is not set or is default.")
        if GROQ_API_KEY == "your-groq-api-key-here":
            logger.warning("Groq API key is not set. Skipping Groq API call.")
            return []
        groq_url = "https://api.groq.com/openai/v1/chat/completions"
        prompt = f"""
You are a professional writing assistant. Analyze the following text word by word and provide comprehensive writing suggestions to improve {goal} for a {tone} tone targeting {audience} audience.

Text: "{content}"

Respond ONLY with a valid JSON object in the following format:
{{
  "suggestions": [
    {{
      "type": "spelling|grammar|clarity|tone|engagement|style|punctuation|tense",
      "category": "specific category",
      "original_text": "exact text to be changed",
      "suggested_text": "improved version",
      "explanation": "detailed explanation of why this change improves the writing",
      "confidence": 0.85,
      "severity": "low|medium|high"
    }}
  ]
}}
NO explanation, NO extra text, ONLY the JSON object.
"""
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        logger.info(f"Sending Groq API request: {groq_url}")
        logger.info(f"Prompt: {prompt[:200]}...")
        async with aiohttp.ClientSession() as session:
            response = await session.post(groq_url, headers=headers, json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 800
            }, timeout=20)
            logger.info(f"Groq response status: {response.status}")
            response_text = await response.text()
            logger.info(f"Groq response text: {response_text[:500]}")
            if response.status == 200:
                try:
                    data = await response.json()
                    response_text = data["choices"][0]["message"]["content"]
                    json_match = re.search(r'\{[\s\S]*\}', response_text)
                    if json_match:
                        json_str = json_match.group(0)
                        try:
                            parsed_response = json.loads(json_str)
                            suggestions = []
                            for i, suggestion in enumerate(parsed_response.get("suggestions", [])):
                                original_text = suggestion.get("original_text", "")
                                suggested_text = suggestion.get("suggested_text", "")
                                if not suggested_text.strip() and suggestion.get("type") not in ["spelling", "grammar"]:
                                    continue
                                position = {"start": 0, "end": len(original_text)}
                                if original_text and original_text in content:
                                    start_pos = content.find(original_text)
                                    if start_pos != -1:
                                        position = {"start": start_pos, "end": start_pos + len(original_text)}
                                suggestions.append(AISuggestion(
                                    id=f"groq_suggestion_{i}",
                                    type=suggestion.get("type", "general"),
                                    category=suggestion.get("category", "improvement"),
                                    original_text=original_text,
                                    suggested_text=suggested_text,
                                    explanation=suggestion.get("explanation", ""),
                                    confidence=suggestion.get("confidence", 0.8),
                                    position=position,
                                    severity=suggestion.get("severity", "medium")
                                ))
                            return suggestions
                        except Exception as e:
                            logger.warning(f"Failed to parse Groq JSON: {e}\nRaw JSON: {json_str}")
                            return []
                except Exception as e:
                    logger.warning(f"Failed to parse Groq API response JSON: {e}\nRaw response: {response_text}")
                    return []
            else:
                logger.warning(f"Groq API returned non-200 status: {response.status}\nResponse: {response_text}")
                return []
    except Exception as e:
        logger.warning(f"Groq API failed: {e}")
    return []

async def try_huggingface_api(content: str, goal: str, tone: str, audience: str) -> List[AISuggestion]:
    print("DEBUG: try_huggingface_api called with:", content)
    """Use Hugging Face grammar correction model to get a single suggestion."""
    url = "https://api-inference.huggingface.co/models/prithivida/grammar_error_correcter_v1"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": content}
    try:
        logger.info(f"Calling Hugging Face API with content: {content}")
        logger.info(f"Headers: {headers}")
        logger.info(f"Payload: {payload}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                result = await resp.json()
                logger.info(f"Hugging Face response: {result}")
                # The model returns a list of dicts with 'generated_text'
                if isinstance(result, list) and result and 'generated_text' in result[0]:
                    corrected = result[0]['generated_text']
                else:
                    corrected = content
                if corrected.strip() == content.strip():
                    return []  # No correction needed
                return [
                    AISuggestion(
                        id="hf-1",
                        type="grammar",
                        category="correction",
                        original_text=content,
                        suggested_text=corrected,
                        explanation="Corrected using Hugging Face grammar model.",
                        confidence=0.95,
                        position={"start": 0, "end": len(content)},
                        severity="medium",
                        rule="grammar"
                    )
                ]
    except Exception as e:
        logger.error(f"Hugging Face API error: {e}")
        return []

def get_fallback_suggestions(content: str) -> List[AISuggestion]:
    """Get fallback suggestions when AI APIs are not available"""
    suggestions = []
    
    # Expanded spelling and grammar checks
    common_mistakes = {
        # Existing and expanded common misspellings
        "recieve": "receive", "definately": "definitely", "seperate": "separate", "occured": "occurred", "neccessary": "necessary", "accomodate": "accommodate", "begining": "beginning", "beleive": "believe", "calender": "calendar", "collegue": "colleague", "wich": "which", "adress": "address", "enviroment": "environment", "goverment": "government", "occassion": "occasion", "publically": "publicly", "untill": "until", "writting": "writing", "thier": "their", "teh": "the", "alot": "a lot", "seperately": "separately", "succesful": "successful", "tommorow": "tomorrow", "wierd": "weird", "schol": "school",
        "acheive": "achieve", "arguement": "argument", "buisness": "business", "comming": "coming", "happend": "happened", "occurence": "occurrence", "becuase": "because", "agian": "again", "embarass": "embarrass", "foriegn": "foreign", "gratefull": "grateful", "humerous": "humorous", "noticable": "noticeable", "posession": "possession", "prefered": "preferred", "presance": "presence", "realy": "really", "remeber": "remember", "suprise": "surprise", "tendancy": "tendency", "treshold": "threshold", "truely": "truly", "ture": "true", "accomodate": "accommodate", "occuring": "occurring", "perseverence": "perseverance", "reciept": "receipt", "restaraunt": "restaurant", "seige": "siege", "supercede": "supersede", "untill": "until", "wierd": "weird", "writen": "written", "yatch": "yacht",
        # New additions
        "seperated": "separated", "occassionally": "occasionally", "mispell": "misspell", "adress": "address", "concious": "conscious", "conciousness": "consciousness", "embarassment": "embarrassment", "existance": "existence", "goverment": "government", "harrass": "harass", "independant": "independent", "neccessary": "necessary", "occured": "occurred", "occuring": "occurring", "posession": "possession", "publically": "publicly", "reccommend": "recommend", "recieve": "receive", "refered": "referred", "seperately": "separately", "succesful": "successful", "tommorrow": "tomorrow", "untill": "until", "wierd": "weird", "writting": "writing", "definate": "definite", "goverment": "government", "happend": "happened", "knowlege": "knowledge", "occassion": "occasion", "occurence": "occurrence", "prefered": "preferred", "reciept": "receipt", "sieze": "seize", "tommorow": "tomorrow", "untill": "until", "wierd": "weird", "writen": "written", "yatch": "yacht"
    }
    # Scan for all common mistakes
    for mistake, correction in common_mistakes.items():
        pattern = re.compile(rf'\b{mistake}\b', re.IGNORECASE)
        for match in pattern.finditer(content):
            start_pos = match.start()
            suggestions.append(AISuggestion(
                id=f"fallback_spelling_{len(suggestions)}",
                type="spelling",
                category="spelling",
                original_text=content[start_pos:start_pos + len(mistake)],
                suggested_text=correction,
                explanation=f"'{mistake}' is misspelled. The correct spelling is '{correction}'.",
                confidence=0.95,
                position={"start": start_pos, "end": start_pos + len(mistake)},
                severity="high"
            ))
    # Add more common confusions
    confusion_patterns = [
        (r"\byour\b", "you're", "Did you mean 'you're' (you are)?", "grammar"),
        (r"\byou're\b", "your", "Did you mean 'your' (possessive)?", "grammar"),
        (r"\bits\b", "it's", "Did you mean 'it's' (it is)?", "grammar"),
        (r"\bit's\b", "its", "Did you mean 'its' (possessive)?", "grammar"),
        (r"\btheir\b", "they're", "Did you mean 'they're' (they are)?", "grammar"),
        (r"\bthey're\b", "their", "Did you mean 'their' (possessive)?", "grammar"),
        (r"\bthere\b", "their", "Did you mean 'their' (possessive)?", "grammar"),
    ]
    for pattern, correction, explanation, type_ in confusion_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            start_pos = match.start()
            suggestions.append(AISuggestion(
                id=f"fallback_confusion_{len(suggestions)}",
                type=type_,
                category=type_,
                original_text=match.group(0),
                suggested_text=correction,
                explanation=explanation,
                confidence=0.85,
                position={"start": start_pos, "end": start_pos + len(match.group(0))},
                severity="medium"
            ))
    # Basic grammar/tense checks (expandable)
    patterns = [
        (r'\bI am going to the store yesterday\b', "I went to the store yesterday", "Incorrect verb tense. Use past tense for actions that happened in the past.", "tense"),
        (r'\bShe have been working\b', "She has been working", "Incorrect subject-verb agreement. Use 'has' with singular third-person subjects.", "grammar"),
        (r"\bHe don't\b", "He doesn't", "Incorrect verb form. Use 'doesn't' for third-person singular.", "grammar"),
        (r'\bI has\b', "I have", "Incorrect verb form. Use 'have' with 'I'.", "grammar"),
        (r'\bmore better\b', "better", "Redundant comparative. Use 'better' instead of 'more better'.", "clarity"),
        (r'\bmost best\b', "best", "Redundant superlative. Use 'best' instead of 'most best'.", "clarity"),
        (r'\bI didn\'t went\b', "I didn't go", "Incorrect past tense after 'didn't'. Use base form.", "tense"),
        (r'\bbetween you and I\b', "between you and me", "Incorrect pronoun case after 'between'.", "grammar"),
        (r'\bhe go\b', "he goes", "Incorrect verb form. Use 'goes' with 'he' (third-person singular).", "grammar"),
        (r'\bshe go\b', "she goes", "Incorrect verb form. Use 'goes' with 'she' (third-person singular).", "grammar"),
        (r'\bit go\b', "it goes", "Incorrect verb form. Use 'goes' with 'it' (third-person singular).", "grammar"),
        (r'\bhe have\b', "he has", "Incorrect verb form. Use 'has' with 'he'.", "grammar"),
        (r'\bshe have\b', "she has", "Incorrect verb form. Use 'has' with 'she'.", "grammar"),
        (r'\bit have\b', "it has", "Incorrect verb form. Use 'has' with 'it'.", "grammar"),
        (r'\bI goes\b', "I go", "Incorrect verb form. Use 'go' with 'I'.", "grammar"),
        (r'\bthey goes\b', "they go", "Incorrect verb form. Use 'go' with 'they'.", "grammar"),
        (r'\bwe goes\b', "we go", "Incorrect verb form. Use 'go' with 'we'.", "grammar"),
        (r'\byou goes\b', "you go", "Incorrect verb form. Use 'go' with 'you'.", "grammar"),
        (r"\bshe don't\b", "she doesn't", "Incorrect verb form. Use 'doesn't' for third-person singular.", "grammar"),
        (r"\bit don't\b", "it doesn't", "Incorrect verb form. Use 'doesn't' for third-person singular.", "grammar"),
        (r'\bI seen\b', "I saw", "Incorrect verb form. Use 'saw' as the past tense of 'see'.", "grammar"),
        (r'\bI done\b', "I did", "Incorrect verb form. Use 'did' as the past tense of 'do'.", "grammar"),
        (r'\bI have ate\b', "I have eaten", "Incorrect verb form. Use 'eaten' with 'have'.", "grammar"),
        (r'\bI gone\b', "I went", "Incorrect verb form. Use 'went' as the past tense of 'go'.", "grammar"),
        (r'\bI brang\b', "I brought", "Incorrect verb form. Use 'brought' as the past tense of 'bring'.", "grammar"),
        (r'\bI runned\b', "I ran", "Incorrect verb form. Use 'ran' as the past tense of 'run'.", "grammar"),
        (r'\bI writed\b', "I wrote", "Incorrect verb form. Use 'wrote' as the past tense of 'write'.", "grammar"),
        (r'\bI buyed\b', "I bought", "Incorrect verb form. Use 'bought' as the past tense of 'buy'.", "grammar"),
        (r'\bI thinked\b', "I thought", "Incorrect verb form. Use 'thought' as the past tense of 'think'.", "grammar"),
        (r'\bI feeled\b', "I felt", "Incorrect verb form. Use 'felt' as the past tense of 'feel'.", "grammar"),
        (r'\bI sleeped\b', "I slept", "Incorrect verb form. Use 'slept' as the past tense of 'sleep'.", "grammar"),
        (r'\bI drinked\b', "I drank", "Incorrect verb form. Use 'drank' as the past tense of 'drink'.", "grammar"),
        (r'\bI catched\b', "I caught", "Incorrect verb form. Use 'caught' as the past tense of 'catch'.", "grammar"),
        (r'\bI teached\b', "I taught", "Incorrect verb form. Use 'taught' as the past tense of 'teach'.", "grammar"),
        (r'\bI bringed\b', "I brought", "Incorrect verb form. Use 'brought' as the past tense of 'bring'.", "grammar"),
        (r'\bI costed\b', "I cost", "Incorrect verb form. Use 'cost' as the past tense of 'cost'.", "grammar"),
        (r'\bI hurted\b', "I hurt", "Incorrect verb form. Use 'hurt' as the past tense of 'hurt'.", "grammar"),
        (r'\bI putted\b', "I put", "Incorrect verb form. Use 'put' as the past tense of 'put'.", "grammar"),
        (r'\bI readed\b', "I read", "Incorrect verb form. Use 'read' as the past tense of 'read'.", "grammar"),
        (r'\bI sended\b', "I sent", "Incorrect verb form. Use 'sent' as the past tense of 'send'.", "grammar"),
        (r'\bI shaked\b', "I shook", "Incorrect verb form. Use 'shook' as the past tense of 'shake'.", "grammar"),
        (r'\bI shooted\b', "I shot", "Incorrect verb form. Use 'shot' as the past tense of 'shoot'.", "grammar"),
        (r'\bI spended\b', "I spent", "Incorrect verb form. Use 'spent' as the past tense of 'spend'.", "grammar"),
        (r'\bI standed\b', "I stood", "Incorrect verb form. Use 'stood' as the past tense of 'stand'.", "grammar"),
        (r'\bI swimmed\b', "I swam", "Incorrect verb form. Use 'swam' as the past tense of 'swim'.", "grammar"),
        (r'\bI understanded\b', "I understood", "Incorrect verb form. Use 'understood' as the past tense of 'understand'.", "grammar"),
        (r'\bI weared\b', "I wore", "Incorrect verb form. Use 'wore' as the past tense of 'wear'.", "grammar"),
        (r'\bI winced\b', "I won", "Incorrect verb form. Use 'won' as the past tense of 'win'.", "grammar"),
        (r"It don’t works", "It doesn’t work", "Incorrect verb form. Use 'doesn’t work'.", "grammar"),
        (r"She haven’t a car", "She doesn’t have a car", "Incorrect negative. Use 'doesn’t have'.", "grammar"),
        (r"He is richest than me", "He is richer than me", "Incorrect comparative. Use 'richer'.", "grammar"),
        (r"The room was more bigger than before", "The room was bigger than before", "Redundant comparative. Use 'bigger'.", "clarity"),
        (r"It’s depend on the weather", "It depends on the weather", "Incorrect verb form. Use 'depends'.", "grammar"),
        (r"He is afraid from spiders", "He is afraid of spiders", "Incorrect preposition. Use 'afraid of'.", "grammar"),
        (r"She sings good", "She sings well", "Incorrect adverb. Use 'well'.", "grammar"),
        (r"I saw him before two hours", "I saw him two hours ago", "Incorrect time expression. Use 'two hours ago'.", "grammar"),
        (r"The house is more old", "The house is older", "Incorrect comparative. Use 'older'.", "grammar"),
        (r"You must not to be late", "You must not be late", "Do not use 'to' after 'must not'.", "grammar"),
        (r"He never comes late, isn’t it", "He never comes late, does he?", "Incorrect question tag. Use 'does he?'.", "grammar"),
        (r"She cried because she was fear", "She cried because she was afraid", "Incorrect adjective. Use 'afraid'.", "grammar"),
        (r"I didn’t saw him at the party", "I didn’t see him at the party", "Incorrect verb form after 'didn’t'. Use base form 'see'.", "grammar"),
        (r"He has much money", "He has a lot of money", "Use 'a lot of' for quantity with countable/uncountable nouns.", "grammar"),
        (r"The dog was barked loudly", "The dog barked loudly", "Incorrect passive. Use active voice 'barked loudly'.", "grammar"),
        (r"It was very hotly in the room", "It was very hot in the room", "Incorrect adverb. Use 'hot'.", "grammar"),
        (r"My friend she is very kind", "My friend is very kind", "Redundant subject. Remove 'she'.", "grammar"),
        (r"The informations you gave are wrong", "The information you gave is wrong", "'Information' is uncountable. Use singular form.", "grammar"),
        (r"He don’t wants help", "He doesn’t want help", "Incorrect verb form. Use 'doesn’t want'.", "grammar"),
        (r"I was there yesterday, didn’t I", "I was there yesterday, wasn’t I?", "Incorrect question tag. Use 'wasn’t I?'.", "grammar"),
        (r"I lost my keys, isn’t it", "I lost my keys, didn’t I?", "Incorrect question tag. Use 'didn’t I?'.", "grammar"),
        (r"The cake was baking by my mom", "The cake was baked by my mom", "Incorrect passive. Use 'baked'.", "grammar"),
        (r"He don’t likes coffee", "He doesn’t like coffee", "Incorrect verb form. Use 'doesn’t like'.", "grammar"),
        (r"The girl which sings is my sister", "The girl who sings is my sister", "Use 'who' for people.", "grammar"),
        (r"He go school by foot", "He goes to school on foot", "Incorrect verb form and preposition. Use 'goes to school on foot'.", "grammar"),
        (r"She told that she will goes", "She said that she will go", "Incorrect verb form. Use 'will go'.", "grammar"),
        (r"I will going to the shop", "I will go to the shop", "Incorrect verb form. Use 'will go'.", "grammar"),
        (r"They has arrived now", "They have arrived now", "Incorrect verb form. Use 'have' with 'they'.", "grammar"),
        (r"She don’t has any idea", "She doesn’t have any idea", "Incorrect verb form. Use 'doesn’t have'.", "grammar"),
        (r"You should goes now", "You should go now", "Incorrect verb form. Use 'go' after 'should'.", "grammar"),
        (r"He needs to goes home", "He needs to go home", "Incorrect verb form. Use 'go' after 'needs to'.", "grammar"),
        (r"She study hard for exam", "She studies hard for the exam", "Incorrect verb form and missing article. Use 'studies hard for the exam'.", "grammar"),
        (r"The both boys are friends", "Both boys are friends", "Remove 'the' before 'both'.", "grammar"),
        (r"She married with a doctor", "She married a doctor", "Incorrect preposition. Use 'married a doctor'.", "grammar"),
        (r"He prefer tea than coffee", "He prefers tea to coffee", "Incorrect verb form and preposition. Use 'prefers tea to coffee'.", "grammar"),
        (r"We was played cricket yesterday", "We played cricket yesterday", "Incorrect verb form. Use 'played'.", "grammar"),
        (r"The work is more easy now", "The work is easier now", "Incorrect comparative. Use 'easier'.", "grammar"),
        (r"I am having a doubt", "I have a question", "Use 'have a question' instead of 'having a doubt'.", "grammar"),
        (r"He is very more intelligent", "He is much more intelligent", "Use 'much more intelligent'.", "grammar"),
        (r"You was sleeping during class", "You were sleeping during class", "Incorrect verb form. Use 'were' with 'you'.", "grammar"),
        (r"I cannot able to do it", "I cannot do it", "Redundant modal. Use 'cannot do it'.", "grammar"),
        (r"\bI am studying in the university\b", "I am studying at the university", "Incorrect preposition. Use 'at' with 'university'.", "grammar"),
    (r"\bShe borned in 2001\b", "She was born in 2001", "Incorrect verb form. Use 'was born'.", "grammar"),
    (r"\bHe haven’t arrived yet\b", "He hasn’t arrived yet", "Incorrect verb form. Use 'hasn’t' with 'he'.", "grammar"),
    (r"\bThey speaks very well\b", "They speak very well", "Incorrect verb form. Use 'speak' with 'they'.", "grammar"),
    (r"\bIs raining outside\b", "It is raining outside", "Missing subject. Use 'It is raining'.", "grammar"),
    (r"\bI likes ice cream\b", "I like ice cream", "Incorrect verb form. Use 'like' with 'I'.", "grammar"),
    (r"\bHe always arrive late\b", "He always arrives late", "Incorrect verb form. Use 'arrives' with 'he'.", "grammar"),
    (r"\bI am having headache\b", "I have a headache", "Use 'have a headache'.", "grammar"),
    (r"\bShe don’t study hard\b", "She doesn’t study hard", "Incorrect verb form. Use 'doesn’t study'.", "grammar"),
    (r"\bWe no went to the park\b", "We didn’t go to the park", "Incorrect negative past. Use 'didn’t go'.", "grammar"),
    (r"\bShe is married to a engineer\b", "She is married to an engineer", "Incorrect article. Use 'an' before vowel sound.", "grammar"),
    (r"\bIt’s too much hot today\b", "It’s too hot today", "Incorrect quantifier. Use 'too hot'.", "grammar"),
    (r"\bHe is more taller than me\b", "He is taller than me", "Redundant comparative. Use 'taller'.", "clarity"),
    (r"\bThe child have many toys\b", "The child has many toys", "Incorrect verb form. Use 'has' with 'child'.", "grammar"),
    (r"\bWhere are you going at\b", "Where are you going?", "Do not use 'at' after 'going'.", "grammar"),
    (r"\bI am going to home\b", "I am going home", "Do not use 'to' before 'home'.", "grammar"),
    (r"\bThis dress is more prettier\b", "This dress is prettier", "Redundant comparative. Use 'prettier'.", "clarity"),
    (r"\bWe was listening to music\b", "We were listening to music", "Incorrect verb form. Use 'were' with 'we'.", "grammar"),
    (r"\bI thinked about it\b", "I thought about it", "Incorrect past tense. Use 'thought'.", "grammar"),
    (r"\bThey not want to participate\b", "They do not want to participate", "Incorrect negative. Use 'do not want'.", "grammar"),
    (r"\bHe drinks coffee on every morning\b", "He drinks coffee every morning", "Do not use 'on' before 'every morning'.", "grammar"),
    (r"\bYou has to try this\b", "You have to try this", "Incorrect verb form. Use 'have' with 'you'.", "grammar"),
    (r"\bThat’s belong to me\b", "That belongs to me", "Incorrect verb form. Use 'belongs'.", "grammar"),
    (r"\bHe is very much talented\b", "He is very talented", "Redundant quantifier. Use 'very talented'.", "grammar"),
    (r"\bI can to do it\b", "I can do it", "Do not use 'to' after 'can'.", "grammar"),
    (r"\bThey was happy for see us\b", "They were happy to see us", "Incorrect verb form and preposition. Use 'were happy to see'.", "grammar"),
    (r"\bShe did a accident\b", "She had an accident", "Incorrect verb and article. Use 'had an accident'.", "grammar"),
    (r"\bHe not understand the question\b", "He does not understand the question", "Incorrect negative. Use 'does not understand'.", "grammar"),
    (r"\bThis is the more important topic\b", "This is the most important topic", "Incorrect superlative. Use 'most important'.", "grammar"),
    (r"\bI am not agree with him\b", "I do not agree with him", "Incorrect negative. Use 'do not agree'.", "grammar"),
    (r"\bHe have many books\b", "He has many books", "Incorrect verb form. Use 'has' with 'he'.", "grammar"),
        (r"Their going to the market now", "They're going to the market now", "Incorrect word. Use 'they're' for 'they are'.", "grammar"),
    (r"I has completed the work", "I have completed the work", "Incorrect verb form. Use 'have' with 'I'.", "grammar"),
    (r"The dog bark loud in the night", "The dog barks loudly at night", "Incorrect verb form and adverb. Use 'barks loudly at night'.", "grammar"),
    (r"Its a beautiful day, isn’t it\?", "It's a beautiful day, isn't it?", "Missing apostrophe and question mark. Use 'It's' and proper punctuation.", "punctuation"),
    (r"Your the best player here", "You're the best player here", "Incorrect word. Use 'you're' for 'you are'.", "grammar"),
    (r"I am interesting in learning AI", "I am interested in learning AI", "Incorrect adjective. Use 'interested'.", "grammar"),
    (r"They was happy with the results", "They were happy with the results", "Incorrect verb form. Use 'were' with 'they'.", "grammar"),
    # Punctuation & Capitalization
    (r"^i went to london last week", "I went to London last week", "Capitalize 'I' and 'London'.", "capitalization"),
    (r"Let’s eat grandma!", "Let’s eat, grandma!", "Missing comma changes meaning. Add comma after 'eat'.", "punctuation"),
    (r"Do you know where my phone is$", "Do you know where my phone is?", "Missing question mark at the end.", "punctuation"),
    (r"^she said she will come later\.", "She said she will come later.", "Capitalize first word of the sentence.", "capitalization"),
    (r"He said “I’m tired”.", "He said, \"I'm tired.\"", "Use proper quotation marks and comma before quote.", "punctuation"),
    # Run-on Sentences & Fragments (simple cases)
    (r"He studied hard he passed the exam\.", "He studied hard, and he passed the exam.", "Run-on sentence. Use a comma and conjunction.", "grammar"),
    (r"The weather was nice we went for a picnic\.", "The weather was nice, so we went for a picnic.", "Run-on sentence. Use a comma and conjunction.", "grammar"),
    (r"I like pizza I don’t like burgers\.", "I like pizza, but I don’t like burgers.", "Run-on sentence. Use a comma and conjunction.", "grammar"),
    # Fragments
    (r"^Although she was tired\.", "Although she was tired, she finished her work.", "Fragment. Complete the sentence.", "grammar"),
    (r"^Because I forgot my umbrella\.", "Because I forgot my umbrella, I got wet.", "Fragment. Complete the sentence.", "grammar"),
    # Misused Words
    (r"Their going to the park after dinner\.", "They're going to the park after dinner.", "Incorrect word. Use 'they're' for 'they are'.", "grammar"),
    (r"Your welcome to join us\.", "You're welcome to join us.", "Incorrect word. Use 'you're' for 'you are'.", "grammar"),
    (r"I have less friends than him\.", "I have fewer friends than he does.", "Use 'fewer' with countable nouns and 'than he does'.", "grammar"),
    (r"There is many problems with the code\.", "There are many problems with the code.", "Incorrect verb form. Use 'are' with plural 'problems'.", "grammar"),
    (r"I except your invitation\.", "I accept your invitation.", "Incorrect word. Use 'accept' for agreement.", "grammar"),
    (r'\bI writed\b', "I wrote", "Incorrect verb form. Use 'wrote' as the past tense of 'write'.", "grammar")
    ]
    for pattern, correction, explanation, type_ in patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            start_pos = match.start()
            suggestions.append(AISuggestion(
                id=f"fallback_{type_}_{len(suggestions)}",
                type=type_,
                category=type_,
                original_text=match.group(0),
                suggested_text=correction,
                explanation=explanation,
                confidence=0.9,
                position={"start": start_pos, "end": start_pos + len(match.group(0))},
                severity="medium"
            ))
    # Punctuation: double spaces
    for match in re.finditer(r'  +', content):
        start_pos = match.start()
        suggestions.append(AISuggestion(
            id=f"fallback_punct_{len(suggestions)}",
            type="punctuation",
            category="punctuation",
            original_text=match.group(0),
            suggested_text=" ",
            explanation="Multiple spaces found. Use a single space.",
            confidence=0.8,
            position={"start": start_pos, "end": start_pos + len(match.group(0))},
            severity="low"
        ))

    return suggestions

async def analyze_text_with_ai(content: str) -> AIAnalytics:
    """Analyze text using AI for various metrics"""
    try:
        # Calculate basic metrics
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Readability score (simplified Flesch Reading Ease)
        if sentences and words:
            avg_sentence_length = len(words) / len(sentences)
            readability_score = max(0, min(100, 100 - (avg_sentence_length * 1.5)))
        else:
            readability_score = 50
        
        # Sentiment analysis (simplified)
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
        
        positive_count = sum(1 for word in words if word.lower() in positive_words)
        negative_count = sum(1 for word in words if word.lower() in negative_words)
        
        if words:
            sentiment_score = (positive_count - negative_count) / len(words) * 100
        else:
            sentiment_score = 0
        
        # Tone analysis
        formal_words = ["therefore", "furthermore", "consequently", "utilize", "facilitate"]
        informal_words = ["gonna", "wanna", "gotta", "cool", "awesome"]
        
        formal_count = sum(1 for word in words if word.lower() in formal_words)
        informal_count = sum(1 for word in words if word.lower() in informal_words)
        
        tone_analysis = {
            "formal": formal_count / max(len(words), 1) * 100,
            "informal": informal_count / max(len(words), 1) * 100,
            "neutral": 100 - (formal_count + informal_count) / max(len(words), 1) * 100
        }
        
        # Complexity score
        unique_words = len(set(words))
        complexity_score = (unique_words / max(len(words), 1)) * 100
        
        # Engagement score
        question_count = content.count('?')
        exclamation_count = content.count('!')
        engagement_score = min(100, (question_count + exclamation_count) * 10)
        
        # Word diversity
        word_diversity = (unique_words / max(len(words), 1)) * 100
        
        # Sentence variety
        sentence_lengths = [len(s.split()) for s in sentences]
        if sentence_lengths:
            sentence_variety = (max(sentence_lengths) - min(sentence_lengths)) / max(max(sentence_lengths), 1) * 100
        else:
            sentence_variety = 0
        
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
        logger.error(f"Error analyzing text: {e}")
        return AIAnalytics(
            readability_score=50,
            sentiment_score=0,
            tone_analysis={"formal": 0, "informal": 0, "neutral": 100},
            complexity_score=50,
            engagement_score=0,
            word_diversity=50,
            sentence_variety=0
        )

def calculate_text_stats(content: str) -> Dict[str, Any]:
    """Calculate basic text statistics"""
    words = content.split()
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return {
        "word_count": len(words),
        "character_count": len(content),
        "sentence_count": len(sentences),
        "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
        "average_words_per_sentence": len(words) / max(len(sentences), 1),
        "reading_time_minutes": len(words) / 200,  # Average reading speed
        "unique_words": len(set(words)),
        "vocabulary_diversity": len(set(words)) / max(len(words), 1)
    }

def calculate_document_score(analytics: AIAnalytics) -> int:
    """Calculate overall document score based on analytics"""
    try:
        # Weighted scoring based on different metrics
        readability_weight = 0.25
        engagement_weight = 0.20
        diversity_weight = 0.20
        variety_weight = 0.15
        sentiment_weight = 0.20
        
        score = (
            analytics.readability_score * readability_weight +
            analytics.engagement_score * engagement_weight +
            analytics.word_diversity * diversity_weight +
            analytics.sentence_variety * variety_weight +
            (analytics.sentiment_score + 50) * sentiment_weight  # Normalize sentiment to 0-100
        )
        
        return max(0, min(100, int(score)))
    except Exception as e:
        logger.error(f"Error calculating document score: {e}")
        return 50

async def store_analytics(user_id: str, document_id: str, analytics: AIAnalytics):
    """Store analytics data for a document"""
    try:
        analytics_db[f"{user_id}_{document_id}"] = {
            "user_id": user_id,
            "document_id": document_id,
            "analytics": analytics.dict(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error storing analytics: {e}")

# Plagiarism check endpoint
@app.post("/api/ai/plagiarism/check", response_model=PlagiarismResponse)
async def check_plagiarism_ai(request: PlagiarismRequest):
    """Check content for plagiarism"""
    start_time = datetime.now()
    
    try:
        matches = []
        
        if request.check_web:
            web_matches = await check_web_sources(request.content)
            matches.extend(web_matches)
        
        if request.check_academic:
            academic_matches = await check_academic_sources(request.content)
            matches.extend(academic_matches)
        
        # Calculate overall similarity score
        if matches:
            overall_score = sum(match.similarity for match in matches) / len(matches)
        else:
            overall_score = 0
        
        # Determine risk level
        if overall_score > 80:
            risk_level = "high"
        elif overall_score > 50:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PlagiarismResponse(
            overall_score=overall_score,
            risk_level=risk_level,
            matches=matches,
            processing_time=processing_time,
            sources_checked=len(matches)
        )
        
    except Exception as e:
        logger.error(f"Error checking plagiarism: {e}")
        raise HTTPException(status_code=500, detail="Failed to check plagiarism")

async def check_web_sources(content: str) -> List[PlagiarismMatch]:
    """Check web sources for plagiarism (simplified)"""
    # This is a simplified implementation
    # In production, you would integrate with actual plagiarism detection services
    matches = []
    
    # Simulate checking against common phrases
    common_phrases = [
        "The quick brown fox jumps over the lazy dog",
        "To be or not to be, that is the question",
        "All the world's a stage"
    ]
    
    for i, phrase in enumerate(common_phrases):
        if phrase.lower() in content.lower():
            similarity = len(phrase) / len(content) * 100
            if similarity > 10:  # Only report if similarity is significant
                matches.append(PlagiarismMatch(
                    id=f"web_match_{i}",
                    source="Web Source",
                    similarity=similarity,
                    matched_text=phrase,
                    source_text=phrase,
                    url="https://example.com",
                    type="web",
                    confidence=0.8
                ))
    
    return matches

async def check_academic_sources(content: str) -> List[PlagiarismMatch]:
    """Check academic sources for plagiarism (simplified)"""
    # This is a simplified implementation
    # In production, you would integrate with academic databases
    matches = []
    
    # Simulate checking against academic papers
    academic_phrases = [
        "The results indicate a significant correlation",
        "Previous research has shown",
        "This study demonstrates"
    ]
    
    for i, phrase in enumerate(academic_phrases):
        if phrase.lower() in content.lower():
            similarity = len(phrase) / len(content) * 100
            if similarity > 10:
                matches.append(PlagiarismMatch(
                    id=f"academic_match_{i}",
                    source="Academic Database",
                    similarity=similarity,
                    matched_text=phrase,
                    source_text=phrase,
                    url="https://scholar.google.com",
                    type="academic",
                    confidence=0.9
                ))
    
    return matches

def calculate_similarity_score(text: str) -> float:
    """Calculate similarity score between texts (simplified)"""
    # This is a simplified implementation
    # In production, you would use more sophisticated algorithms
    words = set(text.lower().split())
    return len(words) / max(len(text.split()), 1) * 100

# Writing insights endpoint
@app.post("/api/ai/insights", response_model=InsightResponse)
async def get_writing_insights(request: InsightRequest):
    """Get AI-powered writing insights"""
    try:
        # Get user's documents
        user_docs = [doc for doc in documents_db.values() if doc.get("user_id") == request.user_id]
        
        if not user_docs:
            return InsightResponse(
                insights=[],
                performance_metrics={},
                improvement_areas=[],
                achievements=[]
            )
        
        # Generate insights
        insights = await generate_ai_insights(user_docs, request.time_range)
        
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
        logger.error(f"Error getting writing insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get writing insights")

async def generate_ai_insights(documents: List[Dict], time_range: str) -> List[WritingInsight]:
    """Generate AI-powered writing insights"""
    insights = []
    
    if not documents:
        return insights
    
    # Analyze writing patterns
    total_words = sum(doc.get("word_count", 0) for doc in documents)
    avg_score = sum(doc.get("score", 0) for doc in documents) / len(documents)
    
    # Insight 1: Writing volume
    if total_words > 1000:
        insights.append(WritingInsight(
            type="productivity",
            title="High Writing Volume",
            description=f"You've written {total_words} words across {len(documents)} documents.",
            impact="This shows strong writing consistency and dedication to your craft.",
            recommendation="Consider setting daily writing goals to maintain this momentum."
        ))
    
    # Insight 2: Quality improvement
    if avg_score > 80:
        insights.append(WritingInsight(
            type="quality",
            title="Excellent Writing Quality",
            description=f"Your average document score is {avg_score:.1f}/100.",
            impact="Your writing demonstrates high quality and attention to detail.",
            recommendation="Focus on maintaining this high standard while exploring new writing styles."
        ))
    
    # Insight 3: Consistency
    recent_docs = sorted(documents, key=lambda x: x.get("last_modified", ""), reverse=True)[:5]
    if len(recent_docs) >= 3:
        insights.append(WritingInsight(
            type="consistency",
            title="Consistent Writing Habit",
            description="You've been writing regularly with good consistency.",
            impact="Regular writing practice improves skills and builds momentum.",
            recommendation="Try to maintain this consistency and consider daily writing sessions."
        ))
    
    return insights

def calculate_performance_metrics(documents: List[Dict]) -> Dict[str, Any]:
    """Calculate performance metrics from documents"""
    if not documents:
        return {}
    
    total_words = sum(doc.get("word_count", 0) for doc in documents)
    avg_score = sum(doc.get("score", 0) for doc in documents) / len(documents)
    
    # Calculate writing frequency
    dates = [doc.get("last_modified", "") for doc in documents]
    if dates:
        latest_date = max(dates)
        earliest_date = min(dates)
        days_span = (datetime.fromisoformat(latest_date) - datetime.fromisoformat(earliest_date)).days
        writing_frequency = len(documents) / max(days_span, 1)
    else:
        writing_frequency = 0
    
    return {
        "total_documents": len(documents),
        "total_words": total_words,
        "average_score": round(avg_score, 1),
        "writing_frequency": round(writing_frequency, 2),
        "best_score": max(doc.get("score", 0) for doc in documents),
        "average_words_per_document": total_words / len(documents)
    }

def identify_improvement_areas(documents: List[Dict]) -> List[str]:
    """Identify areas for improvement"""
    areas = []
    
    if not documents:
        return areas
    
    avg_score = sum(doc.get("score", 0) for doc in documents) / len(documents)
    
    if avg_score < 70:
        areas.append("Overall writing quality needs improvement")
    
    word_counts = [doc.get("word_count", 0) for doc in documents]
    avg_words = sum(word_counts) / len(word_counts)
    
    if avg_words < 100:
        areas.append("Consider writing longer, more detailed content")
    
    if len(documents) < 5:
        areas.append("Increase writing frequency for better skill development")
    
    return areas

def generate_achievements(documents: List[Dict]) -> List[Dict[str, Any]]:
    """Generate achievements based on writing performance"""
    achievements = []
    
    if not documents:
        return achievements
    
    total_words = sum(doc.get("word_count", 0) for doc in documents)
    avg_score = sum(doc.get("score", 0) for doc in documents) / len(documents)
    
    # Word count achievements
    if total_words >= 1000:
        achievements.append({
            "title": "Word Warrior",
            "description": f"Wrote {total_words} words",
            "icon": "📝",
            "unlocked": True
        })
    
    if total_words >= 5000:
        achievements.append({
            "title": "Prolific Writer",
            "description": f"Wrote {total_words} words",
            "icon": "✍️",
            "unlocked": True
        })
    
    # Quality achievements
    if avg_score >= 80:
        achievements.append({
            "title": "Quality Master",
            "description": f"Average score: {avg_score:.1f}/100",
            "icon": "🏆",
            "unlocked": True
        })
    
    # Consistency achievements
    if len(documents) >= 10:
            achievements.append({
            "title": "Consistent Writer",
            "description": f"Created {len(documents)} documents",
            "icon": "📚",
            "unlocked": True
            })
    
    return achievements

# User statistics endpoint
@app.get("/api/users/{user_id}/statistics")
async def get_user_statistics(user_id: str, days: int = 7):
    """Get user writing statistics"""
    try:
        # Get user's documents
        user_docs = [doc for doc in documents_db.values() if doc.get("user_id") == user_id]
        
        if not user_docs:
            return {
                "total_documents": 0,
                "total_words": 0,
                "average_score": 0,
                "writing_streak": 0,
                "recent_activity": []
            }
        
        # Calculate statistics
        total_documents = len(user_docs)
        total_words = sum(doc.get("word_count", 0) for doc in user_docs)
        average_score = sum(doc.get("score", 0) for doc in user_docs) / len(user_docs)
        
        # Calculate writing streak
        dates = [datetime.fromisoformat(doc.get("last_modified", "")) for doc in user_docs]
        dates.sort(reverse=True)
        
        streak = 0
        current_date = datetime.now().date()
        
        for date in dates:
            if date.date() == current_date - timedelta(days=streak):
                streak += 1
            else:
                break
        
        # Recent activity
        recent_activity = []
        for doc in sorted(user_docs, key=lambda x: x.get("last_modified", ""), reverse=True)[:5]:
            recent_activity.append({
                "title": doc.get("title", ""),
                "word_count": doc.get("word_count", 0),
                "score": doc.get("score", 0),
                "date": doc.get("last_modified", "")
            })
        
        return {
            "total_documents": total_documents,
            "total_words": total_words,
            "average_score": round(average_score, 1),
            "writing_streak": streak,
            "recent_activity": recent_activity
        }
        
    except Exception as e:
        logger.error(f"Error getting user statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user statistics")

def analyze_grammar_context(content: str, word: str, position: int) -> str:
    """Analyze grammar context around a word"""
    # This is a simplified implementation
    # In production, you would use more sophisticated NLP libraries
    words = content.split()
    if position < len(words):
        word_at_position = words[position]
        if word_at_position.lower() == word.lower():
            # Basic context analysis
            if position > 0:
                prev_word = words[position - 1]
                if prev_word.lower() in ["a", "an", "the"]:
                    return f"Article '{prev_word}' suggests this should be a noun"
            if position < len(words) - 1:
                next_word = words[position + 1]
                if next_word.lower().endswith("ing"):
                    return f"Following word '{next_word}' suggests this should be a verb"
    return "Context analysis not available"

@app.post("/api/ai/rewrite", response_model=RewriteResponse)
async def rewrite_text(request: RewriteRequest):
    """Rewrite text to match the given goal (e.g., formal, casual, marketing, friendly)"""
    try:
        # Try Groq API
        rewritten = None
        if GROQ_API_KEY != "your-groq-api-key-here":
            groq_url = "https://api.groq.com/openai/v1/chat/completions"
            prompt = f"""Rewrite the following text to be more {request.goal}.\n\nText: \"{request.content}\"\n\nRewritten ({request.goal}):"""
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            async with aiohttp.ClientSession() as session:
                response = await session.post(groq_url, headers=headers, json={
                    "model": "llama3-70b-8192",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800
                }, timeout=20)
                if response.status == 200:
                    data = await response.json()
                    rewritten = data["choices"][0]["message"]["content"].strip()
        # Fallback
        if not rewritten:
            rewritten = f"[{request.goal.capitalize()}] {request.content}"
        return RewriteResponse(rewritten_text=rewritten)
    except Exception as e:
        logger.error(f"Error rewriting text: {e}")
        return RewriteResponse(rewritten_text=request.content)

@app.post("/api/ai/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """Summarize the given text using AI or fallback."""
    try:
        summary = None
        if GROQ_API_KEY != "your-groq-api-key-here":
            groq_url = "https://api.groq.com/openai/v1/chat/completions"
            prompt = f"""Please provide a concise summary of the following text in no more than 100 words:\n\nText: \"{request.content}\"\n\nSummary:"""
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            async with aiohttp.ClientSession() as session:
                response = await session.post(groq_url, headers=headers, json={
                    "model": "llama3-70b-8192",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 200
                }, timeout=20)
                if response.status == 200:
                    data = await response.json()
                    summary = data["choices"][0]["message"]["content"].strip()
        # Fallback
        if not summary:
            sentences = re.split(r'[.!?]+', request.content)
            sentences = [s.strip() for s in sentences if s.strip()]
            summary = '. '.join(sentences[:2]) + ('.' if sentences else '')
        return SummarizeResponse(summary=summary)
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        return SummarizeResponse(summary="Summary not available.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 