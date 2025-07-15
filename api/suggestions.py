# FastAPI Backend - Suggestions API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
from groq import Groq
from transformers import pipeline
import re

app = FastAPI()

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Initialize Hugging Face models
grammar_checker = pipeline("text-classification", model="textattack/roberta-base-CoLA")
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

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

@app.post("/api/suggestions", response_model=SuggestionResponse)
async def get_suggestions(text_input: TextInput):
    """
    Get writing suggestions using Groq API and Hugging Face models
    """
    try:
        content = text_input.content
        goal = text_input.goal
        
        # Grammar checking using Hugging Face
        grammar_issues = await check_grammar(content)
        
        # Get suggestions from Groq
        groq_suggestions = await get_groq_suggestions(content, goal)
        
        # Sentiment analysis for tone suggestions
        tone_suggestions = await analyze_tone(content)
        
        # Combine all suggestions
        all_suggestions = grammar_issues + groq_suggestions + tone_suggestions
        
        # Calculate statistics
        stats = calculate_stats(content)
        
        return SuggestionResponse(
            suggestions=all_suggestions,
            stats=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def check_grammar(text: str) -> List[Suggestion]:
    """Check grammar using Hugging Face model"""
    suggestions = []
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    for i, sentence in enumerate(sentences):
        if len(sentence.strip()) > 0:
            # Check grammar using CoLA model
            result = grammar_checker(sentence.strip())
            
            if result[0]['label'] == 'UNACCEPTABLE' and result[0]['score'] > 0.7:
                suggestions.append(Suggestion(
                    id=f"grammar_{i}",
                    type="grammar",
                    text=sentence.strip(),
                    suggestion="Review grammar in this sentence",
                    explanation="This sentence may have grammatical issues",
                    position={"start": text.find(sentence), "end": text.find(sentence) + len(sentence)}
                ))
    
    return suggestions

async def get_groq_suggestions(text: str, goal: str) -> List[Suggestion]:
    """Get suggestions from Groq API"""
    suggestions = []
    
    try:
        prompt = f"""
        Analyze the following text for {goal} improvements and provide specific suggestions:
        
        Text: "{text}"
        
        Please provide suggestions in the format:
        - Issue: [specific text]
        - Suggestion: [improved version]
        - Explanation: [why this is better]
        
        Focus on {goal} improvements.
        """
        
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            max_tokens=1000,
            temperature=0.3
        )
        
        # Parse Groq response (simplified)
        content = response.choices[0].message.content
        
        # Extract suggestions (this is a simplified parser)
        suggestion_blocks = content.split('- Issue:')
        
        for i, block in enumerate(suggestion_blocks[1:], 1):
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                issue_text = lines[0].strip()
                suggestion_text = lines[1].replace('- Suggestion:', '').strip()
                explanation = lines[2].replace('- Explanation:', '').strip()
                
                suggestions.append(Suggestion(
                    id=f"groq_{i}",
                    type="clarity",
                    text=issue_text,
                    suggestion=suggestion_text,
                    explanation=explanation,
                    position={"start": text.find(issue_text), "end": text.find(issue_text) + len(issue_text)}
                ))
    
    except Exception as e:
        print(f"Groq API error: {e}")
    
    return suggestions

async def analyze_tone(text: str) -> List[Suggestion]:
    """Analyze tone using Hugging Face sentiment analysis"""
    suggestions = []
    
    try:
        result = sentiment_analyzer(text)
        
        if result[0]['label'] == 'LABEL_0' and result[0]['score'] > 0.7:  # Negative
            suggestions.append(Suggestion(
                id="tone_1",
                type="tone",
                text=text[:100] + "..." if len(text) > 100 else text,
                suggestion="Consider using more positive language",
                explanation="The tone appears negative. Consider rephrasing for better engagement",
                position={"start": 0, "end": min(100, len(text))}
            ))
    
    except Exception as e:
        print(f"Tone analysis error: {e}")
    
    return suggestions

def calculate_stats(text: str) -> dict:
    """Calculate text statistics"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    paragraphs = text.split('\n\n')
    
    return {
        "wordCount": len(words),
        "characters": len(text),
        "sentences": len([s for s in sentences if s.strip()]),
        "paragraphs": len([p for p in paragraphs if p.strip()]),
        "readingTime": max(1, len(words) // 250),
        "readabilityScore": min(100, max(0, 100 - len(words) // 10))  # Simplified score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)