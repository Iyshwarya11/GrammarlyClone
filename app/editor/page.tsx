'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { 
  Save, 
  Download, 
  CheckCircle,
  AlertCircle,
  Lightbulb,
  ArrowLeft,
  Undo,
  Redo,
  Bold,
  Italic,
  Underline,
  Copy,
  BarChart3,
  FileText,
  Clock,
  Star,
  Settings,
  Maximize2,
  Minimize2,
  Eye,
  EyeOff,
  Zap
} from 'lucide-react';
import Link from 'next/link';

interface Suggestion {
  id: string;
  type: string;
  original_text: string;
  suggested_text: string;
  explanation: string;
  confidence: number;
  severity: string;
  position?: { start: number; end: number };
}

interface DocumentStats {
  wordCount: number;
  characters: number;
  sentences: number;
  readabilityScore: number;
  tone: string;
  overallScore: number;
}

interface DocumentHistory {
  id: string;
  title: string;
  content: string;
  lastModified: string;
  wordCount: number;
  score: number;
}

export default function Editor() {
  // Editor state
  const [content, setContent] = useState('Write your content here and see real-time AI suggestions appear in the sidebar. This is a sample text that contains some spelling mistakes like recieve and definately. I am going to the store yesterday. She have been working hard.');
  const [title, setTitle] = useState('Untitled Document');
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [stats, setStats] = useState<DocumentStats>({
    wordCount: 0,
    characters: 0,
    sentences: 0,
    readabilityScore: 0,
    tone: 'neutral',
    overallScore: 0
  });
  
  // UI state
  const [isLoading, setIsLoading] = useState(false);
  const [isDirty, setIsDirty] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [undoStack, setUndoStack] = useState<string[]>([]);
  const [redoStack, setRedoStack] = useState<string[]>([]);
  const [documentHistory, setDocumentHistory] = useState<DocumentHistory[]>([]);
  const [showHistory, setShowHistory] = useState(true);
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [layoutMode, setLayoutMode] = useState<'default' | 'focus' | 'minimal'>('default');
  const [showStats, setShowStats] = useState(true);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [suggestionFilters, setSuggestionFilters] = useState<Set<string>>(
    new Set(['spelling', 'grammar', 'tense', 'clarity', 'style', 'punctuation', 'formal', 'casual', 'marketing', 'friendly', 'general'])
  );
  const [aiStatus, setAiStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [goal, setGoal] = useState('clarity');
  const goalOptions = [
    { value: 'formal', label: 'Formal' },
    { value: 'casual', label: 'Casual' },
    { value: 'marketing', label: 'Marketing' },
    { value: 'friendly', label: 'Friendly' },
    { value: 'clarity', label: 'Clarity' },
  ];

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const titleInputRef = useRef<HTMLInputElement>(null);
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<any>(null);

  const [chainingResults, setChainingResults] = useState<null | {
    corrected: string;
    rewritten: string;
    summary: string;
  }>(null);
  const [chainingLoading, setChainingLoading] = useState(false);
  const [chainingError, setChainingError] = useState<string | null>(null);

  // Voice-to-text logic
  const startListening = () => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      alert('Speech recognition is not supported in this browser. Please use Chrome or Edge.');
      return;
    }
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setContent(prev => prev + (prev && !prev.endsWith(' ') ? ' ' : '') + transcript);
    };
    recognition.onend = () => setIsListening(false);
    recognition.onerror = () => setIsListening(false);
    recognitionRef.current = recognition;
    setIsListening(true);
    recognition.start();
  };
  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  };

  // Enhanced AI suggestions fetching
  const fetchSuggestions = useCallback(async () => {
    if (content.length > 10) {
      try {
        setIsLoading(true);
        setAiStatus('loading');
        
        const response = await fetch('http://localhost:8000/api/ai/suggestions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            content, 
            goal, 
            tone: 'professional', 
            audience: 'general' 
          }),
        });

        if (response.ok) {
          const data = await response.json();
          const newSuggestions = data.suggestions || [];
          
          // Filter suggestions based on user preferences
          const filteredSuggestions = newSuggestions.filter((s: Suggestion) => 
            suggestionFilters.has(s.type)
          );
          
          setSuggestions(filteredSuggestions);
          setAiStatus('success');
          
          // Update overall score based on analytics
          if (data.analytics) {
            const score = Math.round(
              (data.analytics.readability_score * 0.3) +
              (data.analytics.engagement_score * 0.3) +
              (data.analytics.word_diversity * 0.2) +
              (data.analytics.sentence_variety * 0.2)
            );
            setStats(prev => ({ ...prev, overallScore: score }));
          }
        } else {
          setAiStatus('error');
          console.error('Failed to fetch suggestions:', response.status);
        }
      } catch (error) {
        console.error('Error fetching suggestions:', error);
        setSuggestions([]);
        setAiStatus('error');
      } finally {
        setIsLoading(false);
      }
    }
  }, [content, suggestionFilters, goal]);

  // Calculate document statistics
  const calculateStats = useCallback(() => {
    const words = content.trim().split(/\s+/).filter(word => word.length > 0);
    const sentences = content.split(/[.!?]+/).filter(sentence => sentence.trim().length > 0);
    
    // Enhanced readability score
    const avgWordsPerSentence = words.length / Math.max(sentences.length, 1);
    const readabilityScore = Math.max(0, Math.min(100, 100 - (avgWordsPerSentence * 1.5)));
    
    // Enhanced tone detection
    const formalWords = ['therefore', 'furthermore', 'consequently', 'utilize', 'facilitate', 'subsequently'];
    const informalWords = ['gonna', 'wanna', 'gotta', 'cool', 'awesome', 'yeah'];
    const formalCount = formalWords.filter(word => content.toLowerCase().includes(word)).length;
    const informalCount = informalWords.filter(word => content.toLowerCase().includes(word)).length;
    
    let tone = 'neutral';
    if (informalCount > formalCount) tone = 'informal';
    else if (formalCount > informalCount) tone = 'formal';

    // Calculate overall score with better weighting
    const errorPenalty = suggestions.length * 3; // Reduced penalty per suggestion
    const baseScore = Math.max(0, 100 - errorPenalty);
    
    setStats({
      wordCount: words.length,
      characters: content.length,
      sentences: sentences.length,
      readabilityScore: Math.round(readabilityScore),
      tone,
      overallScore: Math.max(0, Math.min(100, baseScore))
    });
  }, [content, suggestions.length]);

  // Auto-save document to history
  const autoSaveDocument = useCallback(() => {
    if (content.trim() && title.trim()) {
      const newDocument: DocumentHistory = {
        id: Date.now().toString(),
        title: title,
        content: content,
        lastModified: new Date().toISOString(),
        wordCount: stats.wordCount,
        score: stats.overallScore
      };

      setDocumentHistory(prev => {
        // Remove existing document with same title if exists
        const filtered = prev.filter(doc => doc.title !== title);
        return [newDocument, ...filtered.slice(0, 9)]; // Keep only 10 most recent
      });

      setIsDirty(false);
    }
  }, [content, title, stats.wordCount, stats.overallScore]);

  // Load document from history
  const loadDocument = (doc: DocumentHistory) => {
    setTitle(doc.title);
    setContent(doc.content);
    setIsDirty(false);
  };

  // Enhanced suggestion application with better error prevention
  const applySuggestion = (suggestion: Suggestion) => {
    // Normalize quotes and apostrophes
    let originalText = suggestion.original_text.replace(/^"|"$/g, '').replace(/[’‘]/g, "'").trim();
    let suggestedText = suggestion.suggested_text.replace(/^"|"$/g, '').replace(/[’‘]/g, "'").trim();

    if (!originalText || !suggestedText) {
      console.warn('Invalid suggestion: missing original or suggested text');
      return;
    }

    // Build a regex for whole word, case-insensitive, normalized apostrophes
    const regex = new RegExp(`\\b${originalText.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')}\\b`, 'i');
    const match = content.match(regex);

    if (!match) {
      console.warn('Original text not found in content');
      return;
    }

    // Replace only the first occurrence, preserving the rest of the content
    const newContent = content.replace(regex, suggestedText);

    setUndoStack(prev => [...prev, content]);
    setContent(newContent);
    setIsDirty(true);
    setSuggestions(prev => prev.filter(s => s.id !== suggestion.id));
    setTimeout(autoSaveDocument, 1000);
  };

  // Undo/Redo functions
  const undo = () => {
    if (undoStack.length > 0) {
      const previousContent = undoStack[undoStack.length - 1];
      setRedoStack(prev => [...prev, content]);
      setUndoStack(prev => prev.slice(0, -1));
      setContent(previousContent);
    }
  };

  const redo = () => {
    if (redoStack.length > 0) {
      const nextContent = redoStack[redoStack.length - 1];
      setUndoStack(prev => [...prev, content]);
      setRedoStack(prev => prev.slice(0, -1));
      setContent(nextContent);
    }
  };

  // Text formatting
  const formatText = (format: string) => {
    if (!selectedText) return;
    
    let formattedText = '';
    switch (format) {
      case 'bold': formattedText = `**${selectedText}**`; break;
      case 'italic': formattedText = `*${selectedText}*`; break;
      case 'underline': formattedText = `__${selectedText}__`; break;
      default: return;
    }
    
    const newContent = content.replace(selectedText, formattedText);
    setUndoStack(prev => [...prev, content]);
    setContent(newContent);
    setIsDirty(true);
  };

  // Copy and download functions
  const copyDocument = () => {
    navigator.clipboard.writeText(content);
  };

  const downloadDocument = () => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Handle text selection
  const handleTextSelection = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const selected = content.substring(start, end);
      setSelectedText(selected);
    }
  };

  // Handle title editing
  const handleTitleEdit = () => {
    setIsEditingTitle(true);
    setTimeout(() => {
      if (titleInputRef.current) {
        titleInputRef.current.focus();
        titleInputRef.current.select();
      }
    }, 100);
  };

  const handleTitleSave = () => {
    setIsEditingTitle(false);
    setIsDirty(true);
  };

  const handleTitleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleTitleSave();
    } else if (e.key === 'Escape') {
      setIsEditingTitle(false);
    }
  };

  // Toggle suggestion filters
  const toggleSuggestionFilter = (type: string) => {
    setSuggestionFilters(prev => {
      const newFilters = new Set(prev);
      if (newFilters.has(type)) {
        newFilters.delete(type);
      } else {
        newFilters.add(type);
      }
      return newFilters;
    });
  };

  // Effects
  useEffect(() => {
    calculateStats();
  }, [calculateStats]);

  useEffect(() => {
    const timeoutId = setTimeout(fetchSuggestions, 1000);
    return () => clearTimeout(timeoutId);
  }, [fetchSuggestions]);

  useEffect(() => {
    setIsDirty(true);
  }, [content]);

  // Auto-save effect
  useEffect(() => {
    if (isDirty && content.length > 10) {
      const timeoutId = setTimeout(autoSaveDocument, 5000);
      return () => clearTimeout(timeoutId);
    }
  }, [content, isDirty, autoSaveDocument]);

  const getSuggestionIcon = (type: string) => {
    switch (type) {
      case 'spelling': return <AlertCircle className="w-4 h-4 text-red-500" />;
      case 'grammar': return <Lightbulb className="w-4 h-4 text-yellow-500" />;
      case 'tense': return <Clock className="w-4 h-4 text-purple-500" />;
      case 'clarity': return <Eye className="w-4 h-4 text-blue-500" />;
      case 'style': return <CheckCircle className="w-4 h-4 text-green-500" />;
      default: return <Lightbulb className="w-4 h-4 text-gray-500" />;
    }
  };

  const getSuggestionColor = (type: string) => {
    switch (type) {
      case 'spelling': return 'border-red-200 bg-red-50';
      case 'grammar': return 'border-yellow-200 bg-yellow-50';
      case 'tense': return 'border-purple-200 bg-purple-50';
      case 'clarity': return 'border-blue-200 bg-blue-50';
      case 'style': return 'border-green-200 bg-green-50';
      default: return 'border-gray-200 bg-gray-50';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600 bg-green-100';
    if (score >= 60) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getLayoutClasses = () => {
    switch (layoutMode) {
      case 'focus':
        return 'grid-cols-1 lg:grid-cols-6';
      case 'minimal':
        return 'grid-cols-1 lg:grid-cols-8';
      default:
        return 'grid-cols-1 lg:grid-cols-4';
    }
  };

  const runPromptChaining = async () => {
    setChainingLoading(true);
    setChainingError(null);
    setChainingResults(null);
    try {
      // 1. Correct
      const suggestionsRes = await fetch('http://localhost:8000/api/ai/suggestions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, goal, tone: 'professional', audience: 'general' }),
      });
      if (!suggestionsRes.ok) throw new Error('Correction step failed');
      const suggestionsData = await suggestionsRes.json();
      let correctedText = content;
      if (suggestionsData.suggestions && suggestionsData.suggestions.length > 0) {
        // Apply all suggestions in order (simple left-to-right, may not handle overlaps)
        let tempText = content;
        suggestionsData.suggestions.forEach((s: any) => {
          if (s.original_text && s.suggested_text && tempText.includes(s.original_text)) {
            tempText = tempText.replace(s.original_text, s.suggested_text);
          }
        });
        correctedText = tempText;
      }
      // 2. Rewrite
      const rewriteRes = await fetch('http://localhost:8000/api/ai/rewrite', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: correctedText, goal }),
      });
      if (!rewriteRes.ok) throw new Error('Rewrite step failed');
      const rewriteData = await rewriteRes.json();
      const rewrittenText = rewriteData.rewritten_text || correctedText;
      // 3. Summarize
      const summarizeRes = await fetch('http://localhost:8000/api/ai/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: rewrittenText }),
      });
      if (!summarizeRes.ok) throw new Error('Summarize step failed');
      const summarizeData = await summarizeRes.json();
      const summary = summarizeData.summary || '';
      setChainingResults({ corrected: correctedText, rewritten: rewrittenText, summary });
    } catch (err: any) {
      setChainingError(err.message || 'Prompt chaining failed');
    } finally {
      setChainingLoading(false);
    }
  };

  return (
    <div className="min-h-screen min-w-full w-full h-full bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <Link href="/">
                <Button variant="ghost" size="icon">
                  <ArrowLeft className="w-5 h-5" />
                </Button>
              </Link>
              <div className="flex items-center space-x-4">
                {isEditingTitle ? (
                  <Input
                    ref={titleInputRef}
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    onBlur={handleTitleSave}
                    onKeyDown={handleTitleKeyDown}
                    className="w-64 font-semibold text-lg border-2 border-blue-500 focus:ring-2 focus:ring-blue-500"
                    placeholder="Document title..."
                  />
                ) : (
                  <div 
                    className="w-64 font-semibold text-lg px-3 py-2 border border-transparent hover:border-gray-300 rounded cursor-pointer"
                    onClick={handleTitleEdit}
                  >
                    {title}
                  </div>
                )}
                <Badge variant={isDirty ? "destructive" : "secondary"}>
                  {isDirty ? 'Unsaved' : 'Saved'}
                </Badge>
                <Badge className={`font-bold ${getScoreColor(stats.overallScore)}`}>
                  <Star className="w-3 h-3 mr-1" />
                  {stats.overallScore}/100
                </Badge>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              {/* Layout controls */}
              <div className="flex items-center space-x-1 border rounded-lg p-1">
                <Button
                  variant={layoutMode === 'minimal' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setLayoutMode('minimal')}
                >
                  <Minimize2 className="w-4 h-4" />
                </Button>
                <Button
                  variant={layoutMode === 'default' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setLayoutMode('default')}
                >
                  <Settings className="w-4 h-4" />
                </Button>
                <Button
                  variant={layoutMode === 'focus' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setLayoutMode('focus')}
                >
                  <Maximize2 className="w-4 h-4" />
                </Button>
              </div>

              <Separator orientation="vertical" className="h-6" />

              {/* Formatting toolbar */}
              <div className="flex items-center space-x-1 border rounded-lg p-1">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => formatText('bold')}
                  disabled={!selectedText}
                >
                  <Bold className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => formatText('italic')}
                  disabled={!selectedText}
                >
                  <Italic className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => formatText('underline')}
                  disabled={!selectedText}
                >
                  <Underline className="w-4 h-4" />
                </Button>
              </div>

              <Separator orientation="vertical" className="h-6" />

              {/* Undo/Redo */}
              <Button
                variant="ghost"
                size="sm"
                onClick={undo}
                disabled={undoStack.length === 0}
              >
                <Undo className="w-4 h-4" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={redo}
                disabled={redoStack.length === 0}
              >
                <Redo className="w-4 h-4" />
              </Button>

              <Separator orientation="vertical" className="h-6" />

              {/* Document actions */}
              <Button variant="outline" size="sm" onClick={copyDocument}>
                <Copy className="w-4 h-4 mr-2" />
                Copy
              </Button>
              <Button variant="outline" size="sm" onClick={downloadDocument}>
                <Download className="w-4 h-4 mr-2" />
                Download
              </Button>
              <Button size="sm" onClick={autoSaveDocument}>
                <Save className="w-4 h-4 mr-2" />
                Save
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className={`grid ${getLayoutClasses()} gap-6`}>
          {/* Left Sidebar - Document History */}
          {layoutMode !== 'minimal' && (
            <div className="lg:col-span-1">
              {/* Create Document Button */}
              <Button
                className="w-full mb-4"
                variant="default"
                onClick={() => {
                  setTitle('Untitled Document');
                  setContent('');
                  setIsDirty(false);
                }}
              >
                + Create Document
              </Button>
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center space-x-2">
                    <FileText className="w-4 h-4" />
                    <span>Document History</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowHistory(!showHistory)}
                      className="ml-auto"
                    >
                      {showHistory ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </CardTitle>
                </CardHeader>
                {showHistory && (
                  <CardContent>
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                      {documentHistory.length === 0 ? (
                        <div className="text-center py-4 text-gray-500">
                          <FileText className="w-6 h-6 mx-auto mb-2" />
                          <p className="text-sm">No saved documents</p>
                        </div>
                      ) : (
                        documentHistory.map((doc) => (
                          <div
                            key={doc.id}
                            className="p-3 border rounded-lg hover:bg-gray-50 cursor-pointer transition-colors"
                            onClick={() => loadDocument(doc)}
                          >
                            <div className="flex items-center justify-between mb-1">
                              <h4 className="text-sm font-medium text-gray-900 truncate">
                                {doc.title}
                              </h4>
                              <Badge className={`text-xs ${getScoreColor(doc.score)}`}>
                                {doc.score}/100
                              </Badge>
                            </div>
                            <div className="flex items-center justify-between text-xs text-gray-500">
                              <span>{doc.wordCount} words</span>
                              <span className="flex items-center">
                                <Clock className="w-3 h-3 mr-1" />
                                {new Date(doc.lastModified).toLocaleDateString()}
                              </span>
                            </div>
                          </div>
                        ))
                      )}
                  </div>
                  </CardContent>
                )}
              </Card>
              {/* Suggestion Filters - moved below Document History */}
              <div className="mt-4 p-3 border rounded-lg bg-gray-50">
                <div className="font-semibold mb-2 text-sm">Suggestion Filters</div>
                <div className="flex flex-wrap gap-4">
                  {['spelling', 'grammar', 'tense', 'clarity', 'style', 'punctuation', 'formal', 'casual', 'marketing', 'friendly', 'general'].map((type) => (
                    <label key={type} className="flex items-center space-x-2 text-sm capitalize cursor-pointer">
                      <input
                        type="checkbox"
                        id={type}
                        checked={suggestionFilters.has(type)}
                        onChange={() => toggleSuggestionFilter(type)}
                        className="rounded border-gray-300 focus:ring-blue-500"
                      />
                      <span>{type}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Statistics Card - moved below suggestion filter */}
              {showStats && (
                <Card className="mt-4">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center space-x-2">
                      <BarChart3 className="w-4 h-4" />
                      <span>Statistics</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setShowStats(false)}
                        className="ml-auto"
                      >
                        <EyeOff className="w-4 h-4" />
                      </Button>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Overall Score</span>
                        <Badge className={`font-bold ${getScoreColor(stats.overallScore)}`}>
                          <Star className="w-3 h-3 mr-1" />
                          {stats.overallScore}/100
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Words</span>
                        <Badge variant="default">{stats.wordCount}</Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Characters</span>
                        <Badge variant="default">{stats.characters}</Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Sentences</span>
                        <Badge variant="default">{stats.sentences}</Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Readability</span>
                        <Badge variant="default">{stats.readabilityScore}%</Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Tone</span>
                        <Badge variant="secondary">{stats.tone}</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {/* Main Editor */}
          <div className={layoutMode === 'focus' ? 'lg:col-span-4' : layoutMode === 'minimal' ? 'lg:col-span-6' : 'lg:col-span-2'}>
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Editor</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="mb-2 flex items-center gap-2">
                  <Button
                    type="button"
                    variant={isListening ? 'destructive' : 'outline'}
                    onClick={isListening ? stopListening : startListening}
                    className="mb-2"
                  >
                    {isListening ? 'Stop Voice Input' : 'Start Voice Input'}
                  </Button>
                  {isListening && <span className="text-red-500 font-semibold">Listening...</span>}
                </div>
                <Textarea
                  ref={textareaRef}
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  onSelect={handleTextSelection}
                  className="min-h-[500px] resize-none border-0 focus:ring-0 text-base leading-relaxed"
                  placeholder="Start writing your content here..."
                />
                {/* Suggestion Filters - removed from here */}
              </CardContent>
            </Card>
          </div>

          {/* Right Sidebar - Suggestions First, Then Statistics */}
          {layoutMode !== 'minimal' && (
            <div className="space-y-6">
              {/* AI Suggestions - First Priority */}
              {showSuggestions && (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center space-x-2">
                      <Zap className="w-4 h-4 text-blue-500" />
                      <span>AI Suggestions</span>
                      <Badge variant="secondary">{suggestions.length}</Badge>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setShowSuggestions(false)}
                        className="ml-auto"
                      >
                        <EyeOff className="w-4 h-4" />
                      </Button>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {isLoading ? (
                        <div className="text-center py-6 text-gray-500">
                          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
                          <p className="text-sm">AI analyzing your text...</p>
                          <p className="text-xs text-gray-400 mt-1">Using Groq & Hugging Face APIs</p>
                        </div>
                      ) : aiStatus === 'error' ? (
                        <div className="text-center py-6 text-gray-500">
                          <AlertCircle className="w-8 h-8 mx-auto mb-2 text-red-500" />
                          <p className="text-sm">AI service temporarily unavailable</p>
                          <p className="text-xs text-gray-400 mt-1">Please try again later</p>
                        </div>
                      ) : suggestions.length === 0 ? (
                        <div className="text-center py-6 text-gray-500">
                          <CheckCircle className="w-8 h-8 mx-auto mb-2 text-green-500" />
                          <p className="text-sm">No AI suggestions at the moment</p>
                          <p className="text-xs text-gray-400 mt-1">Keep writing to see AI-powered suggestions</p>
                        </div>
                      ) : (
                        suggestions.map((suggestion) => (
                          <div
                            key={suggestion.id}
                            className={`p-3 rounded-lg border ${getSuggestionColor(suggestion.type)}`}
                          >
                            <div className="flex items-start space-x-2">
                              {getSuggestionIcon(suggestion.type)}
                              <div className="flex-1">
                                <div className="flex items-center justify-between mb-1">
                                  <span className="text-xs font-medium text-gray-600 uppercase">
                                    {suggestion.type}
                                  </span>
                                  <span className="text-xs font-medium text-gray-600">
                                    {suggestion.severity}
                                  </span>
                                </div>
                                <p className="text-sm text-gray-900 mb-1">
                                  <span className="bg-red-100 text-red-800 px-1 rounded">"{suggestion.original_text}"</span>
                                  <span className="mx-1">→</span>
                                  <span className="bg-green-100 text-green-800 px-1 rounded font-medium">"{suggestion.suggested_text}"</span>
                                </p>
                                <p className="text-xs text-gray-600 mb-2">
                                  {suggestion.explanation}
                                </p>
                                <div className="flex items-center justify-between">
                                  <span className="text-xs text-gray-400">
                                    Confidence: {Math.round(suggestion.confidence * 100)}%
                                  </span>
                                <Button
                                  size="sm"
                                    className="text-xs"
                                  onClick={() => applySuggestion(suggestion)}
                                >
                                  Apply
                                </Button>
                                </div>
                              </div>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Prompt Chaining UI - moved here */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="secondary"
                    onClick={runPromptChaining}
                    disabled={chainingLoading}
                    className="mb-2"
                  >
                    {chainingLoading ? 'Processing...' : 'Correct → Rewrite → Summarize'}
                  </Button>
                  {chainingError && <span className="text-red-500 font-semibold">{chainingError}</span>}
                </div>
                {chainingResults && (
                  <div className="p-4 border rounded-lg bg-gray-50">
                    <div className="mb-2"><span className="font-semibold">Corrected:</span> <span className="text-gray-700">{chainingResults.corrected}</span></div>
                    <div className="mb-2"><span className="font-semibold">Rewritten ({goal}):</span> <span className="text-gray-700">{chainingResults.rewritten}</span></div>
                    <div><span className="font-semibold">Summary:</span> <span className="text-gray-700">{chainingResults.summary}</span></div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}