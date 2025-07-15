'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { 
  Save, 
  Download, 
  Share, 
  Eye, 
  EyeOff,
  RotateCcw,
  CheckCircle,
  AlertCircle,
  Lightbulb,
  Target,
  BookOpen,
  Zap,
  FileText,
  ArrowLeft
} from 'lucide-react';
import Link from 'next/link';

interface Suggestion {
  id: string;
  type: 'grammar' | 'spelling' | 'clarity' | 'tone' | 'engagement';
  text: string;
  suggestion: string;
  explanation: string;
  position: { start: number; end: number };
}

interface DocumentStats {
  wordCount: number;
  characters: number;
  sentences: number;
  paragraphs: number;
  readingTime: number;
  readabilityScore: number;
}

export default function Editor() {
  const [content, setContent] = useState('Write your content here and see real-time suggestions appear in the sidebar. This is a sample text that contains some grammatical errors and can be improved for better clarity and engagement.');
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [stats, setStats] = useState<DocumentStats>({
    wordCount: 0,
    characters: 0,
    sentences: 0,
    paragraphs: 0,
    readingTime: 0,
    readabilityScore: 0
  });
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [activeGoal, setActiveGoal] = useState('clarity');

  // Mock suggestions data
  const mockSuggestions: Suggestion[] = [
    {
      id: '1',
      type: 'grammar',
      text: 'grammatical errors',
      suggestion: 'grammar errors',
      explanation: 'More concise and commonly used phrase',
      position: { start: 120, end: 138 }
    },
    {
      id: '2',
      type: 'clarity',
      text: 'can be improved',
      suggestion: 'could be enhanced',
      explanation: 'Stronger, more confident language',
      position: { start: 143, end: 158 }
    },
    {
      id: '3',
      type: 'tone',
      text: 'This is a sample text',
      suggestion: 'Here\'s an example',
      explanation: 'More conversational and engaging',
      position: { start: 80, end: 101 }
    }
  ];

  useEffect(() => {
    // Calculate stats
    const words = content.trim().split(/\s+/).filter(word => word.length > 0);
    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const paragraphs = content.split(/\n\s*\n/).filter(p => p.trim().length > 0);
    
    setStats({
      wordCount: words.length,
      characters: content.length,
      sentences: sentences.length,
      paragraphs: paragraphs.length,
      readingTime: Math.ceil(words.length / 250),
      readabilityScore: Math.floor(Math.random() * 20) + 80 // Mock score
    });

    // Mock API call for suggestions
    if (content.length > 50) {
      setSuggestions(mockSuggestions);
    } else {
      setSuggestions([]);
    }
  }, [content]);

  const applySuggestion = (suggestion: Suggestion) => {
    const newContent = content.substring(0, suggestion.position.start) + 
                      suggestion.suggestion + 
                      content.substring(suggestion.position.end);
    setContent(newContent);
    setSuggestions(prev => prev.filter(s => s.id !== suggestion.id));
  };

  const getSuggestionIcon = (type: string) => {
    switch (type) {
      case 'grammar': return <AlertCircle className="w-4 h-4 text-red-500" />;
      case 'spelling': return <CheckCircle className="w-4 h-4 text-orange-500" />;
      case 'clarity': return <Lightbulb className="w-4 h-4 text-blue-500" />;
      case 'tone': return <Target className="w-4 h-4 text-purple-500" />;
      case 'engagement': return <Zap className="w-4 h-4 text-green-500" />;
      default: return <AlertCircle className="w-4 h-4 text-gray-500" />;
    }
  };

  const getSuggestionColor = (type: string) => {
    switch (type) {
      case 'grammar': return 'bg-red-50 border-red-200';
      case 'spelling': return 'bg-orange-50 border-orange-200';
      case 'clarity': return 'bg-blue-50 border-blue-200';
      case 'tone': return 'bg-purple-50 border-purple-200';
      case 'engagement': return 'bg-green-50 border-green-200';
      default: return 'bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
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
              <div>
                <h1 className="text-lg font-semibold text-gray-900">Untitled Document</h1>
                <p className="text-sm text-gray-500">Auto-saved 2 minutes ago</p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
              <Button variant="outline" size="sm">
                <Share className="w-4 h-4 mr-2" />
                Share
              </Button>
              <Button size="sm">
                <Save className="w-4 h-4 mr-2" />
                Save
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Main Editor */}
          <div className="lg:col-span-3">
            <Card className="h-full">
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                      <FileText className="w-5 h-5 text-gray-500" />
                      <span className="text-sm font-medium">Document</span>
                    </div>
                    <Badge variant="secondary">
                      Score: {stats.readabilityScore}%
                    </Badge>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowSuggestions(!showSuggestions)}
                    >
                      {showSuggestions ? (
                        <EyeOff className="w-4 h-4" />
                      ) : (
                        <Eye className="w-4 h-4" />
                      )}
                    </Button>
                    <Button variant="ghost" size="sm">
                      <RotateCcw className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="min-h-96">
                  <textarea
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    className="w-full h-96 p-4 border border-gray-200 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent resize-none text-gray-900 text-base leading-relaxed"
                    placeholder="Start writing your content here..."
                    style={{ fontFamily: 'Georgia, serif' }}
                  />
                </div>
                
                {/* Document Stats */}
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="flex items-center justify-between text-sm text-gray-600">
                    <div className="flex items-center space-x-6">
                      <span>{stats.wordCount} words</span>
                      <span>{stats.characters} characters</span>
                      <span>{stats.sentences} sentences</span>
                      <span>{stats.paragraphs} paragraphs</span>
                    </div>
                    <div className="flex items-center space-x-4">
                      <span>{stats.readingTime} min read</span>
                      <Badge variant="outline">
                        Readability: {stats.readabilityScore}%
                      </Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="space-y-6">
              {/* Writing Goals */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center space-x-2">
                    <Target className="w-4 h-4" />
                    <span>Writing Goals</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {['clarity', 'engagement', 'tone', 'delivery'].map((goal) => (
                      <Button
                        key={goal}
                        variant={activeGoal === goal ? 'default' : 'outline'}
                        size="sm"
                        className="w-full justify-start capitalize"
                        onClick={() => setActiveGoal(goal)}
                      >
                        {goal}
                      </Button>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Suggestions */}
              {showSuggestions && (
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Lightbulb className="w-4 h-4" />
                        <span>Suggestions</span>
                      </div>
                      <Badge variant="secondary">{suggestions.length}</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {suggestions.length === 0 ? (
                        <div className="text-center py-6 text-gray-500">
                          <CheckCircle className="w-8 h-8 mx-auto mb-2 text-green-500" />
                          <p className="text-sm">No suggestions at the moment</p>
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
                                </div>
                                <p className="text-sm text-gray-900 mb-1">
                                  "{suggestion.text}" → "{suggestion.suggestion}"
                                </p>
                                <p className="text-xs text-gray-600 mb-2">
                                  {suggestion.explanation}
                                </p>
                                <Button
                                  size="sm"
                                  className="w-full"
                                  onClick={() => applySuggestion(suggestion)}
                                >
                                  Apply
                                </Button>
                              </div>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Performance */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center space-x-2">
                    <BookOpen className="w-4 h-4" />
                    <span>Performance</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Overall Score</span>
                      <Badge variant="default">{stats.readabilityScore}%</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Clarity</span>
                      <Badge variant="secondary">Good</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Engagement</span>
                      <Badge variant="secondary">Excellent</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Tone</span>
                      <Badge variant="secondary">Professional</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}