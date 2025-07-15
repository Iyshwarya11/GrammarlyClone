'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  ArrowLeft,
  Search,
  Upload,
  FileText,
  AlertTriangle,
  CheckCircle,
  ExternalLink,
  Eye,
  Download,
  Share
} from 'lucide-react';

interface PlagiarismResult {
  id: string;
  source: string;
  similarity: number;
  matchedText: string;
  url: string;
  type: 'web' | 'academic' | 'publication';
}

export default function PlagiarismChecker() {
  const [content, setContent] = useState('');
  const [isChecking, setIsChecking] = useState(false);
  const [results, setResults] = useState<PlagiarismResult[]>([]);
  const [overallScore, setOverallScore] = useState(0);

  const mockResults: PlagiarismResult[] = [
    {
      id: '1',
      source: 'Wikipedia - Artificial Intelligence',
      similarity: 15,
      matchedText: 'Artificial intelligence is the simulation of human intelligence processes by machines',
      url: 'https://en.wikipedia.org/wiki/Artificial_intelligence',
      type: 'web'
    },
    {
      id: '2',
      source: 'IEEE Research Paper',
      similarity: 8,
      matchedText: 'Machine learning algorithms have revolutionized the field of data analysis',
      url: 'https://ieeexplore.ieee.org/document/example',
      type: 'academic'
    },
    {
      id: '3',
      source: 'Nature Journal',
      similarity: 5,
      matchedText: 'The implications of artificial intelligence on society are vast and complex',
      url: 'https://nature.com/articles/example',
      type: 'publication'
    }
  ];

  const handleCheck = async () => {
    if (!content.trim()) return;
    
    setIsChecking(true);
    
    // Simulate API call
    setTimeout(() => {
      const totalSimilarity = mockResults.reduce((sum, result) => sum + result.similarity, 0);
      setOverallScore(100 - totalSimilarity);
      setResults(mockResults);
      setIsChecking(false);
    }, 3000);
  };

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-600';
    if (score >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreStatus = (score: number) => {
    if (score >= 90) return 'Excellent';
    if (score >= 70) return 'Good';
    return 'Needs Review';
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'web': return '🌐';
      case 'academic': return '🎓';
      case 'publication': return '📄';
      default: return '📝';
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
                <h1 className="text-lg font-semibold text-gray-900">Plagiarism Checker</h1>
                <p className="text-sm text-gray-500">Verify the originality of your content</p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm">
                <Upload className="w-4 h-4 mr-2" />
                Upload File
              </Button>
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                Export Report
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Input Section */}
          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Search className="w-5 h-5" />
                  <span>Check for Plagiarism</span>
                </CardTitle>
                <CardDescription>
                  Paste your content below or upload a file to check for potential plagiarism
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <textarea
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    placeholder="Paste your content here to check for plagiarism..."
                    className="w-full h-64 p-4 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  />
                  
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-gray-600">
                      {content.length} characters • {content.split(' ').filter(word => word.length > 0).length} words
                    </div>
                    <Button 
                      onClick={handleCheck}
                      disabled={!content.trim() || isChecking}
                    >
                      {isChecking ? (
                        <div className="flex items-center space-x-2">
                          <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                          <span>Checking...</span>
                        </div>
                      ) : (
                        <div className="flex items-center space-x-2">
                          <Search className="w-4 h-4" />
                          <span>Check Plagiarism</span>
                        </div>
                      )}
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Results */}
            {(results.length > 0 || isChecking) && (
              <Card className="mt-6">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <FileText className="w-5 h-5" />
                    <span>Plagiarism Results</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {isChecking ? (
                    <div className="flex items-center justify-center py-12">
                      <div className="text-center">
                        <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                        <p className="text-gray-600">Scanning content for plagiarism...</p>
                        <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
                      </div>
                    </div>
                  ) : (
                    <Tabs defaultValue="overview" className="w-full">
                      <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="overview">Overview</TabsTrigger>
                        <TabsTrigger value="details">Detailed Results</TabsTrigger>
                      </TabsList>
                      
                      <TabsContent value="overview" className="space-y-4">
                        <div className="text-center py-6">
                          <div className={`text-4xl font-bold mb-2 ${getScoreColor(overallScore)}`}>
                            {overallScore}%
                          </div>
                          <div className="text-lg text-gray-600 mb-4">Originality Score</div>
                          <Badge variant={overallScore >= 90 ? 'default' : overallScore >= 70 ? 'secondary' : 'destructive'}>
                            {getScoreStatus(overallScore)}
                          </Badge>
                        </div>
                        
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <div className="text-center p-4 bg-blue-50 rounded-lg">
                            <div className="text-2xl font-bold text-blue-600">{results.length}</div>
                            <div className="text-sm text-gray-600">Sources Found</div>
                          </div>
                          <div className="text-center p-4 bg-yellow-50 rounded-lg">
                            <div className="text-2xl font-bold text-yellow-600">
                              {results.reduce((sum, r) => sum + r.similarity, 0)}%
                            </div>
                            <div className="text-sm text-gray-600">Total Similarity</div>
                          </div>
                          <div className="text-center p-4 bg-green-50 rounded-lg">
                            <div className="text-2xl font-bold text-green-600">{overallScore}%</div>
                            <div className="text-sm text-gray-600">Original Content</div>
                          </div>
                        </div>
                      </TabsContent>
                      
                      <TabsContent value="details" className="space-y-4">
                        {results.map((result) => (
                          <Card key={result.id} className="border-l-4 border-l-yellow-400">
                            <CardHeader className="pb-2">
                              <div className="flex items-center justify-between">
                                <CardTitle className="text-base flex items-center space-x-2">
                                  <span>{getTypeIcon(result.type)}</span>
                                  <span>{result.source}</span>
                                </CardTitle>
                                <Badge variant="outline">{result.similarity}% match</Badge>
                              </div>
                            </CardHeader>
                            <CardContent>
                              <div className="space-y-3">
                                <div className="p-3 bg-yellow-50 rounded-lg">
                                  <p className="text-sm text-gray-900">"{result.matchedText}"</p>
                                </div>
                                <div className="flex items-center justify-between">
                                  <a 
                                    href={result.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="flex items-center space-x-1 text-blue-600 hover:text-blue-800 text-sm"
                                  >
                                    <span>View Source</span>
                                    <ExternalLink className="w-3 h-3" />
                                  </a>
                                  <div className="flex items-center space-x-2">
                                    <Button variant="outline" size="sm">
                                      <Eye className="w-4 h-4 mr-1" />
                                      Preview
                                    </Button>
                                  </div>
                                </div>
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </TabsContent>
                    </Tabs>
                  )}
                </CardContent>
              </Card>
            )}
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="space-y-6">
              {/* Quick Tips */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Quick Tips</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3 text-sm">
                    <div className="flex items-start space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <span>Always cite your sources properly</span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <span>Use quotation marks for direct quotes</span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <span>Paraphrase instead of copying</span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <span>Check before submitting</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* File Upload */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Upload Document</CardTitle>
                  <CardDescription>Support for .docx, .pdf, .txt files</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                    <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                    <p className="text-sm text-gray-600 mb-2">Drop files here or click to browse</p>
                    <Button variant="outline" size="sm">
                      Select Files
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Recent Checks */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Recent Checks</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {[
                      { name: 'Research Paper Draft', score: 92, date: '2 hours ago' },
                      { name: 'Article Review', score: 88, date: '1 day ago' },
                      { name: 'Essay Assignment', score: 95, date: '3 days ago' }
                    ].map((item, index) => (
                      <div key={index} className="flex items-center justify-between p-2 rounded-lg hover:bg-gray-50">
                        <div>
                          <div className="font-medium text-sm">{item.name}</div>
                          <div className="text-xs text-gray-500">{item.date}</div>
                        </div>
                        <Badge variant="outline">{item.score}%</Badge>
                      </div>
                    ))}
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