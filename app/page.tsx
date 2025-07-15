'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  FileText, 
  TrendingUp, 
  Target, 
  BookOpen, 
  PlusCircle,
  Search,
  Settings,
  Bell,
  User,
  CheckCircle,
  Clock,
  BarChart3
} from 'lucide-react';

export default function Dashboard() {
  const [recentDocuments] = useState([
    {
      id: 1,
      title: 'Marketing Proposal Draft',
      lastModified: '2 hours ago',
      wordCount: 1250,
      score: 89,
      status: 'In Progress'
    },
    {
      id: 2,
      title: 'Research Paper - AI Ethics',
      lastModified: '1 day ago',
      wordCount: 3500,
      score: 95,
      status: 'Completed'
    },
    {
      id: 3,
      title: 'Email Campaign Copy',
      lastModified: '3 days ago',
      wordCount: 800,
      score: 92,
      status: 'Reviewed'
    }
  ]);

  const [weeklyStats] = useState({
    wordsWritten: 12500,
    documentsCreated: 8,
    averageScore: 91,
    improvementRate: 15
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                  <CheckCircle className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold text-gray-900">GrammarlyClone</span>
              </div>
              
              <nav className="hidden md:flex space-x-6">
                <Link href="/" className="text-gray-700 hover:text-green-600 transition-colors">
                  Dashboard
                </Link>
                <Link href="/editor" className="text-gray-700 hover:text-green-600 transition-colors">
                  Editor
                </Link>
                <Link href="/insights" className="text-gray-700 hover:text-green-600 transition-colors">
                  Insights
                </Link>
                <Link href="/plagiarism" className="text-gray-700 hover:text-green-600 transition-colors">
                  Plagiarism
                </Link>
              </nav>
            </div>

            <div className="flex items-center space-x-4">
              <div className="relative">
                <Search className="w-5 h-5 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2" />
                <input
                  type="text"
                  placeholder="Search documents..."
                  className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                />
              </div>
              
              <Button variant="ghost" size="icon">
                <Bell className="w-5 h-5" />
              </Button>
              
              <Button variant="ghost" size="icon">
                <Settings className="w-5 h-5" />
              </Button>
              
              <Button variant="ghost" size="icon">
                <User className="w-5 h-5" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Welcome back!</h1>
          <p className="text-gray-600">Let's continue improving your writing.</p>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Link href="/editor">
            <Card className="hover:shadow-lg transition-shadow cursor-pointer border-2 border-transparent hover:border-green-500">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center space-x-2">
                  <PlusCircle className="w-5 h-5 text-green-600" />
                  <span>New Document</span>
                </CardTitle>
                <CardDescription>Start writing with AI-powered assistance</CardDescription>
              </CardHeader>
            </Card>
          </Link>

          <Link href="/insights">
            <Card className="hover:shadow-lg transition-shadow cursor-pointer border-2 border-transparent hover:border-blue-500">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center space-x-2">
                  <BarChart3 className="w-5 h-5 text-blue-600" />
                  <span>View Insights</span>
                </CardTitle>
                <CardDescription>Analyze your writing patterns</CardDescription>
              </CardHeader>
            </Card>
          </Link>

          <Link href="/plagiarism">
            <Card className="hover:shadow-lg transition-shadow cursor-pointer border-2 border-transparent hover:border-purple-500">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center space-x-2">
                  <Search className="w-5 h-5 text-purple-600" />
                  <span>Check Plagiarism</span>
                </CardTitle>
                <CardDescription>Verify content originality</CardDescription>
              </CardHeader>
            </Card>
          </Link>
        </div>

        {/* Weekly Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Words Written</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-gray-900">{weeklyStats.wordsWritten.toLocaleString()}</div>
              <div className="flex items-center text-sm text-green-600 mt-1">
                <TrendingUp className="w-4 h-4 mr-1" />
                <span>+12% from last week</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Documents</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-gray-900">{weeklyStats.documentsCreated}</div>
              <div className="flex items-center text-sm text-blue-600 mt-1">
                <FileText className="w-4 h-4 mr-1" />
                <span>Created this week</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Average Score</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-gray-900">{weeklyStats.averageScore}%</div>
              <div className="flex items-center text-sm text-green-600 mt-1">
                <Target className="w-4 h-4 mr-1" />
                <span>Above average</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Improvement</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-gray-900">+{weeklyStats.improvementRate}%</div>
              <div className="flex items-center text-sm text-purple-600 mt-1">
                <TrendingUp className="w-4 h-4 mr-1" />
                <span>This month</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Recent Documents */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BookOpen className="w-5 h-5" />
              <span>Recent Documents</span>
            </CardTitle>
            <CardDescription>Your latest writing projects</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentDocuments.map((doc) => (
                <div key={doc.id} className="flex items-center justify-between p-4 rounded-lg border hover:bg-gray-50 transition-colors">
                  <div className="flex items-center space-x-4">
                    <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                      <FileText className="w-5 h-5 text-green-600" />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">{doc.title}</h3>
                      <div className="flex items-center space-x-4 text-sm text-gray-500">
                        <span className="flex items-center">
                          <Clock className="w-4 h-4 mr-1" />
                          {doc.lastModified}
                        </span>
                        <span>{doc.wordCount} words</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <Badge variant={doc.status === 'Completed' ? 'default' : 'secondary'}>
                      {doc.status}
                    </Badge>
                    <div className="text-right">
                      <div className="text-sm font-medium text-gray-900">{doc.score}%</div>
                      <div className="text-xs text-gray-500">Score</div>
                    </div>
                    <Link href={`/editor/${doc.id}`}>
                      <Button size="sm">Edit</Button>
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}