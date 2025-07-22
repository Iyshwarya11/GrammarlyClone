'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Target,
  BookOpen,
  Clock,
  Users,
  Award,
  BarChart3,
  PieChart,
  LineChart
} from 'lucide-react';

export default function Insights() {
  const [timeRange, setTimeRange] = useState('week');
  const [insightsData, setInsightsData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchInsights = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch('http://localhost:8000/api/ai/insights', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: 'default', time_range: timeRange }),
        });
        if (!response.ok) {
          console.error('Insights API Error:', response.status, response.statusText);
          setError(`API Error: ${response.status}`);
          setInsightsData(null);
          return;
        }
        const data = await response.json();
        console.log('Insights API Response:', data); // Debug log
        setInsightsData(data);
      } catch (err: any) {
        console.error('Insights Fetch Error:', err);
        setError(err.message || 'Unknown error');
        setInsightsData(null);
      } finally {
        setLoading(false);
      }
    };
    fetchInsights();
  }, [timeRange]);

  const achievements = [
    { id: 1, title: 'Writing Streak', description: '7 days in a row', icon: '🔥', unlocked: true },
    { id: 2, title: 'Word Master', description: '10,000 words written', icon: '📝', unlocked: true },
    { id: 3, title: 'Grammar Guru', description: '95% accuracy rate', icon: '✅', unlocked: false },
    { id: 4, title: 'Clarity Champion', description: 'Consistently clear writing', icon: '💡', unlocked: true }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
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
                <h1 className="text-lg font-semibold text-gray-900">Writing Insights</h1>
                <p className="text-sm text-gray-500">Track your progress and improve your writing</p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Button
                variant={timeRange === 'week' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setTimeRange('week')}
              >
                Week
              </Button>
              <Button
                variant={timeRange === 'month' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setTimeRange('month')}
              >
                Month
              </Button>
              <Button
                variant={timeRange === 'year' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setTimeRange('year')}
              >
                Year
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="goals">Goals</TabsTrigger>
            <TabsTrigger value="achievements">Achievements</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-gray-600">Total Words</CardTitle>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="text-2xl font-bold text-gray-900">Loading...</div>
                  ) : error ? (
                    <div className="text-2xl font-bold text-red-600">{error}</div>
                  ) : insightsData?.performance_metrics ? (
                    <div className="text-2xl font-bold text-gray-900">{typeof insightsData?.performance_metrics?.total_words === "number"
  ? insightsData.performance_metrics.total_words.toLocaleString()
  : "N/A"}</div>
                  ) : (
                    <div className="text-2xl font-bold text-gray-900">N/A</div>
                  )}
                  {loading ? (
                    <div className="flex items-center text-sm text-green-600 mt-1">
                      <TrendingUp className="w-4 h-4 mr-1" />
                      <span>Loading...</span>
                    </div>
                  ) : error ? (
                    <div className="flex items-center text-sm text-red-600 mt-1">
                      <TrendingDown className="w-4 h-4 mr-1" />
                      <span>{error}</span>
                    </div>
                  ) : insightsData?.performance_metrics ? (
                  <div className="flex items-center text-sm text-green-600 mt-1">
                    <TrendingUp className="w-4 h-4 mr-1" />
                      <span>+{insightsData.performance_metrics.improvement_rate}% this week</span>
                    </div>
                  ) : (
                    <div className="flex items-center text-sm text-gray-600 mt-1">
                      <TrendingDown className="w-4 h-4 mr-1" />
                      <span>No data</span>
                  </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-gray-600">Documents</CardTitle>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="text-2xl font-bold text-gray-900">Loading...</div>
                  ) : error ? (
                    <div className="text-2xl font-bold text-red-600">{error}</div>
                  ) : insightsData?.performance_metrics ? (
                    <div className="text-2xl font-bold text-gray-900">{insightsData.performance_metrics.documents_created}</div>
                  ) : (
                    <div className="text-2xl font-bold text-gray-900">N/A</div>
                  )}
                  {loading ? (
                    <div className="flex items-center text-sm text-blue-600 mt-1">
                      <BookOpen className="w-4 h-4 mr-1" />
                      <span>Loading...</span>
                    </div>
                  ) : error ? (
                    <div className="flex items-center text-sm text-red-600 mt-1">
                      <BookOpen className="w-4 h-4 mr-1" />
                      <span>{error}</span>
                    </div>
                  ) : insightsData?.performance_metrics ? (
                  <div className="flex items-center text-sm text-blue-600 mt-1">
                    <BookOpen className="w-4 h-4 mr-1" />
                    <span>Created this week</span>
                  </div>
                  ) : (
                    <div className="flex items-center text-sm text-gray-600 mt-1">
                      <BookOpen className="w-4 h-4 mr-1" />
                      <span>No data</span>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-gray-600">Average Score</CardTitle>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="text-2xl font-bold text-gray-900">Loading...</div>
                  ) : error ? (
                    <div className="text-2xl font-bold text-red-600">{error}</div>
                  ) : insightsData?.performance_metrics ? (
                    <div className="text-2xl font-bold text-gray-900">{insightsData.performance_metrics.average_score}%</div>
                  ) : (
                    <div className="text-2xl font-bold text-gray-900">N/A</div>
                  )}
                  {loading ? (
                    <div className="flex items-center text-sm text-green-600 mt-1">
                      <Target className="w-4 h-4 mr-1" />
                      <span>Loading...</span>
                    </div>
                  ) : error ? (
                    <div className="flex items-center text-sm text-red-600 mt-1">
                      <Target className="w-4 h-4 mr-1" />
                      <span>{error}</span>
                    </div>
                  ) : insightsData?.performance_metrics ? (
                  <div className="flex items-center text-sm text-green-600 mt-1">
                    <Target className="w-4 h-4 mr-1" />
                    <span>Above baseline</span>
                  </div>
                  ) : (
                    <div className="flex items-center text-sm text-gray-600 mt-1">
                      <Target className="w-4 h-4 mr-1" />
                      <span>No data</span>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-gray-600">Time Spent</CardTitle>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="text-2xl font-bold text-gray-900">Loading...</div>
                  ) : error ? (
                    <div className="text-2xl font-bold text-red-600">{error}</div>
                  ) : insightsData?.performance_metrics ? (
                    <div className="text-2xl font-bold text-gray-900">{insightsData.performance_metrics.time_spent}h</div>
                  ) : (
                    <div className="text-2xl font-bold text-gray-900">N/A</div>
                  )}
                  {loading ? (
                    <div className="flex items-center text-sm text-purple-600 mt-1">
                      <Clock className="w-4 h-4 mr-1" />
                      <span>Loading...</span>
                    </div>
                  ) : error ? (
                    <div className="flex items-center text-sm text-red-600 mt-1">
                      <Clock className="w-4 h-4 mr-1" />
                      <span>{error}</span>
                    </div>
                  ) : insightsData?.performance_metrics ? (
                  <div className="flex items-center text-sm text-purple-600 mt-1">
                    <Clock className="w-4 h-4 mr-1" />
                    <span>This week</span>
                  </div>
                  ) : (
                    <div className="flex items-center text-sm text-gray-600 mt-1">
                      <Clock className="w-4 h-4 mr-1" />
                      <span>No data</span>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <LineChart className="w-5 h-5" />
                  <span>Daily Writing Activity</span>
                </CardTitle>
                <CardDescription>Words written and quality scores over time</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="h-64 flex items-end space-x-2">
                    <div className="flex-1 flex flex-col items-center">
                      <div className="w-full bg-gray-200 rounded-t-lg mb-2 relative">
                        <div
                          className="bg-gray-300 rounded-t-lg transition-all duration-300"
                          style={{ height: '100px' }}
                        />
                      </div>
                      <div className="text-xs text-gray-600 mb-1">Loading...</div>
                      <div className="text-xs font-medium">0</div>
                      <Badge variant="outline" className="text-xs mt-1">
                        0%
                      </Badge>
                    </div>
                  </div>
                ) : error ? (
                <div className="h-64 flex items-end space-x-2">
                    <div className="flex-1 flex flex-col items-center">
                      <div className="w-full bg-gray-200 rounded-t-lg mb-2 relative">
                        <div
                          className="bg-red-200 rounded-t-lg transition-all duration-300"
                          style={{ height: '100px' }}
                        />
                      </div>
                      <div className="text-xs text-gray-600 mb-1">Error</div>
                      <div className="text-xs font-medium">0</div>
                      <Badge variant="outline" className="text-xs mt-1">
                        0%
                      </Badge>
                    </div>
                  </div>
                ) : insightsData?.daily_activity ? (
                  <div className="h-64 flex items-end space-x-2">
                    {insightsData.daily_activity.map((day: any, index: number) => (
                      <div key={index} className="flex-1 flex flex-col items-center">
                      <div className="w-full bg-gray-200 rounded-t-lg mb-2 relative">
                        <div
                          className="bg-green-500 rounded-t-lg transition-all duration-300"
                          style={{ height: `${(day.words / 3500) * 200}px` }}
                        />
                      </div>
                      <div className="text-xs text-gray-600 mb-1">{day.day}</div>
                      <div className="text-xs font-medium">{day.words}</div>
                      <Badge variant="outline" className="text-xs mt-1">
                        {day.score}%
                      </Badge>
                    </div>
                  ))}
                </div>
                ) : (
                  <div className="h-64 flex items-end space-x-2">
                    <div className="flex-1 flex flex-col items-center">
                      <div className="w-full bg-gray-200 rounded-t-lg mb-2 relative">
                        <div
                          className="bg-gray-300 rounded-t-lg transition-all duration-300"
                          style={{ height: '100px' }}
                        />
                      </div>
                      <div className="text-xs text-gray-600 mb-1">No data</div>
                      <div className="text-xs font-medium">0</div>
                      <Badge variant="outline" className="text-xs mt-1">
                        0%
                      </Badge>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <PieChart className="w-5 h-5" />
                  <span>Common Issues</span>
                </CardTitle>
                <CardDescription>Most frequent areas for improvement</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
                          <span className="text-sm font-medium">1</span>
                        </div>
                        <div>
                          <div className="font-medium text-gray-900">Loading...</div>
                          <div className="text-sm text-gray-600">Loading...</div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <TrendingUp className="w-4 h-4 text-red-500" />
                        <span className="text-sm font-medium">Loading...</span>
                      </div>
                    </div>
                  </div>
                ) : error ? (
                <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
                          <span className="text-sm font-medium">1</span>
                        </div>
                        <div>
                          <div className="font-medium text-gray-900">Error</div>
                          <div className="text-sm text-gray-600">Error</div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <TrendingDown className="w-4 h-4 text-red-500" />
                        <span className="text-sm font-medium">Error</span>
                      </div>
                    </div>
                  </div>
                ) : insightsData?.top_issues ? (
                  <div className="space-y-3">
                    {insightsData.top_issues.map((issue: any, index: number) => (
                      <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-gray-50">
                        <div className="flex items-center space-x-3">
                          <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
                            <span className="text-sm font-medium">{index + 1}</span>
                          </div>
                          <div>
                            <div className="font-medium text-gray-900">{issue.type}</div>
                            <div className="text-sm text-gray-600">{issue.count} instances</div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        {issue.trend === 'up' ? (
                          <TrendingUp className="w-4 h-4 text-red-500" />
                        ) : (
                          <TrendingDown className="w-4 h-4 text-green-500" />
                        )}
                        <span className="text-sm font-medium">{issue.count}</span>
                      </div>
                    </div>
                  ))}
                </div>
                ) : (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
                          <span className="text-sm font-medium">1</span>
                        </div>
                        <div>
                          <div className="font-medium text-gray-900">No data</div>
                          <div className="text-sm text-gray-600">No data</div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <TrendingDown className="w-4 h-4 text-green-500" />
                        <span className="text-sm font-medium">No data</span>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="performance" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Writing Performance Analytics</CardTitle>
                <CardDescription>Detailed analysis of your writing patterns and quality</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-semibold mb-3">Quality Distribution</h3>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Excellent (90-100%)</span>
                          <span className="text-sm font-medium">Loading...</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Good (80-89%)</span>
                          <span className="text-sm font-medium">Loading...</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Average (70-79%)</span>
                          <span className="text-sm font-medium">Loading...</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Below Average (&lt;70%)</span>
                          <span className="text-sm font-medium">Loading...</span>
                        </div>
                      </div>
                    </div>
                    <div>
                      <h3 className="font-semibold mb-3">Writing Patterns</h3>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Peak Hours</span>
                          <span className="text-sm font-medium">Loading...</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Avg. Session</span>
                          <span className="text-sm font-medium">Loading...</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Best Day</span>
                          <span className="text-sm font-medium">Loading...</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Productivity</span>
                          <span className="text-sm font-medium">Loading...</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : error ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="font-semibold mb-3">Quality Distribution</h3>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Excellent (90-100%)</span>
                          <span className="text-sm font-medium">Error</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Good (80-89%)</span>
                          <span className="text-sm font-medium">Error</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Average (70-79%)</span>
                          <span className="text-sm font-medium">Error</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Below Average (&lt;70%)</span>
                          <span className="text-sm font-medium">Error</span>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-3">Writing Patterns</h3>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Peak Hours</span>
                          <span className="text-sm font-medium">Error</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Avg. Session</span>
                          <span className="text-sm font-medium">Error</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Best Day</span>
                          <span className="text-sm font-medium">Error</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Productivity</span>
                          <span className="text-sm font-medium">Error</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : insightsData?.performance_metrics ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="font-semibold mb-3">Quality Distribution</h3>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Excellent (90-100%)</span>
                          <span className="text-sm font-medium">{typeof insightsData?.performance_metrics?.quality_distribution?.excellent === "number"
  ? insightsData.performance_metrics.quality_distribution.excellent + "%"
  : "N/A"}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Good (80-89%)</span>
                          <span className="text-sm font-medium">{typeof insightsData?.performance_metrics?.quality_distribution?.good === "number"
  ? insightsData.performance_metrics.quality_distribution.good + "%"
  : "N/A"}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Average (70-79%)</span>
                          <span className="text-sm font-medium">{typeof insightsData?.performance_metrics?.quality_distribution?.average === "number"
  ? insightsData.performance_metrics.quality_distribution.average + "%"
  : "N/A"}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Below Average (&lt;70%)</span>
                          <span className="text-sm font-medium">{typeof insightsData?.performance_metrics?.quality_distribution?.below_average === "number"
  ? insightsData.performance_metrics.quality_distribution.below_average + "%"
  : "N/A"}</span>
                        </div>
                      </div>
                    </div>
                    <div>
                      <h3 className="font-semibold mb-3">Writing Patterns</h3>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Peak Hours</span>
                          <span className="text-sm font-medium">{insightsData.performance_metrics.peak_hours}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Avg. Session</span>
                          <span className="text-sm font-medium">{insightsData.performance_metrics.avg_session} minutes</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Best Day</span>
                          <span className="text-sm font-medium">{insightsData.performance_metrics.best_day}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Productivity</span>
                          <span className="text-sm font-medium">{insightsData.performance_metrics.productivity} words/hour</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="font-semibold mb-3">Quality Distribution</h3>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Excellent (90-100%)</span>
                          <span className="text-sm font-medium">N/A</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Good (80-89%)</span>
                          <span className="text-sm font-medium">N/A</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Average (70-79%)</span>
                          <span className="text-sm font-medium">N/A</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Below Average (&lt;70%)</span>
                          <span className="text-sm font-medium">N/A</span>
                        </div>
                      </div>
                    </div>
                    <div>
                      <h3 className="font-semibold mb-3">Writing Patterns</h3>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Peak Hours</span>
                          <span className="text-sm font-medium">N/A</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Avg. Session</span>
                          <span className="text-sm font-medium">N/A</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Best Day</span>
                          <span className="text-sm font-medium">N/A</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Productivity</span>
                          <span className="text-sm font-medium">N/A</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="goals" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Writing Goals Progress</CardTitle>
                <CardDescription>Track your progress toward writing improvement goals</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="space-y-6">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-gray-900">Loading...</span>
                        <span className="text-sm text-gray-600">Loading...</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-gray-300 h-2 rounded-full transition-all duration-300"
                          style={{ width: '0%' }}
                        />
                      </div>
                      <div className="text-xs text-gray-500">
                        Loading...
                      </div>
                    </div>
                  </div>
                ) : error ? (
                <div className="space-y-6">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-gray-900">Error</span>
                        <span className="text-sm text-gray-600">Error</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-red-200 h-2 rounded-full transition-all duration-300"
                          style={{ width: '0%' }}
                        />
                      </div>
                      <div className="text-xs text-gray-500">
                        Error
                      </div>
                    </div>
                  </div>
                ) : insightsData?.writing_goals ? (
                  <div className="space-y-6">
                    {insightsData.writing_goals.map((goal: any, index: number) => (
                      <div key={index} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-gray-900">{goal.goal}</span>
                        <span className="text-sm text-gray-600">{goal.current}% / {goal.target}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-green-500 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${(goal.current / goal.target) * 100}%` }}
                          />
                        </div>
                        <div className="text-xs text-gray-500">
                          {goal.current >= goal.target ? 'Goal achieved!' : `${goal.target - goal.current}% to go`}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-gray-900">No data</span>
                        <span className="text-sm text-gray-600">No data</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-gray-300 h-2 rounded-full transition-all duration-300"
                          style={{ width: '0%' }}
                        />
                      </div>
                      <div className="text-xs text-gray-500">
                        No data
                      </div>
                    </div>
                </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="achievements" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {achievements.map((achievement) => (
                <Card key={achievement.id} className={`${achievement.unlocked ? '' : 'opacity-50'}`}>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-3">
                      <span className="text-2xl">{achievement.icon}</span>
                      <div>
                        <div className="font-semibold">{achievement.title}</div>
                        <div className="text-sm text-gray-600">{achievement.description}</div>
                      </div>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Badge variant={achievement.unlocked ? 'default' : 'secondary'}>
                      {achievement.unlocked ? 'Unlocked' : 'Locked'}
                    </Badge>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}