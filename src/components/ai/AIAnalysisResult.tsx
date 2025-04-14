
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Lightbulb, CodeSquare, Sparkles, ChevronDown, ChevronUp, LineChart as LineChartIcon, TableIcon, BarChart2, PieChart as PieChartIcon } from 'lucide-react';
import { cn } from '@/lib/utils';
import LineChart from '../visualizations/LineChart';
import BarChart from '../visualizations/BarChart';
import PieChart from '../visualizations/PieChart';
import StatisticsCard from '../visualizations/StatisticsCard';
import DataTable from '../visualizations/DataTable';
import { Button } from '../ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';

interface AIAnalysisResultProps {
  result: {
    query: string;
    response: {
      type: string;
      title: string;
      description: string;
      pythonCode?: string;
      modelInfo?: string;
      data?: any[];
      chartType?: 'bar' | 'line' | 'pie' | 'table' | 'stats';
      chartConfig?: any;
      codeOutput?: string;
      usedGemini?: boolean;
      visualizations?: Array<{
        type: 'bar' | 'line' | 'pie' | 'table' | 'stats';
        title: string;
        data: any[];
        config?: any;
      }>;
    };
    timestamp: Date;
  } | null;
}

const AIAnalysisResult: React.FC<AIAnalysisResultProps> = ({ result }) => {
  const [expanded, setExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState<string>('overview');
  
  if (!result) return null;

  const { response } = result;
  
  // Render the appropriate chart based on type
  const renderVisualization = (chartType: string | undefined, data: any[] | undefined, config: any) => {
    if (!chartType || !data) return null;

    try {
      switch (chartType) {
        case 'bar':
          return (
            <BarChart
              data={data}
              xKey={config?.xKey || 'name'}
              yKey={config?.yKey || 'value'}
              title={config?.title || 'Bar Chart'}
              description={config?.description}
              color={config?.color || '#8B5CF6'}
            />
          );
        case 'line':
          return (
            <LineChart
              data={data}
              xKey={config?.xKey || 'name'}
              yKey={config?.yKey || 'value'}
              title={config?.title || 'Line Chart'}
              description={config?.description}
              colors={config?.colors}
            />
          );
        case 'pie':
          return (
            <PieChart
              data={data.map((item: any) => ({
                name: item[config?.nameKey || 'name'] || item.name,
                value: item[config?.valueKey || 'value'] || item.value,
              }))}
              title={config?.title || 'Pie Chart'}
              description={config?.description}
              colors={config?.colors}
            />
          );
        case 'stats':
          return (
            <StatisticsCard
              title={config?.title || 'Statistics'}
              description={config?.description}
              statistics={data}
            />
          );
        case 'table':
          return (
            <DataTable
              data={data}
              title={config?.title || 'Data Table'}
              description={config?.description}
            />
          );
        default:
          return null;
      }
    } catch (error) {
      console.error("Error rendering visualization:", error);
      return (
        <Card className="bg-red-50 border-red-200">
          <CardContent className="p-4">
            <p className="text-red-600">
              Error rendering visualization. The data format might not be compatible with the selected chart type.
            </p>
          </CardContent>
        </Card>
      );
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {response.usedGemini ? (
              <Sparkles className="h-5 w-5 text-primary" />
            ) : (
              <Lightbulb className="h-5 w-5 text-insight-400" />
            )}
            AI Insight Results {response.usedGemini && <span className="ml-2 text-xs bg-primary/10 text-primary px-2 py-0.5 rounded-full">Gemini</span>}
          </CardTitle>
          <CardDescription>
            Results for: "{result.query}"
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid grid-cols-4 mb-4">
              <TabsTrigger value="overview" className="text-xs">
                Overview
              </TabsTrigger>
              <TabsTrigger value="code" className="text-xs">
                Python Code
              </TabsTrigger>
              <TabsTrigger value="output" className="text-xs">
                Code Output
              </TabsTrigger>
              <TabsTrigger value="visualizations" className="text-xs">
                Visualizations
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="overview" className="prose max-w-none">
              <div className="text-lg font-medium text-insight-600 mb-2">
                {response.title}
              </div>
              <p className="text-gray-700 whitespace-pre-line">{response.description}</p>
              
              {response.modelInfo && (
                <div className="mt-4 p-3 bg-blue-50 rounded-md">
                  <p className="text-blue-700 text-sm font-medium">AI Model Information</p>
                  <p className="text-blue-600 text-sm">{response.modelInfo}</p>
                </div>
              )}
            </TabsContent>
            
            <TabsContent value="code">
              {response.pythonCode ? (
                <div className="mt-4">
                  <div className="flex items-center gap-2 text-gray-700 mb-1">
                    <CodeSquare className="h-4 w-4" />
                    <span className="text-sm font-medium">Python Analysis Code</span>
                  </div>
                  <pre className="p-3 bg-gray-50 rounded-md overflow-auto text-xs border max-h-96">
                    <code>{response.pythonCode}</code>
                  </pre>
                  <div className="mt-2 flex justify-end">
                    <Button variant="outline" size="sm" onClick={() => {
                      navigator.clipboard.writeText(response.pythonCode || '');
                    }} className="text-xs">
                      Copy Code
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-8">
                  <CodeSquare className="h-8 w-8 text-gray-400 mb-2" />
                  <p className="text-gray-500">No Python code was generated for this analysis</p>
                </div>
              )}
            </TabsContent>
            
            <TabsContent value="output">
              {response.codeOutput ? (
                <div className="mt-4">
                  <div className="flex items-center gap-2 text-gray-700 mb-1">
                    <TableIcon className="h-4 w-4" />
                    <span className="text-sm font-medium">Code Execution Output</span>
                  </div>
                  <pre className="p-3 bg-gray-50 rounded-md overflow-auto text-xs border max-h-96">
                    <code>{response.codeOutput}</code>
                  </pre>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-8">
                  <TableIcon className="h-8 w-8 text-gray-400 mb-2" />
                  <p className="text-gray-500">No code output is available for this analysis</p>
                </div>
              )}
            </TabsContent>
            
            <TabsContent value="visualizations">
              <div className="space-y-6">
                {response.chartType && response.data ? (
                  renderVisualization(response.chartType, response.data, response.chartConfig)
                ) : response.visualizations && response.visualizations.length > 0 ? (
                  response.visualizations.map((viz, idx) => (
                    <div key={`viz-${idx}`}>
                      {renderVisualization(viz.type, viz.data, viz.config)}
                    </div>
                  ))
                ) : (
                  <div className="flex flex-col items-center justify-center py-8">
                    <BarChart2 className="h-8 w-8 text-gray-400 mb-2" />
                    <p className="text-gray-500">No visualizations are available for this analysis</p>
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

export default AIAnalysisResult;
