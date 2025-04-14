
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Lightbulb, CodeSquare } from 'lucide-react';
import { cn } from '@/lib/utils';
import LineChart from '../visualizations/LineChart';
import BarChart from '../visualizations/BarChart';
import PieChart from '../visualizations/PieChart';
import StatisticsCard from '../visualizations/StatisticsCard';
import DataTable from '../visualizations/DataTable';

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
    };
    timestamp: Date;
  } | null;
}

const AIAnalysisResult: React.FC<AIAnalysisResultProps> = ({ result }) => {
  if (!result) return null;

  const { response } = result;
  
  // Render the appropriate chart based on type
  const renderVisualization = () => {
    if (!response?.chartType || !response?.data) return null;

    try {
      switch (response.chartType) {
        case 'bar':
          return (
            <BarChart
              data={response.data}
              xKey={response.chartConfig?.xKey || 'name'}
              yKey={response.chartConfig?.yKey || 'value'}
              title={response.chartConfig?.title || 'Bar Chart'}
              description={response.chartConfig?.description}
              color={response.chartConfig?.color || '#8B5CF6'}
            />
          );
        case 'line':
          return (
            <LineChart
              data={response.data}
              xKey={response.chartConfig?.xKey || 'name'}
              yKey={response.chartConfig?.yKey || 'value'}
              title={response.chartConfig?.title || 'Line Chart'}
              description={response.chartConfig?.description}
              colors={response.chartConfig?.colors}
            />
          );
        case 'pie':
          return (
            <PieChart
              data={response.data.map((item: any) => ({
                name: item[response.chartConfig?.nameKey || 'name'] || item.name,
                value: item[response.chartConfig?.valueKey || 'value'] || item.value,
              }))}
              title={response.chartConfig?.title || 'Pie Chart'}
              description={response.chartConfig?.description}
              colors={response.chartConfig?.colors}
            />
          );
        case 'stats':
          return (
            <StatisticsCard
              title={response.chartConfig?.title || 'Statistics'}
              description={response.chartConfig?.description}
              statistics={response.data}
            />
          );
        case 'table':
          return (
            <DataTable
              data={response.data}
              title={response.chartConfig?.title || 'Data Table'}
              description={response.chartConfig?.description}
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
            <Lightbulb className="h-5 w-5 text-insight-400" />
            AI Insight Results
          </CardTitle>
          <CardDescription>
            Results for: "{result.query}"
          </CardDescription>
        </CardHeader>
        <CardContent className="prose max-w-none">
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
          
          {response.pythonCode && (
            <div className="mt-4">
              <div className="flex items-center gap-2 text-gray-700 mb-1">
                <CodeSquare className="h-4 w-4" />
                <span className="text-sm font-medium">Python Analysis Code</span>
              </div>
              <pre className="p-3 bg-gray-50 rounded-md overflow-auto text-xs border">
                <code>{response.pythonCode}</code>
              </pre>
            </div>
          )}
        </CardContent>
      </Card>

      {renderVisualization()}
    </div>
  );
};

export default AIAnalysisResult;
