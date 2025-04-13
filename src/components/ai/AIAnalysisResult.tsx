
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Lightbulb, BarChart3, LineChartIcon, PieChartIcon } from 'lucide-react';
import { cn } from '@/lib/utils';
import LineChart from '../visualizations/LineChart';
import BarChart from '../visualizations/BarChart';
import PieChartComponent from '../visualizations/PieChart';
import StatisticsCard from '../visualizations/StatisticsCard';
import DataTable from '../visualizations/DataTable';

interface AIAnalysisResultProps {
  result: {
    query: string;
    response: {
      type: string;
      title: string;
      description: string;
      data?: any;
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
    if (!response.chartType || !response.data) return null;

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
            yKey={response.chartConfig?.yKeys || 'value'}
            title={response.chartConfig?.title || 'Line Chart'}
            description={response.chartConfig?.description}
            colors={response.chartConfig?.colors}
          />
        );
      case 'pie':
        return (
          <PieChartComponent
            data={response.data.map((item: any) => ({
              name: item[response.chartConfig?.nameKey || 'name'],
              value: item[response.chartConfig?.valueKey || 'value'],
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
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-insight-400" />
            AI Analysis Results
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
        </CardContent>
      </Card>

      <div className="mt-4 overflow-hidden">
        {renderVisualization()}
      </div>
    </div>
  );
};

export default AIAnalysisResult;
