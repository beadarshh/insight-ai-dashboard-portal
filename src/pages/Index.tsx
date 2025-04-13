
import React, { useState, useRef, useEffect } from 'react';
import { toast } from 'sonner';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  FileSpreadsheet, 
  LineChart as LineChartIcon, 
  BarChart2, 
  PieChart as PieChartIcon, 
  Sparkles,
  Download,
  RefreshCw
} from 'lucide-react';

import DashboardLayout from '@/components/layout/DashboardLayout';
import FileUpload from '@/components/upload/FileUpload';
import DashboardOverview from '@/components/dashboard/DashboardOverview';
import DataTable from '@/components/visualizations/DataTable';
import BarChart from '@/components/visualizations/BarChart';
import LineChart from '@/components/visualizations/LineChart';
import PieChart from '@/components/visualizations/PieChart';
import StatisticsCard from '@/components/visualizations/StatisticsCard';
import AIAnalysisPrompt from '@/components/ai/AIAnalysisPrompt';
import AIAnalysisResult from '@/components/ai/AIAnalysisResult';
import { useAuth } from '@/providers/AuthProvider';
import { useData } from '@/providers/DataProvider';

import { 
  calculateStatistics, 
  getFrequencyDistribution,
  getNumericColumns,
  getCategoricalColumns,
  detectColumnTypes,
  getAutomatedInsights
} from '@/lib/data-analysis';
import { cn } from '@/lib/utils';

// Mock function to simulate AI analysis
const mockAIAnalysis = async (prompt: string, data: any[]) => {
  // Simulate processing time
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  const lowerPrompt = prompt.toLowerCase();
  
  // Detect column types
  const columnTypes = detectColumnTypes(data);
  const numericColumns = getNumericColumns(data);
  const categoricalColumns = getCategoricalColumns(data);
  
  // Logic for different types of analysis
  if (lowerPrompt.includes('summary') || lowerPrompt.includes('summarize') || lowerPrompt.includes('overview')) {
    return {
      type: 'summary',
      title: 'Dataset Summary',
      description: `This dataset contains ${data.length} rows and ${Object.keys(data[0]).length} columns. It includes ${numericColumns.length} numeric columns and ${categoricalColumns.length} categorical columns.`,
      chartType: 'table',
      data: data.slice(0, 10),
      chartConfig: {
        title: 'Data Sample',
        description: 'First 10 rows of the dataset'
      }
    };
  }
  
  if (lowerPrompt.includes('distribution') || lowerPrompt.includes('frequency')) {
    // Find a categorical column to analyze
    const targetColumn = categoricalColumns[0];
    if (targetColumn) {
      const distribution = getFrequencyDistribution(data, targetColumn, 10);
      return {
        type: 'distribution',
        title: `Distribution of ${targetColumn}`,
        description: `This chart shows the frequency distribution of values in the ${targetColumn} column.`,
        chartType: 'bar',
        data: distribution,
        chartConfig: {
          xKey: 'value',
          yKey: 'count',
          title: `Distribution of ${targetColumn}`,
          description: 'Top 10 values by frequency'
        }
      };
    }
  }
  
  if (lowerPrompt.includes('trend') || lowerPrompt.includes('over time') || lowerPrompt.includes('monthly') || 
      lowerPrompt.includes('yearly') || lowerPrompt.includes('time series')) {
    // Look for datetime columns
    const dateColumns = Object.entries(columnTypes)
      .filter(([_, type]) => type === 'date' || type === 'datetime')
      .map(([col, _]) => col);
    
    if (dateColumns.length > 0 && numericColumns.length > 0) {
      // Create a time series
      const dateColumn = dateColumns[0];
      const valueColumn = numericColumns[0];
      
      // Prepare data for time series (simplified mock)
      const timeSeriesData = data
        .slice(0, 20) // Just use a sample for mock data
        .map(row => ({
          date: row[dateColumn],
          value: row[valueColumn]
        }))
        .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
      
      return {
        type: 'trend',
        title: `${valueColumn} Over Time`,
        description: `This chart shows the trend of ${valueColumn} over time.`,
        chartType: 'line',
        data: timeSeriesData,
        chartConfig: {
          xKey: 'date',
          yKeys: ['value'],
          title: `${valueColumn} Trends`,
          description: `Values of ${valueColumn} over time`
        }
      };
    }
  }
  
  if (lowerPrompt.includes('comparison') || lowerPrompt.includes('compare') || lowerPrompt.includes('versus') || 
      lowerPrompt.includes('vs') || lowerPrompt.includes('by')) {
    if (categoricalColumns.length > 0 && numericColumns.length > 0) {
      const categoryColumn = categoricalColumns[0];
      const valueColumn = numericColumns[0];
      
      // Create aggregated data for comparison
      const categories = [...new Set(data.map(row => row[categoryColumn]))].slice(0, 10);
      const comparisonData = categories.map(category => {
        const filteredRows = data.filter(row => row[categoryColumn] === category);
        const sum = filteredRows.reduce((acc, row) => acc + (Number(row[valueColumn]) || 0), 0);
        
        return {
          category,
          value: sum
        };
      })
      .sort((a, b) => b.value - a.value);
      
      return {
        type: 'comparison',
        title: `${valueColumn} by ${categoryColumn}`,
        description: `This chart compares the total ${valueColumn} across different ${categoryColumn} categories.`,
        chartType: 'bar',
        data: comparisonData,
        chartConfig: {
          xKey: 'category',
          yKey: 'value',
          title: `${valueColumn} by ${categoryColumn}`,
          description: `Comparison of total ${valueColumn} across ${categoryColumn} categories`
        }
      };
    }
  }
  
  // Default response if no specific analysis is matched
  if (numericColumns.length > 0) {
    const column = numericColumns[0];
    const stats = calculateStatistics(data, column);
    
    if (stats) {
      return {
        type: 'statistics',
        title: `Statistical Analysis of ${column}`,
        description: `Basic statistical measures for the ${column} column.`,
        chartType: 'stats',
        data: [
          { label: 'Mean', value: stats.mean },
          { label: 'Median', value: stats.median },
          { label: 'Min', value: stats.min },
          { label: 'Max', value: stats.max },
          { label: 'Standard Deviation', value: stats.stdDev },
          { label: 'Count', value: stats.count }
        ],
        chartConfig: {
          title: `Statistics for ${column}`,
          description: 'Basic statistical measures'
        }
      };
    }
  }
  
  // Generic fallback
  return {
    type: 'generic',
    title: 'Data Analysis',
    description: 'I analyzed your data but couldn\'t determine a specific visualization that matches your prompt. Please try a more specific query.',
    chartType: 'table',
    data: data.slice(0, 10),
    chartConfig: {
      title: 'Data Sample',
      description: 'First 10 rows of the dataset'
    }
  };
};

const Index = () => {
  const { user } = useAuth();
  const { addFile, addAnalysis, setCurrentData } = useData();
  
  // State for uploaded data
  const [data, setData] = useState<any[]>([]);
  const [fileName, setFileName] = useState<string>('');
  const [sheetNames, setSheetNames] = useState<string[]>([]);
  
  // State for AI analysis
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [aiResult, setAiResult] = useState<any>(null);
  
  // Auto-generate visualizations when data is loaded
  const [automaticVisualizations, setAutomaticVisualizations] = useState<any[]>([]);
  
  // Ref to hold data for downloads
  const dataRef = useRef<any[]>([]);
  
  // Update ref when data changes
  useEffect(() => {
    dataRef.current = data;
  }, [data]);
  
  // Handle file upload
  const handleDataLoaded = (newData: any[], sheets: string[], name: string) => {
    setData(newData);
    setSheetNames(sheets);
    setFileName(name);
    
    // Set data in the global state
    setCurrentData(newData, name);
    
    // Save file to history
    if (user) {
      addFile({
        fileName: name,
        uploadDate: new Date(),
        rowCount: newData.length,
        columnCount: Object.keys(newData[0] || {}).length,
        fileSize: `${Math.round((JSON.stringify(newData).length / 1024) * 10) / 10} KB`
      });
    }
    
    // Reset AI analysis
    setAiResult(null);
    
    // Generate automatic visualizations
    generateAutomaticVisualizations(newData);
  };
  
  // Generate automatic visualizations based on data
  const generateAutomaticVisualizations = (data: any[]) => {
    if (data.length === 0) return;
    
    const visualizations = [];
    const numericColumns = getNumericColumns(data);
    const categoricalColumns = getCategoricalColumns(data);
    
    // 1. If we have categorical columns, create a frequency distribution
    if (categoricalColumns.length > 0) {
      const catColumn = categoricalColumns[0];
      const distribution = getFrequencyDistribution(data, catColumn, 10);
      
      visualizations.push({
        type: 'bar',
        title: `Distribution of ${catColumn}`,
        data: distribution,
        config: {
          xKey: 'value',
          yKey: 'count'
        }
      });
      
      // If we have a second categorical column and at least one numeric column,
      // create a grouped visualization
      if (categoricalColumns.length > 1 && numericColumns.length > 0) {
        const catColumn2 = categoricalColumns[1];
        const numColumn = numericColumns[0];
        
        // Get top 5 categories from each categorical column
        const topCats1 = getFrequencyDistribution(data, catColumn, 5).map(d => d.value);
        const topCats2 = getFrequencyDistribution(data, catColumn2, 5).map(d => d.value);
        
        // Filter data to only include top categories
        const filteredData = data.filter(row => 
          topCats1.includes(row[catColumn]) && topCats2.includes(row[catColumn2])
        );
        
        // Group by first category and calculate average of numeric column for each group
        const groupedData: Record<string, any> = {};
        
        filteredData.forEach(row => {
          const cat1 = String(row[catColumn]);
          const cat2 = String(row[catColumn2]);
          const num = Number(row[numColumn]) || 0;
          
          if (!groupedData[cat1]) {
            groupedData[cat1] = { [catColumn]: cat1 };
          }
          
          if (!groupedData[cat1][cat2]) {
            groupedData[cat1][cat2] = num;
          } else {
            groupedData[cat1][cat2] += num;
          }
        });
        
        const multiSeriesData = Object.values(groupedData);
        
        if (multiSeriesData.length > 0) {
          visualizations.push({
            type: 'bar_multi',
            title: `${numColumn} by ${catColumn} and ${catColumn2}`,
            data: multiSeriesData,
            config: {
              xKey: catColumn,
              yKeys: topCats2
            }
          });
        }
      }
    }
    
    // 2. If we have numeric columns, create a histogram-like visualization for the first one
    if (numericColumns.length > 0) {
      const numColumn = numericColumns[0];
      const stats = calculateStatistics(data, numColumn);
      
      if (stats) {
        visualizations.push({
          type: 'stats',
          title: `Statistics for ${numColumn}`,
          data: [
            { label: 'Mean', value: stats.mean },
            { label: 'Median', value: stats.median },
            { label: 'Min', value: stats.min },
            { label: 'Max', value: stats.max },
            { label: 'Standard Deviation', value: stats.stdDev }
          ]
        });
      }
      
      // If we have more than one numeric column, create a scatter plot
      if (numericColumns.length > 1) {
        const numColumn2 = numericColumns[1];
        
        const scatterData = data
          .filter(row => 
            row[numColumn] !== null && row[numColumn] !== undefined && 
            row[numColumn2] !== null && row[numColumn2] !== undefined
          )
          .slice(0, 100) // Limit to 100 points for performance
          .map(row => ({
            x: Number(row[numColumn]),
            y: Number(row[numColumn2])
          }));
        
        if (scatterData.length > 10) {
          visualizations.push({
            type: 'scatter',
            title: `${numColumn} vs ${numColumn2}`,
            data: scatterData,
            config: {
              xKey: 'x',
              yKey: 'y',
              xLabel: numColumn,
              yLabel: numColumn2
            }
          });
        }
      }
    }
    
    // 3. If we have date columns and numeric columns, create a time series
    const columnTypes = detectColumnTypes(data);
    const dateColumns = Object.entries(columnTypes)
      .filter(([_, type]) => type === 'date' || type === 'datetime')
      .map(([col, _]) => col);
    
    if (dateColumns.length > 0 && numericColumns.length > 0) {
      const dateColumn = dateColumns[0];
      const numColumn = numericColumns[0];
      
      const timeSeriesData = data
        .filter(row => row[dateColumn] && row[numColumn] !== null && row[numColumn] !== undefined)
        .slice(0, 100)
        .map(row => ({
          date: row[dateColumn],
          value: Number(row[numColumn])
        }))
        .sort((a, b) => {
          const dateA = new Date(a.date);
          const dateB = new Date(b.date);
          return dateA.getTime() - dateB.getTime();
        });
      
      if (timeSeriesData.length > 5) {
        visualizations.push({
          type: 'line',
          title: `${numColumn} Over Time`,
          data: timeSeriesData,
          config: {
            xKey: 'date',
            yKey: 'value'
          }
        });
      }
    }
    
    // 4. Create a pie chart for a categorical column if available
    if (categoricalColumns.length > 0) {
      const catColumn = categoricalColumns[0];
      const distribution = getFrequencyDistribution(data, catColumn, 8);
      
      visualizations.push({
        type: 'pie',
        title: `Distribution of ${catColumn}`,
        data: distribution.map(item => ({
          name: item.value,
          value: item.count
        }))
      });
    }
    
    setAutomaticVisualizations(visualizations);
  };
  
  // Handle AI analysis request
  const handleAIAnalysis = async (prompt: string) => {
    if (data.length === 0) {
      toast.error("Please upload data before requesting AI analysis");
      return;
    }
    
    setIsAnalyzing(true);
    
    try {
      // In a real application, this would call an API endpoint
      const result = await mockAIAnalysis(prompt, data);
      
      const analysisResult = {
        query: prompt,
        response: result,
        timestamp: new Date()
      };
      
      setAiResult(analysisResult);
      
      // Save analysis to history if user is logged in
      if (user) {
        addAnalysis({
          query: prompt,
          timestamp: new Date(),
          fileId: 'current',
          fileName: fileName,
          resultType: result.type
        });
      }
      
      toast.success("AI analysis complete!");
    } catch (error) {
      console.error("Error in AI analysis:", error);
      toast.error("Failed to generate AI analysis");
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  // Handle export to CSV
  const handleExportCSV = () => {
    if (dataRef.current.length === 0) {
      toast.error("No data available to export");
      return;
    }
    
    try {
      const columns = Object.keys(dataRef.current[0]);
      const csvContent = [
        columns.join(','),
        ...dataRef.current.map(row => 
          columns.map(col => {
            const value = row[col];
            // Handle strings with commas, quotes, etc.
            if (typeof value === 'string' && (value.includes(',') || value.includes('"') || value.includes('\n'))) {
              return `"${value.replace(/"/g, '""')}"`;
            }
            return value !== null && value !== undefined ? value : '';
          }).join(',')
        )
      ].join('\n');
      
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${fileName.split('.')[0] || 'export'}_analysis.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      toast.success("CSV file exported successfully");
    } catch (error) {
      console.error("Error exporting CSV:", error);
      toast.error("Failed to export CSV file");
    }
  };
  
  // Render automatic visualizations
  const renderAutomaticVisualizations = () => {
    if (automaticVisualizations.length === 0) return null;
    
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {automaticVisualizations.map((viz, index) => {
          switch (viz.type) {
            case 'bar':
              return (
                <BarChart
                  key={`viz-${index}`}
                  data={viz.data}
                  xKey={viz.config.xKey}
                  yKey={viz.config.yKey}
                  title={viz.title}
                />
              );
            
            case 'bar_multi': 
              return (
                <BarChart
                  key={`viz-${index}`}
                  data={viz.data}
                  xKey={viz.config.xKey}
                  yKey={viz.config.yKeys[0]}
                  title={viz.title}
                />
              );
              
            case 'line':
              return (
                <LineChart
                  key={`viz-${index}`}
                  data={viz.data}
                  xKey={viz.config.xKey}
                  yKey={viz.config.yKey}
                  title={viz.title}
                />
              );
              
            case 'pie':
              return (
                <PieChart
                  key={`viz-${index}`}
                  data={viz.data}
                  title={viz.title}
                />
              );
              
            case 'stats':
              return (
                <StatisticsCard
                  key={`viz-${index}`}
                  statistics={viz.data}
                  title={viz.title}
                />
              );
              
            default:
              return null;
          }
        })}
      </div>
    );
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {data.length === 0 ? (
          <Card className="border-dashed">
            <CardHeader>
              <CardTitle>Welcome to InsightAI Dashboard</CardTitle>
              <CardDescription>
                Upload an Excel file to start exploring your data and generating insights
              </CardDescription>
            </CardHeader>
            <CardContent>
              <FileUpload onDataLoaded={handleDataLoaded} />
            </CardContent>
          </Card>
        ) : (
          <>
            {/* Dashboard Header with actions */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
              <div>
                <h1 className="text-2xl font-bold tracking-tight">{fileName}</h1>
                <p className="text-muted-foreground">
                  {data.length.toLocaleString()} rows • {Object.keys(data[0]).length} columns • {sheetNames.length} sheets
                </p>
              </div>
              
              <div className="flex gap-2">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => {
                    setData([]);
                    setFileName('');
                    setSheetNames([]);
                    setAiResult(null);
                    setAutomaticVisualizations([]);
                    setCurrentData(null, null);
                  }}
                  className="flex items-center gap-2"
                >
                  <RefreshCw className="h-4 w-4" />
                  <span>New Upload</span>
                </Button>
                
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={handleExportCSV}
                  className="flex items-center gap-2"
                >
                  <Download className="h-4 w-4" />
                  <span>Export CSV</span>
                </Button>
              </div>
            </div>
            
            {/* Dashboard content */}
            <Tabs defaultValue="overview" className="space-y-6">
              <div className="overflow-x-auto">
                <TabsList className="grid grid-cols-5 md:w-[600px]">
                  <TabsTrigger value="overview" className="flex items-center gap-2">
                    <FileSpreadsheet className="h-4 w-4" />
                    <span className="hidden sm:inline">Overview</span>
                  </TabsTrigger>
                  <TabsTrigger value="visualizations" className="flex items-center gap-2">
                    <BarChart2 className="h-4 w-4" />
                    <span className="hidden sm:inline">Visualizations</span>
                  </TabsTrigger>
                  <TabsTrigger value="data" className="flex items-center gap-2">
                    <LineChartIcon className="h-4 w-4" />
                    <span className="hidden sm:inline">Data</span>
                  </TabsTrigger>
                  <TabsTrigger value="statistics" className="flex items-center gap-2">
                    <PieChartIcon className="h-4 w-4" />
                    <span className="hidden sm:inline">Statistics</span>
                  </TabsTrigger>
                  <TabsTrigger value="ai-insights" className="flex items-center gap-2">
                    <Sparkles className="h-4 w-4" />
                    <span className="hidden sm:inline">AI Insights</span>
                  </TabsTrigger>
                </TabsList>
              </div>
              
              {/* Overview Tab */}
              <TabsContent value="overview" className="space-y-6">
                <DashboardOverview data={data} fileName={fileName} />
                
                <div className="mt-6">
                  <h2 className="text-xl font-semibold mb-4">Data Preview</h2>
                  <div className="overflow-hidden rounded-lg border">
                    <DataTable 
                      data={data.slice(0, 10)} 
                      title="First 10 Rows" 
                    />
                  </div>
                </div>
              </TabsContent>
              
              {/* Visualizations Tab */}
              <TabsContent value="visualizations" className="space-y-6">
                <h2 className="text-xl font-semibold mb-4">Automated Visualizations</h2>
                {renderAutomaticVisualizations()}
              </TabsContent>
              
              {/* Data Tab */}
              <TabsContent value="data" className="space-y-6 overflow-hidden">
                <div className="overflow-hidden rounded-lg border">
                  <DataTable 
                    data={data} 
                    title="Complete Dataset" 
                  />
                </div>
              </TabsContent>
              
              {/* Statistics Tab */}
              <TabsContent value="statistics" className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {getNumericColumns(data).slice(0, 4).map(column => {
                    const stats = calculateStatistics(data, column);
                    if (!stats) return null;
                    
                    return (
                      <StatisticsCard
                        key={column}
                        title={`Statistics for ${column}`}
                        statistics={[
                          { label: 'Mean', value: stats.mean },
                          { label: 'Median', value: stats.median },
                          { label: 'Minimum', value: stats.min },
                          { label: 'Maximum', value: stats.max },
                          { label: 'Standard Deviation', value: stats.stdDev },
                          { label: 'Count', value: stats.count }
                        ]}
                      />
                    );
                  })}
                </div>
              </TabsContent>
              
              {/* AI Insights Tab */}
              <TabsContent value="ai-insights" className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="lg:col-span-1">
                    <AIAnalysisPrompt 
                      onAnalyze={handleAIAnalysis} 
                      isLoading={isAnalyzing}
                    />
                  </div>
                  
                  <div className="lg:col-span-2">
                    {aiResult ? (
                      <div className="overflow-hidden">
                        <AIAnalysisResult result={aiResult} />
                      </div>
                    ) : (
                      <Card className="h-full flex items-center justify-center bg-muted/50">
                        <CardContent className="py-12 text-center">
                          <div className="mx-auto w-12 h-12 rounded-full bg-insight-100 flex items-center justify-center mb-4">
                            <Sparkles className="h-6 w-6 text-insight-600" />
                          </div>
                          <h3 className="text-lg font-medium mb-2">AI Analysis</h3>
                          <p className="text-muted-foreground max-w-md mx-auto">
                            Ask a question about your data to generate AI-powered insights and visualizations
                          </p>
                        </CardContent>
                      </Card>
                    )}
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </>
        )}
      </div>
    </DashboardLayout>
  );
};

export default Index;
