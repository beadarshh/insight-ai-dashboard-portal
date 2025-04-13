
import React from 'react';
import { 
  File, BarChart3, PieChart, Table2, AlertCircle, 
  Info, Layers, Battery 
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { detectColumnTypes, analyzeMissingValues, getNumericColumns, getCategoricalColumns } from '@/lib/data-analysis';
import { cn } from '@/lib/utils';

interface DashboardOverviewProps {
  data: any[];
  fileName: string;
}

const DashboardOverview: React.FC<DashboardOverviewProps> = ({ data, fileName }) => {
  // Data overview analysis
  const columnTypes = detectColumnTypes(data);
  const missingValues = analyzeMissingValues(data);
  const numericColumns = getNumericColumns(data);
  const categoricalColumns = getCategoricalColumns(data);
  
  // Calculate column type counts
  const columnTypeCounts = Object.values(columnTypes).reduce((acc: Record<string, number>, type) => {
    acc[type] = (acc[type] || 0) + 1;
    return acc;
  }, {});
  
  // Calculate data completeness
  const totalCells = data.length * Object.keys(data[0]).length;
  const missingCells = Object.values(missingValues).reduce((sum, { count }) => sum + count, 0);
  const completenessPercentage = ((totalCells - missingCells) / totalCells) * 100;
  
  // Find columns with high missing values
  const columnsWithHighMissing = Object.entries(missingValues)
    .filter(([_, info]) => info.percentage > 20)
    .sort((a, b) => b[1].percentage - a[1].percentage)
    .slice(0, 5);

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {/* File Information */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Dataset Overview</CardTitle>
          <File className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="space-y-2">
              <p className="text-xl font-bold">{fileName}</p>
              <p className="text-xs text-muted-foreground">
                {new Date().toLocaleDateString()}
              </p>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Rows</p>
                <p className="text-lg font-bold">{data.length.toLocaleString()}</p>
              </div>
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Columns</p>
                <p className="text-lg font-bold">{Object.keys(data[0]).length}</p>
              </div>
            </div>
            
            <Separator />
            
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Data Completeness</span>
                <span className="font-medium">{completenessPercentage.toFixed(1)}%</span>
              </div>
              <Progress value={completenessPercentage} className="h-2" />
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Column Types */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Column Types</CardTitle>
          <Layers className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 rounded-full bg-insight-400"></div>
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">Numeric</p>
                  <p className="text-lg font-bold">{numericColumns.length}</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">Categorical</p>
                  <p className="text-lg font-bold">{categoricalColumns.length}</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">Date/Time</p>
                  <p className="text-lg font-bold">
                    {Object.values(columnTypes).filter(type => 
                      type === 'date' || type === 'datetime'
                    ).length}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 rounded-full bg-gray-500"></div>
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">Other</p>
                  <p className="text-lg font-bold">
                    {Object.keys(data[0]).length - numericColumns.length - 
                     categoricalColumns.length - 
                     Object.values(columnTypes).filter(type => 
                       type === 'date' || type === 'datetime'
                     ).length}
                  </p>
                </div>
              </div>
            </div>
            
            <Separator />
            
            <div className="space-y-2">
              <p className="text-xs font-medium">Top Numeric Columns</p>
              <div className="space-y-1">
                {numericColumns.slice(0, 3).map(column => (
                  <p key={column} className="text-xs truncate">• {column}</p>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Data Quality */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Data Quality Issues</CardTitle>
          <AlertCircle className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {columnsWithHighMissing.length > 0 ? (
              <div className="space-y-3">
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">Columns with missing values</p>
                  <p className="text-lg font-bold text-amber-500">
                    {columnsWithHighMissing.length} columns
                  </p>
                </div>
                
                <div className="space-y-2">
                  {columnsWithHighMissing.map(([column, info]) => (
                    <div key={column} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span 
                          className="text-muted-foreground truncate max-w-[150px]" 
                          title={column}
                        >
                          {column}
                        </span>
                        <span className={cn(
                          "font-medium",
                          info.percentage > 50 ? "text-red-500" : "text-amber-500"
                        )}>
                          {info.percentage.toFixed(1)}%
                        </span>
                      </div>
                      <Progress 
                        value={info.percentage} 
                        className={cn(
                          "h-1",
                          info.percentage > 50 ? "bg-red-100" : "bg-amber-100"
                        )} 
                      />
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-32 text-center">
                <div className="space-y-2">
                  <div className="mx-auto w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                    <Battery className="h-5 w-5 text-green-600" />
                  </div>
                  <p className="text-sm font-medium text-green-600">No significant data quality issues detected</p>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default DashboardOverview;
