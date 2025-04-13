
import React, { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { FileSpreadsheet, AlertCircle, Clock, BarChart2 } from 'lucide-react';
import { 
  calculateStatistics, 
  getNumericColumns,
  getCategoricalColumns,
  analyzeMissingValues
} from '@/lib/data-analysis';

interface DashboardOverviewProps {
  data: any[];
  fileName: string;
}

const DashboardOverview: React.FC<DashboardOverviewProps> = ({ data, fileName }) => {
  const metrics = useMemo(() => {
    if (!data || data.length === 0 || !data[0]) {
      return {
        rowCount: 0,
        columnCount: 0,
        numericColumns: 0,
        categoricalColumns: 0,
        missingValues: 0,
        missingPercent: 0
      };
    }
    
    const numericCols = getNumericColumns(data);
    const categoricalCols = getCategoricalColumns(data);
    const columnCount = Object.keys(data[0]).length;
    
    // Calculate missing values
    const missingValuesAnalysis = analyzeMissingValues(data);
    let totalMissing = 0;
    let totalCells = data.length * columnCount;
    
    if (missingValuesAnalysis && typeof missingValuesAnalysis === 'object') {
      Object.values(missingValuesAnalysis).forEach(info => {
        if (info && typeof info === 'object' && 'count' in info) {
          totalMissing += info.count as number;
        }
      });
    }
    
    const missingPercent = totalCells > 0 ? (totalMissing / totalCells) * 100 : 0;
    
    return {
      rowCount: data.length,
      columnCount: columnCount,
      numericColumns: numericCols.length,
      categoricalColumns: categoricalCols.length,
      missingValues: totalMissing,
      missingPercent: missingPercent
    };
  }, [data]);
  
  const { fileType, uploadTime } = useMemo(() => {
    const type = fileName.split('.').pop()?.toLowerCase() || '';
    const time = new Date().toLocaleTimeString();
    
    return {
      fileType: type === 'csv' ? 'CSV' : type === 'xlsx' || type === 'xls' ? 'Excel' : 'Unknown',
      uploadTime: time
    };
  }, [fileName]);
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Data Points
          </CardTitle>
          <FileSpreadsheet className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{metrics.rowCount.toLocaleString()}</div>
          <p className="text-xs text-muted-foreground mt-1">
            {metrics.columnCount} columns • {fileType} file
          </p>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Column Types
          </CardTitle>
          <BarChart2 className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{metrics.numericColumns} / {metrics.categoricalColumns}</div>
          <p className="text-xs text-muted-foreground mt-1">
            {metrics.numericColumns} numeric • {metrics.categoricalColumns} categorical
          </p>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Missing Values
          </CardTitle>
          <AlertCircle className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{metrics.missingValues.toLocaleString()}</div>
          <p className="text-xs text-muted-foreground mt-1">
            {metrics.missingPercent.toFixed(1)}% of all cells are missing
          </p>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Upload Time
          </CardTitle>
          <Clock className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{uploadTime}</div>
          <p className="text-xs text-muted-foreground mt-1">
            File processed on {new Date().toLocaleDateString()}
          </p>
        </CardContent>
      </Card>
    </div>
  );
};

export default DashboardOverview;
