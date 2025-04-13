
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  LineChart as RechartLineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Legend
} from 'recharts';

interface LineChartProps {
  data: any[];
  xKey: string;
  yKey: string | string[];
  title: string;
  description?: string;
  colors?: string[];
}

const defaultColors = ["#8B5CF6", "#F97316", "#0EA5E9", "#10B981", "#EC4899", "#7E69AB"];

const LineChart: React.FC<LineChartProps> = ({ 
  data, 
  xKey, 
  yKey, 
  title, 
  description,
  colors = defaultColors
}) => {
  // Ensure yKey is always an array
  const yKeys = Array.isArray(yKey) ? yKey : [yKey];
  
  // Make a copy to avoid modifying original
  const chartData = [...data];

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent className="pt-2">
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <RechartLineChart
              data={chartData}
              margin={{
                top: 5,
                right: 30,
                left: 20,
                bottom: 60,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis 
                dataKey={xKey} 
                angle={-45}
                textAnchor="end"
                height={70}
                tick={{ fontSize: 12 }}
              />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              {yKeys.map((key, index) => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={colors[index % colors.length]}
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                />
              ))}
            </RechartLineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default LineChart;
