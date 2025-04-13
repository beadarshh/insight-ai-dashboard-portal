
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart as RechartBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface BarChartProps {
  data: any[];
  xKey: string;
  yKey: string;
  title: string;
  description?: string;
  color?: string;
}

const BarChart: React.FC<BarChartProps> = ({ 
  data, 
  xKey, 
  yKey, 
  title, 
  description,
  color = "#8B5CF6" 
}) => {
  // Make a copy to avoid modifying original data
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
            <RechartBarChart
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
              <Bar dataKey={yKey} fill={color} />
            </RechartBarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default BarChart;
