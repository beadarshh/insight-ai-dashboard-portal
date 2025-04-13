
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { PieChart as RechartPieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

interface PieChartProps {
  data: Array<{ name: string; value: number }>;
  title: string;
  description?: string;
  colors?: string[];
}

const defaultColors = [
  "#8B5CF6", // Purple
  "#F97316", // Orange
  "#0EA5E9", // Blue
  "#10B981", // Green
  "#EC4899", // Pink
  "#7E69AB", // Secondary Purple
  "#6E59A5", // Tertiary Purple
  "#D6BCFA", // Light Purple
  "#6366F1", // Indigo
  "#F59E0B", // Amber
  "#06B6D4", // Cyan
  "#D946EF", // Pink
];

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white p-3 border rounded shadow">
        <p className="font-medium">{payload[0].name}</p>
        <p className="text-gray-700">{`Value: ${payload[0].value}`}</p>
        <p className="text-gray-500">{`Percentage: ${(payload[0].payload.percentage || 0).toFixed(1)}%`}</p>
      </div>
    );
  }
  return null;
};

const PieChart: React.FC<PieChartProps> = ({ 
  data, 
  title, 
  description,
  colors = defaultColors 
}) => {
  // Calculate percentages for each slice
  const total = data.reduce((sum, item) => sum + item.value, 0);
  const dataWithPercentage = data.map(item => ({
    ...item,
    percentage: (item.value / total) * 100
  }));

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <RechartPieChart>
              <Pie
                data={dataWithPercentage}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={130}
                innerRadius={65}
                paddingAngle={1}
                fill="#8884d8"
                dataKey="value"
                nameKey="name"
                label={({ name, percentage }) => `${name}: ${percentage.toFixed(1)}%`}
              >
                {dataWithPercentage.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend layout="horizontal" verticalAlign="bottom" align="center" />
            </RechartPieChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default PieChart;
