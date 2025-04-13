
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

// Custom render function for pie chart labels that prevents overlapping
const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, index, name, value }: any) => {
  // Position the label outside the pie with enough distance to avoid overlapping
  const RADIAN = Math.PI / 180;
  const radius = outerRadius * 1.2; // Increase this value to move labels further from the pie
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);
  
  // Only show percentage for segments that are large enough to be significant (greater than 3%)
  if (percent < 0.03) return null;
  
  return (
    <text 
      x={x} 
      y={y} 
      fill="#333"
      textAnchor={x > cx ? 'start' : 'end'}
      dominantBaseline="central"
      fontSize={12}
      fontWeight={500}
    >
      {`${name}: ${(percent * 100).toFixed(1)}%`}
    </text>
  );
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
        <div className="h-96 w-full"> {/* Increased height for better spacing */}
          <ResponsiveContainer width="100%" height="100%">
            <RechartPieChart>
              <Pie
                data={dataWithPercentage}
                cx="50%"
                cy="45%" // Move the pie chart a bit higher to leave more space for labels
                labelLine={false}
                outerRadius={120}
                innerRadius={60}
                paddingAngle={1}
                fill="#8884d8"
                dataKey="value"
                nameKey="name"
                label={renderCustomizedLabel}
              >
                {dataWithPercentage.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend 
                layout="horizontal" 
                verticalAlign="bottom" 
                align="center"
                wrapperStyle={{ paddingTop: 20 }} // Add padding to move the legend down
              />
            </RechartPieChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default PieChart;
