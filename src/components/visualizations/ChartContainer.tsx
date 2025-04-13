
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';

interface ChartContainerProps {
  title: string;
  description?: string;
  className?: string;
  children: React.ReactNode;
}

const ChartContainer: React.FC<ChartContainerProps> = ({ 
  title, 
  description, 
  className = '',
  children 
}) => {
  return (
    <Card className={`overflow-hidden ${className}`}>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg font-semibold">{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent className="pt-2">
        {/* Chart container with proper padding to prevent label overlap */}
        <div className="pt-4 px-4 pb-8">
          {children}
        </div>
      </CardContent>
    </Card>
  );
};

export default ChartContainer;
