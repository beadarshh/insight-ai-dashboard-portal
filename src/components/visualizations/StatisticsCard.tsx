
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';

interface StatisticsCardProps {
  title: string;
  description?: string;
  statistics: {
    label: string;
    value: string | number;
  }[];
}

const StatisticsCard: React.FC<StatisticsCardProps> = ({ title, description, statistics }) => {
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[150px]">Metric</TableHead>
              <TableHead>Value</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {statistics.map((stat, index) => (
              <TableRow key={index}>
                <TableCell className="font-medium">{stat.label}</TableCell>
                <TableCell>
                  {typeof stat.value === 'number' && !Number.isInteger(stat.value)
                    ? stat.value.toFixed(4)
                    : stat.value}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
};

export default StatisticsCard;
