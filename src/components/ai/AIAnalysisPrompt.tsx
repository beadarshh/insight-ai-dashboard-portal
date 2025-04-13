
import React, { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Sparkles, Send, Loader2, BarChart4, LineChart, PieChart } from 'lucide-react';

interface AIAnalysisPromptProps {
  onAnalyze: (prompt: string) => void;
  isLoading: boolean;
}

const AIAnalysisPrompt: React.FC<AIAnalysisPromptProps> = ({ onAnalyze, isLoading }) => {
  const [prompt, setPrompt] = useState('');
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim() && !isLoading) {
      onAnalyze(prompt.trim());
    }
  };
  
  const predefinedQueries = [
    {
      name: "Summarize Dataset",
      description: "Generate a high-level summary of the data",
      icon: <Sparkles className="h-4 w-4" />,
      query: "Summarize this dataset and provide key insights"
    },
    {
      name: "Distribution Analysis",
      description: "Analyze distribution of categorical columns",
      icon: <PieChart className="h-4 w-4" />,
      query: "Show the distribution of the most important categorical column"
    },
    {
      name: "Trend Analysis",
      description: "Analyze trends in numeric data",
      icon: <LineChart className="h-4 w-4" />,
      query: "Show trends over time for the most important numeric column"
    },
    {
      name: "Comparison",
      description: "Compare data across categories",
      icon: <BarChart4 className="h-4 w-4" />,
      query: "Compare values across different categories"
    }
  ];

  const handlePredefinedQuery = (query: string) => {
    setPrompt(query);
  };

  return (
    <Card className="h-full">
      <CardContent className="p-4 flex flex-col h-full">
        <div className="flex items-center gap-2 text-insight-600 mb-4">
          <Sparkles className="h-5 w-5" />
          <h3 className="font-medium">AI Analysis</h3>
        </div>
        
        <form onSubmit={handleSubmit} className="flex flex-col h-full gap-4">
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Ask a question about your data..."
            className="min-h-[120px] resize-none flex-grow mb-2"
            disabled={isLoading}
          />
          
          <div className="flex flex-wrap gap-2 mb-4">
            {predefinedQueries.map((item, index) => (
              <Button 
                key={index}
                type="button" 
                variant="outline" 
                size="sm"
                className="text-xs flex items-center gap-1"
                onClick={() => handlePredefinedQuery(item.query)}
                disabled={isLoading}
              >
                {item.icon}
                {item.name}
              </Button>
            ))}
          </div>
          
          <Button 
            type="submit" 
            className="w-full"
            disabled={!prompt.trim() || isLoading}
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Send className="mr-2 h-4 w-4" />
                Analyze Data
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

export default AIAnalysisPrompt;
