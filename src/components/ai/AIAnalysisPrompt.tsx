
import React, { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Sparkles, Send, Loader2, BarChart4, LineChart, PieChart, FileSearch, TrendingUp, BrainCog, CodeSquare, Code } from 'lucide-react';

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
      name: "Data Summary",
      description: "Generate a high-level summary of the data using Python and pandas",
      icon: <FileSearch className="h-4 w-4" />,
      query: "Summarize this dataset and provide key insights using Python analysis"
    },
    {
      name: "ML Clustering",
      description: "Apply machine learning clustering to find patterns",
      icon: <BrainCog className="h-4 w-4" />,
      query: "Apply K-means clustering to identify natural groupings in the data"
    },
    {
      name: "Predictive Model",
      description: "Create a predictive model for numeric columns",
      icon: <TrendingUp className="h-4 w-4" />,
      query: "Build a prediction model for the most important numeric column using scikit-learn"
    },
    {
      name: "Anomaly Detection",
      description: "Find outliers and anomalies in the dataset",
      icon: <BarChart4 className="h-4 w-4" />,
      query: "Detect anomalies and outliers in the data using IsolationForest algorithm"
    },
    {
      name: "NLP Analysis",
      description: "Analyze text columns using NLP techniques",
      icon: <CodeSquare className="h-4 w-4" />,
      query: "Apply NLP techniques to extract insights from text columns"
    },
    {
      name: "Gemini Analysis",
      description: "Use Google Gemini AI to analyze patterns in the data",
      icon: <Sparkles className="h-4 w-4" />,
      query: "Analyze this dataset using Google Gemini AI and provide detailed insights"
    }
  ];

  const handlePredefinedQuery = (query: string) => {
    setPrompt(query);
  };

  return (
    <Card className="h-full">
      <CardContent className="p-4 flex flex-col h-full">
        <div className="flex items-center gap-2 text-primary mb-4">
          <Code className="h-5 w-5" />
          <h3 className="font-medium">Python AI Analysis</h3>
        </div>
        
        <form onSubmit={handleSubmit} className="flex flex-col h-full gap-4">
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Ask a specific question about your data for Python and AI-powered analysis..."
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
                title={item.description}
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
                Running Python Analysis...
              </>
            ) : (
              <>
                <Send className="mr-2 h-4 w-4" />
                Analyze with Python
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

export default AIAnalysisPrompt;
