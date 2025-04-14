
import React, { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Sparkles, Send, Loader2, BarChart4, LineChart, PieChart, FileSearch, TrendingUp, BrainCog, CodeSquare, Code } from 'lucide-react';

interface AIAnalysisPromptProps {
  onAnalyze: (prompt: string, useGemini?: boolean) => void;
  isLoading: boolean;
}

const AIAnalysisPrompt: React.FC<AIAnalysisPromptProps> = ({ onAnalyze, isLoading }) => {
  const [prompt, setPrompt] = useState('');
  const [useGemini, setUseGemini] = useState(false);
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim() && !isLoading) {
      onAnalyze(prompt.trim(), useGemini);
    }
  };
  
  const predefinedQueries = [
    {
      name: "Data Summary",
      description: "Generate a high-level summary of the data using AI analysis",
      icon: <FileSearch className="h-4 w-4" />,
      query: "Summarize this dataset and provide key insights"
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
      query: "Build a prediction model for the most important numeric column"
    },
    {
      name: "Anomaly Detection",
      description: "Find outliers and anomalies in the dataset",
      icon: <BarChart4 className="h-4 w-4" />,
      query: "Detect anomalies and outliers in the data using machine learning"
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

  const handlePredefinedQuery = (query: string, isGemini: boolean = false) => {
    setPrompt(query);
    setUseGemini(isGemini || query.toLowerCase().includes('gemini'));
  };

  return (
    <Card className="h-full">
      <CardContent className="p-4 flex flex-col h-full">
        <div className="flex items-center gap-2 text-primary mb-4">
          <Sparkles className="h-5 w-5" />
          <h3 className="font-medium">AI Insight</h3>
        </div>
        
        <form onSubmit={handleSubmit} className="flex flex-col h-full gap-4">
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Ask a specific question about your data for AI-powered analysis..."
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
                className={`text-xs flex items-center gap-1 ${item.name === "Gemini Analysis" ? "border-primary/50 bg-primary/5" : ""}`}
                onClick={() => handlePredefinedQuery(item.query, item.name === "Gemini Analysis")}
                disabled={isLoading}
                title={item.description}
              >
                {item.icon}
                {item.name}
              </Button>
            ))}
          </div>
          
          <div className="flex flex-col gap-2">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="useGemini"
                checked={useGemini}
                onChange={(e) => setUseGemini(e.target.checked)}
                className="mr-2"
                disabled={isLoading}
              />
              <label htmlFor="useGemini" className="text-sm text-muted-foreground">
                Use Google Gemini AI for enhanced analysis capabilities
              </label>
            </div>
            
            <Button 
              type="submit" 
              className="w-full"
              disabled={!prompt.trim() || isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Running AI Analysis...
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Generate AI Insight
                </>
              )}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

export default AIAnalysisPrompt;
