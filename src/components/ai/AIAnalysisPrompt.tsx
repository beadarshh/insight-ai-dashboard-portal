
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Send, Sparkles, Loader2 } from 'lucide-react';
import { toast } from 'sonner';

interface AIAnalysisPromptProps {
  onAnalyze: (prompt: string) => Promise<void>;
  isLoading: boolean;
}

const AIAnalysisPrompt: React.FC<AIAnalysisPromptProps> = ({ onAnalyze, isLoading }) => {
  const [prompt, setPrompt] = useState('');
  const [isThinking, setIsThinking] = useState(false);

  const handleAnalyze = async () => {
    if (!prompt.trim()) {
      toast.error("Please enter a prompt for analysis");
      return;
    }

    setIsThinking(true);
    try {
      await onAnalyze(prompt);
      setPrompt(''); // Clear the prompt after successful analysis
    } catch (error) {
      console.error("AI Analysis error:", error);
      toast.error("Failed to generate AI analysis. Please try again.");
    } finally {
      setIsThinking(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleAnalyze();
    }
  };

  const examplePrompts = [
    "Summarize the main trends in this dataset",
    "Compare the top 5 categories by revenue",
    "Show monthly sales trends for the past year",
    "Identify outliers in the dataset",
    "Create a correlation heatmap of numeric variables"
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-insight-400" />
          AI Analysis
        </CardTitle>
        <CardDescription>
          Ask questions about your data to generate insights
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <Textarea
            placeholder="Example: Compare sales performance by region over the last 6 months"
            className="min-h-[100px] resize-none"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading || isThinking}
          />
          
          <div className="flex flex-col gap-3">
            <div className="text-sm font-medium text-gray-500">Example prompts:</div>
            <div className="flex flex-wrap gap-2">
              {examplePrompts.map((examplePrompt, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                  onClick={() => setPrompt(examplePrompt)}
                  disabled={isLoading || isThinking}
                >
                  {examplePrompt}
                </Button>
              ))}
            </div>
          </div>
          
          <div className="flex justify-end">
            <Button
              onClick={handleAnalyze}
              disabled={!prompt.trim() || isLoading || isThinking}
              className="bg-insight-500 hover:bg-insight-600"
            >
              {isThinking || isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Send className="mr-2 h-4 w-4" />
                  Generate Analysis
                </>
              )}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default AIAnalysisPrompt;
