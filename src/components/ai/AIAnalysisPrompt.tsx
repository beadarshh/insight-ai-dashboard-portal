
import React, { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Sparkles, Send, Loader2 } from 'lucide-react';

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

  return (
    <Card>
      <CardContent className="p-4">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex items-center gap-2 text-insight-600 mb-2">
            <Sparkles className="h-5 w-5" />
            <h3 className="font-medium">AI Analysis</h3>
          </div>
          
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Ask a question about your data... 
Examples:
- Summarize this dataset
- Show the distribution of [column]
- Compare [column] by [category]
- Show trends over time for [column]"
            className="min-h-[120px] resize-none"
            disabled={isLoading}
          />
          
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
