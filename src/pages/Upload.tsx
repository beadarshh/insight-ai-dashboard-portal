
import React from 'react';
import { toast } from 'sonner';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import DashboardLayout from '@/components/layout/DashboardLayout';
import FileUpload from '@/components/upload/FileUpload';
import { useNavigate } from 'react-router-dom';
import { useData } from '@/providers/DataProvider';

const Upload = () => {
  const navigate = useNavigate();
  const { addFile, setCurrentData } = useData();
  
  const handleDataLoaded = (newData: any[], sheets: string[], name: string) => {
    if (!newData || newData.length === 0 || !newData[0]) {
      toast.error("No valid data found in the file");
      return;
    }
    
    try {
      setCurrentData(newData, name);
      
      addFile({
        fileName: name,
        uploadDate: new Date(),
        rowCount: newData.length,
        columnCount: Object.keys(newData[0] || {}).length,
        fileSize: `${Math.round((JSON.stringify(newData).length / 1024) * 10) / 10} KB`
      });
      
      toast.success("File processed successfully!");
      
      // Navigate to the dashboard after successful upload
      navigate('/');
    } catch (error) {
      console.error("Error handling data:", error);
      toast.error("Error processing file data");
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-3xl mx-auto">
        <h1 className="text-2xl font-bold mb-6">Upload Data File</h1>
        
        <Card>
          <CardHeader>
            <CardTitle>Upload File</CardTitle>
            <CardDescription>
              Upload an Excel or CSV file to start analysis. The file will be processed and you'll be redirected to the dashboard.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <FileUpload onDataLoaded={handleDataLoaded} />
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
};

export default Upload;
