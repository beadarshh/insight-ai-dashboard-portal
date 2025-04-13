
import React, { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Upload, FileSpreadsheet, X, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';
import * as XLSX from 'xlsx';

interface FileUploadProps {
  onDataLoaded: (data: any[], sheets: string[], fileName: string) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onDataLoaded }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  
  const handleDragLeave = () => {
    setIsDragging(false);
  };
  
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      validateAndSetFile(file);
    }
  };
  
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      validateAndSetFile(file);
    }
  };
  
  const validateAndSetFile = (file: File) => {
    setError(null);
    
    // Check if file is an Excel file
    const validExtensions = ['.xlsx', '.xls', '.csv'];
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    
    if (!validExtensions.includes(fileExtension)) {
      setError('Please upload an Excel (.xlsx, .xls) or CSV file');
      return;
    }
    
    // Check file size (limit to 5MB for demo)
    const maxSize = 5 * 1024 * 1024; // 5MB
    if (file.size > maxSize) {
      setError('File size exceeds 5MB limit');
      return;
    }
    
    setSelectedFile(file);
  };
  
  const handleUpload = async () => {
    if (!selectedFile) return;
    
    try {
      setIsUploading(true);
      setUploadProgress(0);
      
      // Simulate upload progress
      const timer = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(timer);
            return 90;
          }
          return prev + 10;
        });
      }, 100);
      
      // Read the Excel file
      const data = await readExcelFile(selectedFile);
      
      // Complete the progress bar
      clearInterval(timer);
      setUploadProgress(100);
      
      // Notify completion
      toast.success(`Successfully processed ${selectedFile.name}`);
      
      // Reset after successful upload
      setTimeout(() => {
        setIsUploading(false);
        setSelectedFile(null);
        setUploadProgress(0);
        if (fileInputRef.current) fileInputRef.current.value = '';
      }, 1000);
      
    } catch (error) {
      console.error("Error processing Excel file:", error);
      setIsUploading(false);
      setError('Error processing file. Please try again with a different file.');
      toast.error('Failed to process file');
    }
  };
  
  const readExcelFile = (file: File): Promise<any> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        try {
          if (!e.target?.result) {
            reject(new Error('Failed to read file'));
            return;
          }
          
          // Handle different file types
          let data: any[] = [];
          let sheetNames: string[] = [];
          
          if (file.name.endsWith('.csv')) {
            // Process CSV
            const text = e.target.result as string;
            const rows = text.split('\n');
            const headers = rows[0].split(',');
            
            data = rows.slice(1).map(row => {
              const values = row.split(',');
              return headers.reduce((obj, header, i) => {
                obj[header.trim()] = values[i]?.trim() || '';
                return obj;
              }, {} as any);
            }).filter(row => Object.values(row).some(val => val));
            
            sheetNames = ['Sheet1'];
          } else {
            // Process Excel
            const workbook = XLSX.read(e.target.result, { type: 'binary' });
            sheetNames = workbook.SheetNames;
            
            // Prevent deep nesting that could cause stack overflow by limiting depth
            const options = { 
              defval: '', 
              raw: false,
              header: 1 
            };
            
            // Get data from the first sheet for now
            const firstSheet = workbook.Sheets[sheetNames[0]];
            
            // Use a simpler approach to parse the sheet to avoid circular references
            try {
              // First get header row
              const headerRow = XLSX.utils.sheet_to_json(firstSheet, { header: 1 })[0] as string[];
              
              // Then get data rows as array of arrays
              const dataRows = XLSX.utils.sheet_to_json(firstSheet, { header: 1 }).slice(1) as any[][];
              
              // Convert array of arrays to array of objects
              data = dataRows.map(row => {
                const obj: Record<string, any> = {};
                headerRow.forEach((header, i) => {
                  if (header) { // Only process if header exists
                    obj[header] = i < row.length ? row[i] : '';
                  }
                });
                return obj;
              });
            } catch (err) {
              console.error("Error parsing Excel sheet:", err);
              reject(err);
              return;
            }
          }
          
          // Make sure we don't have empty rows that could cause problems
          data = data.filter(row => row && Object.keys(row).length > 0);
          
          // Validate data structure to avoid issues down the line
          if (data.length === 0) {
            reject(new Error('No data found in file'));
            return;
          }
          
          // Pass data to parent component
          onDataLoaded(data, sheetNames, file.name);
          resolve(data);
          
        } catch (err) {
          console.error("Error processing file:", err);
          reject(err);
        }
      };
      
      reader.onerror = () => {
        reject(new Error('Failed to read file'));
      };
      
      // Read as binary string
      reader.readAsBinaryString(file);
    });
  };
  
  const cancelUpload = () => {
    setSelectedFile(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="space-y-4">
      {!selectedFile ? (
        <div
          className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors ${
            isDragging ? 'border-insight-500 bg-insight-50' : 'border-gray-300 hover:border-insight-400'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileInputChange}
            className="hidden"
            accept=".xlsx,.xls,.csv"
          />
          
          <div className="flex flex-col items-center justify-center space-y-3">
            <div className="p-4 bg-insight-100 rounded-full">
              <Upload className="h-8 w-8 text-insight-600" />
            </div>
            <h3 className="text-lg font-medium">Upload Excel or CSV File</h3>
            <p className="text-sm text-muted-foreground max-w-md">
              Drag and drop your file here, or click to browse
            </p>
            <p className="text-xs text-muted-foreground">
              Supports .xlsx, .xls and .csv files up to 5MB
            </p>
          </div>
        </div>
      ) : (
        <Card className="p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <FileSpreadsheet className="h-6 w-6 text-blue-700" />
              </div>
              <div>
                <p className="font-medium truncate max-w-xs">{selectedFile.name}</p>
                <p className="text-sm text-muted-foreground">
                  {(selectedFile.size / 1024).toFixed(1)} KB
                </p>
              </div>
            </div>
            
            <Button
              variant="ghost"
              size="icon"
              onClick={cancelUpload}
              disabled={isUploading}
            >
              <X className="h-5 w-5" />
            </Button>
          </div>
          
          {isUploading ? (
            <div className="space-y-2">
              <Progress value={uploadProgress} className="h-2" />
              <p className="text-xs text-muted-foreground text-right">
                Processing... {uploadProgress}%
              </p>
            </div>
          ) : (
            <Button 
              onClick={handleUpload} 
              className="w-full"
            >
              Process File
            </Button>
          )}
        </Card>
      )}
      
      {error && (
        <div className="flex items-center space-x-2 text-red-500 bg-red-50 p-3 rounded-md">
          <AlertCircle className="h-5 w-5" />
          <p className="text-sm">{error}</p>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
