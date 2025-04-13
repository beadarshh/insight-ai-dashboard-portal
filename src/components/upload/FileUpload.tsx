
import React, { useState, useCallback } from 'react';
import { Upload, AlertCircle, CheckCircle2, FileSpreadsheet } from 'lucide-react';
import { cn } from '@/lib/utils';
import * as XLSX from 'xlsx';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

interface FileUploadProps {
  onDataLoaded: (data: any[], sheetNames: string[], fileName: string) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onDataLoaded }) => {
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [file, setFile] = useState<File | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const processExcelFile = useCallback(async (file: File) => {
    try {
      setIsUploading(true);
      
      // Read the file
      const data = await file.arrayBuffer();
      
      // Parse with XLSX
      const workbook = XLSX.read(data);
      const sheetNames = workbook.SheetNames;
      
      if (sheetNames.length === 0) {
        toast.error("No sheets found in the Excel file");
        return;
      }
      
      // Convert first sheet to JSON
      const firstSheetName = sheetNames[0];
      const worksheet = workbook.Sheets[firstSheetName];
      const jsonData = XLSX.utils.sheet_to_json(worksheet);
      
      // Pass data up to parent
      onDataLoaded(jsonData, sheetNames, file.name);
      
      // Success message
      toast.success(`Successfully loaded ${file.name}`);
      setFile(file);
    } catch (error) {
      console.error("Error processing Excel file:", error);
      toast.error("Failed to process Excel file. Please ensure it's a valid .xlsx or .xls file.");
    } finally {
      setIsUploading(false);
      setIsDragging(false);
    }
  }, [onDataLoaded]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type === "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" || 
          file.type === "application/vnd.ms-excel") {
        processExcelFile(file);
      } else {
        toast.error("Please upload an Excel file (.xlsx or .xls)");
      }
    }
    
    setIsDragging(false);
  }, [processExcelFile]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      processExcelFile(file);
    }
  }, [processExcelFile]);

  const handleResetClick = () => {
    setFile(null);
    toast.success("Ready for a new file upload");
  };

  return (
    <div className="w-full">
      {!file ? (
        <div
          className={cn(
            "border-2 border-dashed rounded-lg p-10 transition-colors text-center",
            isDragging ? "border-insight-400 bg-insight-100/50" : "border-gray-300 hover:border-insight-300",
          )}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="bg-insight-100 p-3 rounded-full">
              <Upload size={24} className="text-insight-600" />
            </div>
            <div className="text-center">
              <h3 className="text-lg font-medium text-gray-900 mb-1">Upload your Excel file</h3>
              <p className="text-sm text-gray-500 mb-4">
                Drag and drop your .xlsx or .xls file here, or click to browse
              </p>
              
              <label htmlFor="file-upload" className="cursor-pointer">
                <span className="px-4 py-2 bg-insight-500 text-white rounded-md hover:bg-insight-600 transition-colors">
                  Browse Files
                </span>
                <input 
                  id="file-upload" 
                  name="file-upload" 
                  type="file" 
                  className="sr-only"
                  accept=".xlsx,.xls"
                  onChange={handleFileChange}
                  disabled={isUploading}
                />
              </label>
            </div>
            
            {isUploading && (
              <div className="flex items-center text-insight-400 animate-pulse space-x-2">
                <div className="animate-spin">
                  <Upload size={18} />
                </div>
                <span>Processing file...</span>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="border rounded-lg p-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-green-100 p-2 rounded-full">
                <FileSpreadsheet size={24} className="text-green-600" />
              </div>
              <div>
                <p className="text-sm font-medium text-gray-900">{file.name}</p>
                <p className="text-xs text-gray-500">
                  {(file.size / 1024).toFixed(2)} KB
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleResetClick}
              >
                Upload Another
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
