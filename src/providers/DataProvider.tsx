
import React, { createContext, useState, useContext } from 'react';

interface File {
  id?: string;
  fileName: string;
  uploadDate: Date;
  rowCount: number;
  columnCount: number;
  fileSize: string;
  data?: any[];
  pythonAnalysisReady?: boolean; // Added to track Python backend processing status
}

interface Analysis {
  id?: string;
  query: string;
  timestamp: Date;
  fileId: string;
  fileName: string;
  resultType: string;
  pythonBackendUsed?: boolean; // Added to track Python backend usage
}

interface DataContextType {
  currentData: any[] | null;
  currentFileName: string | null;
  files: File[];
  analyses: Analysis[];
  selectedFile: File | null;
  setCurrentData: (data: any[] | null, fileName: string | null) => void;
  addFile: (file: File) => void;
  addAnalysis: (analysis: Analysis) => void;
  setSelectedFile: (file: File | null) => void;
  markFileAsProcessed: (fileId: string) => void; // New function to mark file as processed by Python
}

const DataContext = createContext<DataContextType | undefined>(undefined);

export const useData = () => {
  const context = useContext(DataContext);
  if (!context) {
    throw new Error('useData must be used within a DataProvider');
  }
  return context;
};

interface DataProviderProps {
  children: React.ReactNode;
}

export const DataProvider: React.FC<DataProviderProps> = ({ children }) => {
  const [currentData, setCurrentDataState] = useState<any[] | null>(null);
  const [currentFileName, setCurrentFileName] = useState<string | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [selectedFile, setSelectedFileState] = useState<File | null>(null);
  
  const setCurrentData = (data: any[] | null, fileName: string | null) => {
    setCurrentDataState(data);
    setCurrentFileName(fileName);
  };
  
  const addFile = (file: File) => {
    // Check if file already exists by name
    const existingFileIndex = files.findIndex(f => f.fileName === file.fileName);
    
    if (existingFileIndex >= 0) {
      // Update existing file
      const updatedFiles = [...files];
      updatedFiles[existingFileIndex] = {
        ...file,
        id: updatedFiles[existingFileIndex].id || `file-${Date.now()}`,
        uploadDate: new Date(), // Update upload date
        pythonAnalysisReady: file.pythonAnalysisReady || updatedFiles[existingFileIndex].pythonAnalysisReady
      };
      setFiles(updatedFiles);
    } else {
      // Add new file with unique ID
      setFiles(prevFiles => [...prevFiles, { 
        ...file,
        id: `file-${Date.now()}`,
        pythonAnalysisReady: false // Default to not processed
      }]);
    }
  };
  
  const addAnalysis = (analysis: Analysis) => {
    // Add unique ID to analysis
    setAnalyses(prevAnalyses => [
      { ...analysis, id: `analysis-${Date.now()}` }, 
      ...prevAnalyses
    ]);
  };
  
  const setSelectedFile = (file: File | null) => {
    setSelectedFileState(file);
    if (file && file.data) {
      setCurrentData(file.data, file.fileName);
    }
  };
  
  // New function to mark a file as processed by Python backend
  const markFileAsProcessed = (fileId: string) => {
    setFiles(prevFiles => 
      prevFiles.map(file => 
        file.id === fileId 
          ? { ...file, pythonAnalysisReady: true }
          : file
      )
    );
  };
  
  return (
    <DataContext.Provider
      value={{
        currentData,
        currentFileName,
        files,
        analyses,
        selectedFile,
        setCurrentData,
        addFile,
        addAnalysis,
        setSelectedFile,
        markFileAsProcessed,
      }}
    >
      {children}
    </DataContext.Provider>
  );
};
