
import React, { createContext, useState, useContext } from 'react';

interface File {
  fileName: string;
  uploadDate: Date;
  rowCount: number;
  columnCount: number;
  fileSize: string;
  data?: any[];
}

interface Analysis {
  query: string;
  timestamp: Date;
  fileId: string;
  fileName: string;
  resultType: string;
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
        uploadDate: new Date() // Update upload date
      };
      setFiles(updatedFiles);
    } else {
      // Add new file
      setFiles(prevFiles => [...prevFiles, { ...file }]);
    }
  };
  
  const addAnalysis = (analysis: Analysis) => {
    setAnalyses(prevAnalyses => [analysis, ...prevAnalyses]);
  };
  
  const setSelectedFile = (file: File | null) => {
    setSelectedFileState(file);
    if (file && file.data) {
      setCurrentData(file.data, file.fileName);
    }
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
      }}
    >
      {children}
    </DataContext.Provider>
  );
};
