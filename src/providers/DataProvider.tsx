
import React, { createContext, useContext, useState, useEffect } from 'react';
import { UserFileHistory, UserAnalysisHistory } from '../types/user';
import { useAuth } from './AuthProvider';

interface DataProviderState {
  files: UserFileHistory[];
  analyses: UserAnalysisHistory[];
  currentData: any[] | null;
  currentFileName: string | null;
}

interface DataContextType extends DataProviderState {
  addFile: (file: Omit<UserFileHistory, 'id'>) => void;
  addAnalysis: (analysis: Omit<UserAnalysisHistory, 'id'>) => void;
  setCurrentData: (data: any[] | null, fileName: string | null) => void;
  getFileById: (id: string) => UserFileHistory | undefined;
}

const DataContext = createContext<DataContextType | undefined>(undefined);

export const DataProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, user } = useAuth();
  
  const [state, setState] = useState<DataProviderState>({
    files: [],
    analyses: [],
    currentData: null,
    currentFileName: null
  });

  useEffect(() => {
    if (isAuthenticated && user) {
      // Load user data from local storage
      const savedFiles = localStorage.getItem(`files_${user.id}`);
      const savedAnalyses = localStorage.getItem(`analyses_${user.id}`);
      
      setState(prev => ({
        ...prev,
        files: savedFiles ? JSON.parse(savedFiles) : [],
        analyses: savedAnalyses ? JSON.parse(savedAnalyses) : []
      }));
    } else {
      // Clear data if not authenticated
      setState({
        files: [],
        analyses: [],
        currentData: null,
        currentFileName: null
      });
    }
  }, [isAuthenticated, user]);

  const saveToStorage = (key: string, data: any) => {
    if (isAuthenticated && user) {
      localStorage.setItem(`${key}_${user.id}`, JSON.stringify(data));
    }
  };

  const addFile = (fileData: Omit<UserFileHistory, 'id'>) => {
    const newFile = {
      id: `file_${Date.now()}`,
      ...fileData
    };
    
    setState(prev => {
      const updatedFiles = [...prev.files, newFile];
      saveToStorage('files', updatedFiles);
      return { ...prev, files: updatedFiles };
    });
  };

  const addAnalysis = (analysisData: Omit<UserAnalysisHistory, 'id'>) => {
    const newAnalysis = {
      id: `analysis_${Date.now()}`,
      ...analysisData
    };
    
    setState(prev => {
      const updatedAnalyses = [...prev.analyses, newAnalysis];
      saveToStorage('analyses', updatedAnalyses);
      return { ...prev, analyses: updatedAnalyses };
    });
  };

  const setCurrentData = (data: any[] | null, fileName: string | null) => {
    setState(prev => ({ ...prev, currentData: data, currentFileName: fileName }));
  };

  const getFileById = (id: string) => {
    return state.files.find(file => file.id === id);
  };

  return (
    <DataContext.Provider value={{ 
      ...state, 
      addFile, 
      addAnalysis, 
      setCurrentData,
      getFileById
    }}>
      {children}
    </DataContext.Provider>
  );
};

export const useData = () => {
  const context = useContext(DataContext);
  if (context === undefined) {
    throw new Error('useData must be used within a DataProvider');
  }
  return context;
};
