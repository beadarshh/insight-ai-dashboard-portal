
export interface User {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
  profileImage?: string;
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  loading: boolean;
}

export interface UserFileHistory {
  id: string;
  fileName: string;
  uploadDate: Date;
  rowCount: number;
  columnCount: number;
  fileSize: string;
}

export interface UserAnalysisHistory {
  id: string;
  query: string;
  timestamp: Date;
  fileId: string;
  fileName: string;
  resultType: string;
}

export interface UserProfile {
  user: User;
  fileHistory: UserFileHistory[];
  analysisHistory: UserAnalysisHistory[];
}
