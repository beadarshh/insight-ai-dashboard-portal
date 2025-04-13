
import React, { createContext, useContext, useState, useEffect } from 'react';
import { User, AuthState } from '../types/user';

interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  signup: (name: string, email: string, password: string) => Promise<void>;
  logout: () => void;
  updateUser: (user: Partial<User>) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Mock users for demo purposes
const mockUsers = [
  {
    id: '1',
    name: 'Demo User',
    email: 'demo@example.com',
    password: 'password', // In real app, this would be hashed
    createdAt: new Date(),
    profileImage: undefined
  }
];

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    user: null,
    loading: true
  });

  useEffect(() => {
    // Check for saved session on component mount
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      try {
        const parsedUser = JSON.parse(savedUser);
        setAuthState({
          isAuthenticated: true,
          user: {
            ...parsedUser,
            createdAt: new Date(parsedUser.createdAt)
          },
          loading: false
        });
      } catch (error) {
        console.error('Error parsing saved user:', error);
        localStorage.removeItem('user');
        setAuthState({
          isAuthenticated: false,
          user: null,
          loading: false
        });
      }
    } else {
      setAuthState(prev => ({ ...prev, loading: false }));
    }
  }, []);

  const login = async (email: string, password: string) => {
    // Simulating API call delay
    await new Promise(resolve => setTimeout(resolve, 800));
    
    const user = mockUsers.find(u => u.email === email && u.password === password);
    
    if (!user) {
      throw new Error('Invalid email or password');
    }
    
    const { password: _, ...userWithoutPassword } = user;
    
    setAuthState({
      isAuthenticated: true,
      user: userWithoutPassword,
      loading: false
    });
    
    // Save to local storage
    localStorage.setItem('user', JSON.stringify(userWithoutPassword));
  };

  const signup = async (name: string, email: string, password: string) => {
    // Simulating API call delay
    await new Promise(resolve => setTimeout(resolve, 800));
    
    if (mockUsers.some(u => u.email === email)) {
      throw new Error('Email already in use');
    }
    
    const newUser = {
      id: `${mockUsers.length + 1}`,
      name,
      email,
      password,
      createdAt: new Date(),
      profileImage: undefined
    };
    
    mockUsers.push(newUser);
    
    const { password: _, ...userWithoutPassword } = newUser;
    
    setAuthState({
      isAuthenticated: true,
      user: userWithoutPassword,
      loading: false
    });
    
    // Save to local storage
    localStorage.setItem('user', JSON.stringify(userWithoutPassword));
  };

  const logout = () => {
    setAuthState({
      isAuthenticated: false,
      user: null,
      loading: false
    });
    localStorage.removeItem('user');
  };

  const updateUser = (userData: Partial<User>) => {
    if (!authState.user) return;
    
    const updatedUser = { ...authState.user, ...userData };
    
    setAuthState({
      ...authState,
      user: updatedUser
    });
    
    localStorage.setItem('user', JSON.stringify(updatedUser));
    
    // In a real app, you would also update the user in your backend here
  };

  return (
    <AuthContext.Provider 
      value={{ 
        ...authState, 
        login, 
        signup, 
        logout,
        updateUser
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
