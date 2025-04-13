
import React, { createContext, useContext, useState, useEffect } from 'react';
import { User, AuthState } from '../types/user';
import { toast } from 'sonner';

interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  signup: (name: string, email: string, password: string) => Promise<void>;
  logout: () => void;
  updateUser: (user: Partial<User>) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Initial users for demo purposes
const initialUsers = [
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
  
  // Store users in local state
  const [users, setUsers] = useState<Array<typeof initialUsers[0]>>([]);

  useEffect(() => {
    // Load saved users from localStorage
    const savedUsers = localStorage.getItem('users');
    if (savedUsers) {
      try {
        const parsedUsers = JSON.parse(savedUsers);
        // Convert string dates back to Date objects
        const usersWithDates = parsedUsers.map((user: any) => ({
          ...user,
          createdAt: new Date(user.createdAt)
        }));
        setUsers(usersWithDates);
      } catch (error) {
        console.error('Error parsing saved users:', error);
        // If there's an error, initialize with the default users
        setUsers(initialUsers);
        localStorage.setItem('users', JSON.stringify(initialUsers));
      }
    } else {
      // Initialize with default users if none exist
      setUsers(initialUsers);
      localStorage.setItem('users', JSON.stringify(initialUsers));
    }

    // Check for saved session
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
    
    // Find user from the stored users array
    const user = users.find(u => u.email === email && u.password === password);
    
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
    
    toast.success('Login successful');
  };

  const signup = async (name: string, email: string, password: string) => {
    // Simulating API call delay
    await new Promise(resolve => setTimeout(resolve, 800));
    
    if (users.some(u => u.email === email)) {
      throw new Error('Email already in use');
    }
    
    const newUser = {
      id: `${users.length + 1}`,
      name,
      email,
      password,
      createdAt: new Date(),
      profileImage: undefined
    };
    
    // Update users array with new user
    const updatedUsers = [...users, newUser];
    setUsers(updatedUsers);
    
    // Save updated users to localStorage
    localStorage.setItem('users', JSON.stringify(updatedUsers));
    
    const { password: _, ...userWithoutPassword } = newUser;
    
    setAuthState({
      isAuthenticated: true,
      user: userWithoutPassword,
      loading: false
    });
    
    // Save current user to local storage
    localStorage.setItem('user', JSON.stringify(userWithoutPassword));
    
    toast.success('Account created successfully');
  };

  const logout = () => {
    setAuthState({
      isAuthenticated: false,
      user: null,
      loading: false
    });
    localStorage.removeItem('user');
    toast.info('You have been logged out');
  };

  const updateUser = (userData: Partial<User>) => {
    if (!authState.user) return;
    
    const updatedUser = { ...authState.user, ...userData };
    
    setAuthState({
      ...authState,
      user: updatedUser
    });
    
    localStorage.setItem('user', JSON.stringify(updatedUser));
    
    // Update the user in the users array
    const updatedUsers = users.map(user => {
      if (user.id === authState.user?.id) {
        return { ...user, ...userData, password: user.password };
      }
      return user;
    });
    
    setUsers(updatedUsers);
    localStorage.setItem('users', JSON.stringify(updatedUsers));
    
    toast.success('Profile updated successfully');
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
