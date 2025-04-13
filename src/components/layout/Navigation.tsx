
import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import {
  BarChart3, 
  FileText, 
  Upload, 
  UserRound, 
  LogOut, 
  Settings,
  ChevronDown
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuth } from '@/providers/AuthProvider';
import { cn } from '@/lib/utils';

const Navigation = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };
  
  return (
    <div className="flex items-center gap-4">
      {/* Dashboard Dropdown */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            <span>Dashboard</span>
            <ChevronDown className="w-4 h-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-[220px] p-2">
          <Link to="/">
            <DropdownMenuItem className={cn(
              "flex items-center gap-3 p-2 cursor-pointer",
              location.pathname === "/" && "bg-accent"
            )}>
              <BarChart3 className="w-4 h-4" />
              <span>Analytics Dashboard</span>
            </DropdownMenuItem>
          </Link>
          
          <Link to="/upload">
            <DropdownMenuItem className={cn(
              "flex items-center gap-3 p-2 cursor-pointer",
              location.pathname === "/upload" && "bg-accent"
            )}>
              <Upload className="w-4 h-4" />
              <span>Data Upload</span>
            </DropdownMenuItem>
          </Link>
          
          <Link to="/reports">
            <DropdownMenuItem className={cn(
              "flex items-center gap-3 p-2 cursor-pointer",
              location.pathname === "/reports" && "bg-accent"
            )}>
              <FileText className="w-4 h-4" />
              <span>Reports</span>
            </DropdownMenuItem>
          </Link>
        </DropdownMenuContent>
      </DropdownMenu>
      
      {/* Profile Dropdown */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" className="flex items-center gap-2 text-insight-600">
            <UserRound className="w-4 h-4" />
            <span>Profile</span>
            <ChevronDown className="w-4 h-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-[220px] p-2">
          <DropdownMenuLabel>
            {user ? user.name : 'Account'}
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          
          <Link to="/profile">
            <DropdownMenuItem className={cn(
              "flex items-center gap-3 p-2 cursor-pointer",
              location.pathname === "/profile" && "bg-accent"
            )}>
              <UserRound className="w-4 h-4" />
              <span>My Profile</span>
            </DropdownMenuItem>
          </Link>
          
          <Link to="/settings">
            <DropdownMenuItem className={cn(
              "flex items-center gap-3 p-2 cursor-pointer",
              location.pathname === "/settings" && "bg-accent"
            )}>
              <Settings className="w-4 h-4" />
              <span>Settings</span>
            </DropdownMenuItem>
          </Link>
          
          <DropdownMenuSeparator />
          
          <DropdownMenuItem 
            className="flex items-center gap-3 p-2 cursor-pointer text-red-500 hover:text-red-600 focus:text-red-600"
            onClick={handleLogout}
          >
            <LogOut className="w-4 h-4" />
            <span>Logout</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
};

export default Navigation;
