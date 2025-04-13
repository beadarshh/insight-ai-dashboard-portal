
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
} from "@/components/ui/navigation-menu";
import { cn } from '@/lib/utils';
import { BarChart3, FileText, Upload, UserRound } from 'lucide-react';

const Navigation = () => {
  const location = useLocation();
  
  return (
    <NavigationMenu>
      <NavigationMenuList>
        <NavigationMenuItem>
          <NavigationMenuTrigger>Dashboard</NavigationMenuTrigger>
          <NavigationMenuContent>
            <div className="grid gap-3 p-4 w-[400px] md:w-[500px]">
              <Link to="/" className={cn(
                "flex items-center gap-3 p-3 rounded-lg hover:bg-accent transition-colors",
                location.pathname === "/" && "bg-accent"
              )}>
                <BarChart3 className="w-5 h-5" />
                <div className="flex flex-col">
                  <span className="font-medium">Analytics Dashboard</span>
                  <span className="text-sm text-muted-foreground">Visualize and explore your data</span>
                </div>
              </Link>
              <Link to="/upload" className={cn(
                "flex items-center gap-3 p-3 rounded-lg hover:bg-accent transition-colors",
                location.pathname === "/upload" && "bg-accent"
              )}>
                <Upload className="w-5 h-5" />
                <div className="flex flex-col">
                  <span className="font-medium">Data Upload</span>
                  <span className="text-sm text-muted-foreground">Upload and manage your datasets</span>
                </div>
              </Link>
              <Link to="/reports" className={cn(
                "flex items-center gap-3 p-3 rounded-lg hover:bg-accent transition-colors",
                location.pathname === "/reports" && "bg-accent"
              )}>
                <FileText className="w-5 h-5" />
                <div className="flex flex-col">
                  <span className="font-medium">Reports</span>
                  <span className="text-sm text-muted-foreground">View saved analysis reports</span>
                </div>
              </Link>
            </div>
          </NavigationMenuContent>
        </NavigationMenuItem>

        <NavigationMenuItem>
          <NavigationMenuTrigger>Profile</NavigationMenuTrigger>
          <NavigationMenuContent>
            <div className="grid gap-3 p-4 w-[300px]">
              <Link to="/profile" className={cn(
                "flex items-center gap-3 p-3 rounded-lg hover:bg-accent transition-colors",
                location.pathname === "/profile" && "bg-accent"
              )}>
                <UserRound className="w-5 h-5" />
                <div className="flex flex-col">
                  <span className="font-medium">My Profile</span>
                  <span className="text-sm text-muted-foreground">View and edit your profile</span>
                </div>
              </Link>
              <Link to="/history" className={cn(
                "flex items-center gap-3 p-3 rounded-lg hover:bg-accent transition-colors",
                location.pathname === "/history" && "bg-accent"
              )}>
                <BarChart3 className="w-5 h-5" />
                <div className="flex flex-col">
                  <span className="font-medium">Analysis History</span>
                  <span className="text-sm text-muted-foreground">View your previous analyses</span>
                </div>
              </Link>
            </div>
          </NavigationMenuContent>
        </NavigationMenuItem>
      </NavigationMenuList>
    </NavigationMenu>
  );
};

export default Navigation;
