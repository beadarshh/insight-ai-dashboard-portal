
import React, { useState } from 'react';
import { ChevronLeft, ChevronRight, BarChart3, Upload, Database, Settings, Bell, LayoutDashboard } from 'lucide-react';
import { cn } from '@/lib/utils';
import Navigation from './Navigation';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      {/* Sidebar */}
      <div
        className={cn(
          "bg-white border-r border-gray-200 transition-all duration-300 flex flex-col",
          collapsed ? "w-16" : "w-64"
        )}
      >
        {/* Logo Area */}
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <h1 className={cn("font-bold text-insight-600 text-xl", collapsed && "hidden")}>
            InsightAI
          </h1>
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-1 rounded-lg hover:bg-gray-100"
          >
            {collapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
          </button>
        </div>

        {/* Nav Links */}
        <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
          <NavItem
            icon={<LayoutDashboard size={20} />}
            label="Dashboard"
            active={true}
            collapsed={collapsed}
            href="/"
          />
          <NavItem
            icon={<Upload size={20} />}
            label="Data Upload"
            active={false}
            collapsed={collapsed}
            href="/upload"
          />
          <NavItem
            icon={<BarChart3 size={20} />}
            label="Visualizations"
            active={false}
            collapsed={collapsed}
            href="/visualizations"
          />
          <NavItem
            icon={<Database size={20} />}
            label="Data Management"
            active={false}
            collapsed={collapsed}
            href="/data"
          />
          <NavItem
            icon={<Settings size={20} />}
            label="Settings"
            active={false}
            collapsed={collapsed}
            href="/settings"
          />
        </nav>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 py-4 px-6 flex justify-between items-center">
          <h2 className="text-lg font-medium">Data Analysis Dashboard</h2>
          <div className="flex items-center gap-4">
            <Navigation />
            <button className="p-2 rounded-full hover:bg-gray-100 relative">
              <Bell size={20} />
              <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
            </button>
            <div className="w-8 h-8 rounded-full bg-insight-400 text-white flex items-center justify-center">
              U
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-y-auto p-6 bg-gray-50">
          {children}
        </main>
      </div>
    </div>
  );
};

interface NavItemProps {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  collapsed: boolean;
  href: string;
}

const NavItem: React.FC<NavItemProps> = ({ icon, label, active, collapsed, href }) => {
  return (
    <a
      href={href}
      className={cn(
        "flex items-center space-x-3 p-3 rounded-lg transition-colors",
        active
          ? "bg-insight-100 text-insight-600"
          : "text-gray-700 hover:bg-gray-100"
      )}
    >
      <span>{icon}</span>
      {!collapsed && <span>{label}</span>}
    </a>
  );
};

export default DashboardLayout;
