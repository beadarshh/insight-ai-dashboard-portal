
import React from 'react';
import Navigation from './Navigation';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="sticky top-0 z-30 w-full bg-white border-b border-gray-200 py-4 px-6 flex justify-between items-center">
        <h2 className="text-lg font-medium">Data Analysis Dashboard</h2>
        <Navigation />
      </header>

      {/* Page Content */}
      <main className="container mx-auto py-6 px-4 md:px-6">
        {children}
      </main>
    </div>
  );
};

export default DashboardLayout;
