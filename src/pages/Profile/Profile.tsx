
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { FileIcon, UserRound, History } from 'lucide-react';
import { toast } from 'sonner';
import { useAuth } from '@/providers/AuthProvider';
import { useData } from '@/providers/DataProvider';
import DashboardLayout from '@/components/layout/DashboardLayout';
import { format } from 'date-fns';
import { useNavigate } from 'react-router-dom';

const Profile = () => {
  const { user, updateUser, logout } = useAuth();
  const { files, analyses, setSelectedFile } = useData();
  const navigate = useNavigate();
  
  const [name, setName] = useState(user?.name || '');
  const [email, setEmail] = useState(user?.email || '');
  const [isUpdating, setIsUpdating] = useState(false);
  
  if (!user) {
    return null;
  }
  
  const handleUpdateProfile = (e: React.FormEvent) => {
    e.preventDefault();
    setIsUpdating(true);
    
    try {
      updateUser({ name, email });
      toast.success('Profile updated successfully!');
    } catch (error) {
      toast.error('Failed to update profile');
    } finally {
      setIsUpdating(false);
    }
  };
  
  const handleLogout = () => {
    logout();
    toast.info('Logged out successfully');
  };
  
  const handleViewAnalysis = (file: any) => {
    setSelectedFile(file);
    navigate('/');
    toast.success(`Viewing analysis for ${file.fileName}`);
  };
  
  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold">My Profile</h1>
          <Button variant="outline" onClick={handleLogout}>Logout</Button>
        </div>
        
        <Tabs defaultValue="profile">
          <TabsList>
            <TabsTrigger value="profile" className="flex items-center gap-2">
              <UserRound size={16} />
              Profile
            </TabsTrigger>
            <TabsTrigger value="files" className="flex items-center gap-2">
              <FileIcon size={16} />
              Uploaded Files
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center gap-2">
              <History size={16} />
              Analysis History
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="profile" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle>Account Information</CardTitle>
                <CardDescription>Update your personal details here</CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleUpdateProfile} className="space-y-4">
                  <div className="flex items-center justify-center mb-6">
                    <div className="w-24 h-24 rounded-full bg-insight-100 text-insight-600 flex items-center justify-center text-3xl">
                      {name.charAt(0).toUpperCase()}
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="name">Full Name</Label>
                    <Input 
                      id="name" 
                      value={name} 
                      onChange={(e) => setName(e.target.value)}
                      disabled={isUpdating}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="email">Email</Label>
                    <Input 
                      id="email" 
                      type="email" 
                      value={email} 
                      onChange={(e) => setEmail(e.target.value)}
                      disabled={isUpdating}
                    />
                  </div>
                  
                  <div className="pt-2">
                    <Button 
                      type="submit" 
                      className="w-full" 
                      disabled={isUpdating}
                    >
                      Save Changes
                    </Button>
                  </div>
                </form>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="files" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle>Uploaded Files</CardTitle>
                <CardDescription>All your previously uploaded datasets</CardDescription>
              </CardHeader>
              <CardContent>
                {files.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <FileIcon className="mx-auto h-12 w-12 text-gray-400 mb-2" />
                    <p>You haven't uploaded any files yet.</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {files.map((file) => (
                      <Card key={file.id} className="overflow-hidden">
                        <div className="p-4 flex flex-col md:flex-row items-start md:items-center justify-between">
                          <div className="flex items-center gap-3">
                            <FileIcon className="h-8 w-8 text-insight-500" />
                            <div>
                              <h3 className="font-medium">{file.fileName}</h3>
                              <p className="text-sm text-gray-500">
                                {format(new Date(file.uploadDate), 'PPP')} • {file.rowCount} rows • {file.columnCount} columns
                              </p>
                            </div>
                          </div>
                          <div className="mt-3 md:mt-0">
                            <Button size="sm" variant="outline" onClick={() => handleViewAnalysis(file)}>
                              View Analysis
                            </Button>
                          </div>
                        </div>
                      </Card>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="history" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle>Analysis History</CardTitle>
                <CardDescription>Your previous AI analysis queries</CardDescription>
              </CardHeader>
              <CardContent>
                {analyses.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <History className="mx-auto h-12 w-12 text-gray-400 mb-2" />
                    <p>You haven't performed any analysis yet.</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {analyses.map((analysis) => (
                      <Card key={analysis.id} className="overflow-hidden">
                        <div className="p-4">
                          <p className="font-medium">{analysis.query}</p>
                          <div className="flex justify-between items-center mt-2">
                            <p className="text-sm text-gray-500">
                              {format(new Date(analysis.timestamp), 'PPP')}
                            </p>
                            <span className="inline-flex items-center rounded-full bg-insight-100 px-2.5 py-0.5 text-xs font-medium text-insight-700">
                              {analysis.resultType}
                            </span>
                          </div>
                        </div>
                      </Card>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </DashboardLayout>
  );
};

export default Profile;
