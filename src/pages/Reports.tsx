
import React from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { FileSpreadsheet, Search } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { format } from 'date-fns';
import { useNavigate } from 'react-router-dom';
import { useData } from '@/providers/DataProvider';
import DashboardLayout from '@/components/layout/DashboardLayout';
import { toast } from 'sonner';

const Reports = () => {
  const { files, setSelectedFile, analyses } = useData();
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = React.useState('');
  
  const filteredFiles = files.filter(
    file => file.fileName.toLowerCase().includes(searchTerm.toLowerCase())
  );
  
  const handleFileSelect = (file: any) => {
    setSelectedFile(file);
    navigate('/');
    toast.success(`Viewing analysis for ${file.fileName}`);
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold">Reports & File History</h1>
          <div className="relative w-64">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search files..."
              className="pl-8"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>
        
        <Card>
          <CardHeader>
            <CardTitle>Uploaded Files</CardTitle>
            <CardDescription>
              View and select previously uploaded files for analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            {files.length === 0 ? (
              <div className="text-center py-10">
                <div className="mx-auto w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
                  <FileSpreadsheet className="h-6 w-6 text-muted-foreground" />
                </div>
                <p className="text-muted-foreground">No files have been uploaded yet</p>
                <Button 
                  variant="outline"
                  className="mt-4"
                  onClick={() => navigate('/upload')}
                >
                  Upload First File
                </Button>
              </div>
            ) : (
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>File Name</TableHead>
                      <TableHead>Date</TableHead>
                      <TableHead>Size</TableHead>
                      <TableHead>Rows</TableHead>
                      <TableHead>Columns</TableHead>
                      <TableHead className="text-right">Action</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredFiles.map((file, index) => (
                      <TableRow key={file.id || index}>
                        <TableCell className="font-medium">{file.fileName}</TableCell>
                        <TableCell>
                          {format(new Date(file.uploadDate), 'MMM d, yyyy')}
                        </TableCell>
                        <TableCell>{file.fileSize}</TableCell>
                        <TableCell>{file.rowCount.toLocaleString()}</TableCell>
                        <TableCell>{file.columnCount}</TableCell>
                        <TableCell className="text-right">
                          <Button
                            variant="outline" 
                            size="sm"
                            onClick={() => handleFileSelect(file)}
                          >
                            View Analysis
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>
        
        {analyses.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Recent Analysis Queries</CardTitle>
              <CardDescription>
                View your recent AI analysis queries
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Query</TableHead>
                      <TableHead>File</TableHead>
                      <TableHead>Date</TableHead>
                      <TableHead>Type</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {analyses.map((analysis, index) => (
                      <TableRow key={analysis.id || index}>
                        <TableCell className="font-medium truncate max-w-[300px]">
                          {analysis.query}
                        </TableCell>
                        <TableCell>{analysis.fileName}</TableCell>
                        <TableCell>
                          {format(new Date(analysis.timestamp), 'MMM d, yyyy')}
                        </TableCell>
                        <TableCell className="capitalize">{analysis.resultType}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </DashboardLayout>
  );
};

export default Reports;
