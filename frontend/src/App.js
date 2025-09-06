import React, { useState, useEffect, useMemo } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import './App.css';

import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from './components/ui/dialog';
import { Badge } from './components/ui/badge';
import { Progress } from './components/ui/progress';
import { Separator } from './components/ui/separator';
import { ScrollArea } from './components/ui/scroll-area';
import { Switch } from './components/ui/switch';
import { Textarea } from './components/ui/textarea';
import { toast } from 'sonner';
import { 
  Upload, 
  FileSpreadsheet, 
  Users, 
  Settings, 
  Brain, 
  Check, 
  X, 
  Eye, 
  Download,
  Plus,
  Trash2,
  Edit,
  Search,
  Filter,
  RefreshCw,
  ChevronRight,
  AlertCircle,
  CheckCircle2,
  Zap
} from 'lucide-react';

const API = `${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api`;

// Main Dashboard Component
const Dashboard = () => {
  const [clients, setClients] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchClients();
  }, []);

  const fetchClients = async () => {
    try {
      const response = await axios.get(`${API}/clients`);
      setClients(response.data);
    } catch (error) {
      toast.error('Failed to fetch clients');
    }
  };

  const stats = [
    { title: 'Total Clients', value: clients.length, icon: Users, color: 'text-blue-500' },
    { title: 'Statements Processed', value: '0', icon: FileSpreadsheet, color: 'text-green-500' },
    { title: 'Regex Patterns', value: '0', icon: Brain, color: 'text-purple-500' },
    { title: 'Success Rate', value: '0%', icon: CheckCircle2, color: 'text-emerald-500' }
  ];

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-900 to-slate-600 bg-clip-text text-transparent">
            Tally Statement Processor
          </h1>
          <p className="text-slate-600 mt-2">Automate bank statement processing and ledger classification</p>
        </div>
        <Link to="/upload">
          <Button size="lg" className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700">
            <Upload className="w-5 h-5 mr-2" />
            Process Statement
          </Button>
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <Card key={index} className="border-0 shadow-sm bg-white/50 backdrop-blur-sm">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-600">{stat.title}</p>
                  <p className="text-3xl font-bold text-slate-900 mt-2">{stat.value}</p>
                </div>
                <div className={`p-3 rounded-full bg-slate-50 ${stat.color}`}>
                  <stat.icon className="w-6 h-6" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Card className="border-0 shadow-sm">
          <CardHeader>
            <CardTitle>Recent Activities</CardTitle>
            <CardDescription>Latest statement processing activities</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center p-4 rounded-lg bg-slate-50">
                <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
                <div className="flex-1">
                  <p className="text-sm font-medium">No recent activities</p>
                  <p className="text-xs text-slate-500">Upload your first statement to get started</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-sm">
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
            <CardDescription>Common tasks and shortcuts</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Link to="/clients">
              <Button variant="outline" className="w-full justify-start">
                <Users className="w-4 h-4 mr-2" />
                Manage Clients
              </Button>
            </Link>
            <Link to="/patterns">
              <Button variant="outline" className="w-full justify-start">
                <Brain className="w-4 h-4 mr-2" />
                Regex Patterns
              </Button>
            </Link>
            <Link to="/upload">
              <Button variant="outline" className="w-full justify-start">
                <Upload className="w-4 h-4 mr-2" />
                Upload Statement
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

// File Upload Component
const FileUpload = () => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showMappingModal, setShowMappingModal] = useState(false);
  const [clients, setClients] = useState([]);
  const [selectedClient, setSelectedClient] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    fetchClients();
  }, []);

  const fetchClients = async () => {
    try {
      const response = await axios.get(`${API}/clients`);
      setClients(response.data);
    } catch (error) {
      toast.error('Failed to fetch clients');
    }
  };

  const onDrop = async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API}/upload-statement`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setUploadedFile(file);
      setFileData(response.data);
      setShowMappingModal(true);
      toast.success('File uploaded successfully!');
    } catch (error) {
      toast.error('Failed to upload file');
      console.error('Upload error:', error);
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'text/csv': ['.csv']
    },
    multiple: false
  });

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Upload Bank Statement</h1>
        <p className="text-slate-600 mt-2">Upload Excel or CSV files for processing</p>
      </div>

      <Card className="border-0 shadow-sm">
        <CardContent className="p-8">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-12 text-center transition-all duration-200 cursor-pointer
              ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-slate-300 hover:border-slate-400 hover:bg-slate-50'}
              ${loading ? 'pointer-events-none opacity-50' : ''}
            `}
          >
            <input {...getInputProps()} />
            
            <div className="space-y-4">
              <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto">
                {loading ? (
                  <RefreshCw className="w-8 h-8 text-slate-500 animate-spin" />
                ) : (
                  <Upload className="w-8 h-8 text-slate-500" />
                )}
              </div>
              
              <div>
                <p className="text-lg font-medium text-slate-900">
                  {loading ? 'Processing file...' : 'Drop your bank statement here'}
                </p>
                <p className="text-slate-500 mt-1">
                  {loading ? 'Please wait while we analyze your file' : 'Supports Excel (.xlsx, .xls) and CSV files'}
                </p>
              </div>
              
              {!loading && (
                <Button variant="outline" size="sm">
                  <FileSpreadsheet className="w-4 h-4 mr-2" />
                  Choose File
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Column Mapping Modal */}
      <ColumnMappingModal
        isOpen={showMappingModal}
        onClose={() => setShowMappingModal(false)}
        fileData={fileData}
        clients={clients}
        selectedClient={selectedClient}
        setSelectedClient={setSelectedClient}
        onSuccess={() => {
          setShowMappingModal(false);
          navigate('/dashboard');
          toast.success('Statement processed successfully!');
        }}
      />
    </div>
  );
};

// Column Mapping Modal Component
const ColumnMappingModal = ({ isOpen, onClose, fileData, clients, selectedClient, setSelectedClient, onSuccess }) => {
  const MAX_PREVIEW_COLUMNS = 10;
  const [mapping, setMapping] = useState({});
  const [statementFormat, setStatementFormat] = useState('single_amount_crdr');
  const [loading, setLoading] = useState(false);
  
  // useMemo will re-calculate this data only when fileData changes
const previewData = useMemo(() => {
  if (!fileData) {
    return { headers: [], rows: [] };
  }

  // Find the first empty header
  const firstEmptyHeaderIndex = fileData.headers.findIndex(h => !h || h.trim() === '');
  
  // Determine the number of columns to display
  let columnsToDisplay = firstEmptyHeaderIndex === -1 ? fileData.headers.length : firstEmptyHeaderIndex;
  columnsToDisplay = Math.min(columnsToDisplay, MAX_PREVIEW_COLUMNS);

  // Slice the headers and row data based on the calculated number
  const slicedHeaders = fileData.headers.slice(0, columnsToDisplay);
  const slicedRows = fileData.preview_data.map(row => {
    // Create a new row object with only the desired columns
    const newRow = {};
    slicedHeaders.forEach(header => {
      newRow[header] = row[header];
    });
    return newRow;
  });

  return { headers: slicedHeaders, rows: slicedRows.slice(0, 5) }; // Also limit rows for preview
}, [fileData]);

// --- END OF NEW CODE BLOCK TO ADD ---

  useEffect(() => {
    if (fileData?.suggested_mapping) {
      setMapping(fileData.suggested_mapping);
      setStatementFormat(fileData.suggested_mapping.statement_format || 'single_amount_crdr');
    }
  }, [fileData]);

  const handleConfirmMapping = async () => {
    if (!selectedClient) {
      toast.error('Please select a client');
      return;
    }

    setLoading(true);
    try {
      const mappingData = {
        date_column: mapping.date_column,
        narration_column: mapping.narration_column,
        statement_format: statementFormat,
        ...(statementFormat === 'single_amount_crdr' ? {
          amount_column: mapping.amount_column,
          crdr_column: mapping.crdr_column
        } : {
          credit_column: mapping.credit_column,
          debit_column: mapping.debit_column
        }),
        balance_column: mapping.balance_column === 'none' ? null : mapping.balance_column
      };

      await axios.post(
        `${API}/confirm-mapping/${fileData.file_id}?client_id=${selectedClient}`,
        mappingData
      );
      
      onSuccess();
    } catch (error) {
      toast.error('Failed to confirm mapping');
      console.error('Mapping error:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen || !fileData) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="text-2xl font-bold flex items-center">
            <FileSpreadsheet className="w-6 h-6 mr-2 text-blue-500" />
            Confirm Column Mapping
          </DialogTitle>
          <DialogDescription>
            Review and confirm the column mapping for {fileData.filename}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Client Selection */}
          <div className="space-y-2">
            <Label>Select Client</Label>
            <Select value={selectedClient} onValueChange={setSelectedClient}>
              <SelectTrigger>
                <SelectValue placeholder="Choose a client" />
              </SelectTrigger>
              <SelectContent>
                {clients.map(client => (
                  <SelectItem key={client.id} value={client.id}>
                    {client.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Statement Format Selection */}
          <div className="space-y-4">
            <Label>Statement Format:</Label>
            <div className="flex gap-4">
              <div className="flex items-center space-x-2">
                <input
                  type="radio"
                  id="single_amount"
                  name="format"
                  value="single_amount_crdr"
                  checked={statementFormat === 'single_amount_crdr'}
                  onChange={(e) => setStatementFormat(e.target.value)}
                  className="w-4 h-4 text-blue-600"
                />
                <Label htmlFor="single_amount">Single Amount + CR/DR</Label>
              </div>
              <div className="flex items-center space-x-2">
                <input
                  type="radio"
                  id="separate_columns"
                  name="format"
                  value="separate_credit_debit"
                  checked={statementFormat === 'separate_credit_debit'}
                  onChange={(e) => setStatementFormat(e.target.value)}
                  className="w-4 h-4 text-blue-600"
                />
                <Label htmlFor="separate_columns">Separate Credit/Debit</Label>
              </div>
            </div>
          </div>

          {/* --- START OF REPLACEMENT JSX BLOCK --- */}

{/* Use a 10-column grid for a 30/70 split on large screens */}
<div className="grid grid-cols-10 gap-6">
  {/* Column Mapping section spans 3 columns */}
  <div className="col-span-10 lg:col-span-3 space-y-4">
    <h3 className="text-lg font-semibold">Column Mapping</h3>
    
    <div className="space-y-3">
      {/* ... (All your Select dropdowns for mapping go here, UNCHANGED) ... */}
      {/* Date Column */}
      <div>
        <Label>Date Column</Label>
        <Select value={mapping.date_column || ''} onValueChange={(value) => setMapping(prev => ({...prev, date_column: value}))}>
          <SelectTrigger><SelectValue placeholder="Select date column" /></SelectTrigger>
          <SelectContent>
            {fileData.headers.map(header => (<SelectItem key={header} value={header}>{header}</SelectItem>))}
          </SelectContent>
        </Select>
      </div>
      {/* Description/Narration Column */}
      <div>
        <Label>Description/Narration Column</Label>
        <Select value={mapping.narration_column || ''} onValueChange={(value) => setMapping(prev => ({...prev, narration_column: value}))}>
          <SelectTrigger><SelectValue placeholder="Select narration column" /></SelectTrigger>
          <SelectContent>
            {fileData.headers.map(header => (<SelectItem key={header} value={header}>{header}</SelectItem>))}
          </SelectContent>
        </Select>
      </div>
      {/* Conditional Amount/Credit/Debit Columns (UNCHANGED) */}
      {statementFormat === 'single_amount_crdr' ? (
        <>
          <div>
            <Label>Amount Column</Label>
            <Select value={mapping.amount_column || ''} onValueChange={(value) => setMapping(prev => ({...prev, amount_column: value}))}>
              <SelectTrigger><SelectValue placeholder="Select amount column" /></SelectTrigger>
              <SelectContent>
                {fileData.headers.map(header => (<SelectItem key={header} value={header}>{header}</SelectItem>))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label>CR/DR Column</Label>
            <Select value={mapping.crdr_column || ''} onValueChange={(value) => setMapping(prev => ({...prev, crdr_column: value}))}>
              <SelectTrigger><SelectValue placeholder="Select CR/DR column" /></SelectTrigger>
              <SelectContent>
                {fileData.headers.map(header => (<SelectItem key={header} value={header}>{header}</SelectItem>))}
              </SelectContent>
            </Select>
          </div>
        </>
      ) : (
        <>
          <div>
            <Label>Credit Column</Label>
            <Select value={mapping.credit_column || ''} onValueChange={(value) => setMapping(prev => ({...prev, credit_column: value}))}>
              <SelectTrigger><SelectValue placeholder="Select credit column" /></SelectTrigger>
              <SelectContent>
                {fileData.headers.map(header => (<SelectItem key={header} value={header}>{header}</SelectItem>))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label>Debit Column</Label>
            <Select value={mapping.debit_column || ''} onValueChange={(value) => setMapping(prev => ({...prev, debit_column: value}))}>
              <SelectTrigger><SelectValue placeholder="Select debit column" /></SelectTrigger>
              <SelectContent>
                {fileData.headers.map(header => (<SelectItem key={header} value={header}>{header}</SelectItem>))}
              </SelectContent>
            </Select>
          </div>
        </>
      )}
      {/* Balance Column (UNCHANGED) */}
      <div>
        <Label>Balance Column (Optional)</Label>
        <Select value={mapping.balance_column || ''} onValueChange={(value) => setMapping(prev => ({...prev, balance_column: value}))}>
          <SelectTrigger><SelectValue placeholder="Select balance column" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="none">None</SelectItem>
            {fileData.headers.map(header => (<SelectItem key={header} value={header}>{header}</SelectItem>))}
          </SelectContent>
        </Select>
      </div>
    </div>
  </div>

  {/* Data Preview section spans 7 columns */}
  <div className="col-span-10 lg:col-span-7 space-y-4">
    <h3 className="text-lg font-semibold">Data Preview</h3>
    <ScrollArea className="h-96 border rounded-lg">
      {/* Add a div for horizontal scrolling */}
      <div className="overflow-x-auto p-4">
        <table className="w-full text-sm whitespace-nowrap">
          <thead>
            <tr className="border-b">
              {/* Use our new 'previewData' variable */}
              {previewData.headers.map((header, index) => (
                <th key={index} className="text-left p-2 font-medium bg-slate-50 align-top">
  {/* Use a flex container to stack items vertically */}
  <div className="flex flex-col items-start">
    {/* Header Title */}
    <span className="font-semibold">{header}</span>
    
    {/* Conditionally render the badge below the title */}
    {Object.values(mapping).includes(header) && (
      <Badge 
        variant="default" 
        className="mt-1 text-xs bg-blue-500 hover:bg-blue-600 text-white"
      >
        {Object.keys(mapping).find(key => mapping[key] === header)?.replace(/_/g, ' ')}
      </Badge>
    )}
  </div>
</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {/* Use our new 'previewData' variable */}
            {previewData.rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="border-b">
                {previewData.headers.map((header, colIndex) => (
                  <td key={colIndex} className="p-2 text-xs max-w-32 truncate">
                    {row[header]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ScrollArea>
  </div>
</div>

{/* --- END OF REPLACEMENT JSX BLOCK --- */}

          <div className="flex justify-end gap-4 pt-4 border-t">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button 
              onClick={handleConfirmMapping} 
              disabled={loading || !selectedClient}
              className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Check className="w-4 h-4 mr-2" />
                  Confirm Mapping
                </>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

// Client Management Component
const ClientManagement = () => {
  const [clients, setClients] = useState([]);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newClient, setNewClient] = useState({ name: '' });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchClients();
  }, []);

  const fetchClients = async () => {
    try {
      const response = await axios.get(`${API}/clients`);
      setClients(response.data);
    } catch (error) {
      toast.error('Failed to fetch clients');
    }
  };

  const handleCreateClient = async () => {
    if (!newClient.name.trim()) {
      toast.error('Client name is required');
      return;
    }

    setLoading(true);
    try {
      await axios.post(`${API}/clients`, { name: newClient.name });
      toast.success('Client created successfully!');
      setShowCreateModal(false);
      setNewClient({ name: ''});
      fetchClients();
    } catch (error) {
      toast.error('Failed to create client');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Client Management</h1>
          <p className="text-slate-600 mt-2">Manage your clients and their configurations</p>
        </div>
        <Button onClick={() => setShowCreateModal(true)} className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700">
          <Plus className="w-4 h-4 mr-2" />
          Add Client
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {clients.map(client => (
          <Card key={client.id} className="border-0 shadow-sm hover:shadow-md transition-shadow">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{client.name}</CardTitle>
                <Badge variant="outline">Active</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm text-slate-600">
                <p>Created: {new Date(client.created_at).toLocaleDateString()}</p>
                <p>Patterns: 0</p>
                <p>Statements: 0</p>
              </div>
              <div className="flex gap-2 mt-4">
                <Button variant="outline" size="sm">
                  <Edit className="w-4 h-4 mr-1" />
                  Edit
                </Button>
                <Button variant="outline" size="sm">
                  <Eye className="w-4 h-4 mr-1" />
                  View
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Create Client Modal */}
      <Dialog open={showCreateModal} onOpenChange={setShowCreateModal}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New Client</DialogTitle>
            <DialogDescription>
              Add a new client to the system
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label>Client Name</Label>
              <Input
                value={newClient.name}
                onChange={(e) => setNewClient(prev => ({...prev, name: e.target.value}))}
                placeholder="Enter client name"
              />
            </div>
          </div>
          <div className="flex justify-end gap-4 pt-4">
            <Button variant="outline" onClick={() => setShowCreateModal(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateClient} disabled={loading}>
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Creating...
                </>
              ) : (
                'Create Client'
              )}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

// Navigation Component
const Navigation = () => {
  return (
    <nav className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center">
              <FileSpreadsheet className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold text-slate-900">TallyProcessor</span>
          </Link>
          
          <div className="flex items-center space-x-6">
            <Link to="/" className="text-slate-600 hover:text-slate-900 transition-colors">
              Dashboard
            </Link>
            <Link to="/upload" className="text-slate-600 hover:text-slate-900 transition-colors">
              Upload
            </Link>
            <Link to="/clients" className="text-slate-600 hover:text-slate-900 transition-colors">
              Clients
            </Link>
            <Link to="/patterns" className="text-slate-600 hover:text-slate-900 transition-colors">
              Patterns
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

// Main App Component
function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <Router>
        <Navigation />
        <main className="max-w-7xl mx-auto px-6 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<FileUpload />} />
            <Route path="/clients" element={<ClientManagement />} />
            <Route path="/patterns" element={<div>Regex Patterns (Coming Soon)</div>} />
          </Routes>
        </main>
      </Router>
    </div>
  );
}

export default App;