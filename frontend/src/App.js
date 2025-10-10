import logo from './assets/logo.png'; // Assuming your logo is in src/assets/
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useNavigate, useParams } from 'react-router-dom';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import './App.css';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from './components/ui/command';
import { Popover, PopoverContent, PopoverTrigger } from './components/ui/popover';

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
  ChevronRight,ChevronsUpDown, 
  AlertCircle,
  CheckCircle2,
  HelpCircle,
  ArrowUp,
  ArrowDown,
  Zap
} from 'lucide-react';


// --- ADD THIS ENTIRE HELPER FUNCTION ---
// In App.js

// --- FIND AND REPLACE THE ENTIRE generateTallyCSV FUNCTION ---
const generateTallyCSV = (vouchers, bankLedgerName, statement) => {
  // Define the exact headers Tally expects
  const headers = [
    "Voucher Date", "Voucher Type Name", "Voucher Number", "Buyer/Supplier - Address",
    "Buyer/Supplier - Pincode", "Ledger Name", "Ledger Amount", "Ledger Amount Dr/Cr",
    "Item Name", "Billed Quantity", "Item Rate", "Item Rate per", "Item Amount",
    "Change Mode", "Voucher Narration"
  ].join(',');

  let voucherNumber = 1;

  const rows = vouchers.map(t => {
    const voucherType = t.voucher_type;
    const isContra = voucherType === 'Contra';
    
    // --- THE FIX IS HERE: We now use the standardized keys directly ---
    const amount = parseFloat(String(t['Amount'] || '0').replace(/,/g, ''));
    const formattedDate = (t['Date'] || '').split(' ')[0];
    const narration = t['Narration'] || '';
    // --- END OF FIX ---

    // --- Line 1: The Party Ledger Entry ---
    const partyLedger = t.matched_ledger || 'Suspense';
    const partyCrDr = isContra ? 'Dr' : (voucherType === 'Receipt' ? 'Cr' : 'Dr');
    const line1 = [
        `"${formattedDate}"`, `"${voucherType}"`, `"${voucherNumber}"`, '""',
        '""', `"${partyLedger}"`, `"${amount.toFixed(2)}"`, `"${partyCrDr}"`,
        '""', '""', '""', '""', '""',
        '""', `"${(narration).replace(/"/g, '""')}"`
    ].join(',');
    
    // --- Line 2: The Bank Ledger Entry ---
    const bankCrDr = isContra ? 'Cr' : (voucherType === 'Receipt' ? 'Dr' : 'Cr');
    const line2 = ['""', '""', '""', '""', '""', `"${bankLedgerName}"`, `"${amount.toFixed(2)}"`, `"${bankCrDr}"`, '""', '""', '""', '""', '""', '""', '""'].join(',');
    
    voucherNumber++;
    return `${line1}\n${line2}`;
  }).join('\n');

  return `${headers}\n${rows}`;
};


// --- ADD/REPLACE THIS ENTIRE COMPONENT ---
const DownloadVoucherModal = ({ isOpen, onClose, data, statementId }) => {
  const [filename, setFilename] = useState('');
  const [includeReceipts, setIncludeReceipts] = useState(true);
  const [includePayments, setIncludePayments] = useState(true);
  const [includeContras, setIncludeContras] = useState(true);
  const [isDownloading, setIsDownloading] = useState(false);

  useEffect(() => {
    if (data?.suggested_filename) {
      setFilename(data.suggested_filename);
    }
  }, [data]);

  const handleDownload = async () => {
    setIsDownloading(true);
    try {
      const payload = {
        include_receipts: includeReceipts,
        include_payments: includePayments,
        include_contras: includeContras,
        filename: filename
      };

      const response = await axios.post(
        `${API}/generate-vouchers/${statementId}`,
        payload,
        { responseType: 'blob' }
      );
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      
      const finalFilename = `${filename}.xlsx`;
      link.setAttribute('download', finalFilename);
      document.body.appendChild(link);
      link.click();
      
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(url);
      onClose();

    } catch (error) {
      toast.error("Failed to generate XLSX file.");
    } finally {
      setIsDownloading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Generate Tally Vouchers</DialogTitle>
          <DialogDescription>Select voucher types and confirm filename for download.</DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label>Include Voucher Types:</Label>
            <div className="flex items-center gap-6 pt-2">
                <div className="flex items-center space-x-2"><Switch id="receipts" checked={includeReceipts} onCheckedChange={setIncludeReceipts} disabled={data.receipt_count === 0} /><Label htmlFor="receipts">Receipts ({data.receipt_count})</Label></div>
                <div className="flex items-center space-x-2"><Switch id="payments" checked={includePayments} onCheckedChange={setIncludePayments} disabled={data.payment_count === 0}/><Label htmlFor="payments">Payments ({data.payment_count})</Label></div>
                <div className="flex items-center space-x-2"><Switch id="contras" checked={includeContras} onCheckedChange={setIncludeContras} disabled={data.contra_count === 0}/><Label htmlFor="contras">Contras ({data.contra_count})</Label></div>
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="filename">Filename</Label>
            <div className="flex items-center">
              <Input id="filename" value={filename} onChange={(e) => setFilename(e.target.value)} />
              <span className="ml-2 text-slate-500">.xlsx</span>
            </div>
          </div>
        </div>
        <div className="flex justify-end gap-4 pt-4 border-t">
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={handleDownload} disabled={isDownloading}>
            {isDownloading ? 'Generating...' : <><Download className="w-4 h-4 mr-2" />Download XLSX</>}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
// --- END OF REPLACEMENT ---

const API = `${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api`;

// In App.js

// --- FIND AND REPLACE THE ENTIRE Dashboard COMPONENT ---
const Dashboard = () => {
  const [statsData, setStatsData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      setLoading(true);
      try {
        const response = await axios.get(`${API}/dashboard-stats`);
        setStatsData(response.data);
      } catch (error) {
        toast.error('Failed to fetch dashboard stats');
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, []);

  const stats = [
    { title: 'Total Clients', value: loading ? '...' : statsData?.total_clients, icon: Users, color: 'text-blue-500' },
    { title: 'Statements Processed', value: loading ? '...' : statsData?.statements_processed, icon: FileSpreadsheet, color: 'text-green-500' },
    { title: 'Regex Patterns', value: loading ? '...' : statsData?.regex_patterns, icon: Brain, color: 'text-purple-500' },
    { title: 'Success Rate', value: loading ? '...' : `${statsData?.success_rate}%`, icon: CheckCircle2, color: 'text-emerald-500' }
  ];

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-900 to-slate-600 bg-clip-text text-transparent">
            Acutant BS. Processor
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
            <Link to="/clients"><Button variant="outline" className="w-full justify-start"><Users className="w-4 h-4 mr-2" />Manage Clients</Button></Link>
            <Link to="/patterns"><Button variant="outline" className="w-full justify-start"><Brain className="w-4 h-4 mr-2" />Regex Patterns</Button></Link>
            <Link to="/upload"><Button variant="outline" className="w-full justify-start"><Upload className="w-4 h-4 mr-2" />Upload Statement</Button></Link>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
// --- END OF REPLACEMENT ---
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
        onSuccess={(newStatementId) => {
          setShowMappingModal(false);
  
          // Clear the old file data to prevent resubmission
          setUploadedFile(null);
          setFileData(null);
          // Navigate to the new statement details page
          navigate(`/statements/${newStatementId}`);
        
          toast.success('Statement processed and saved!');
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
  const [bankAccounts, setBankAccounts] = useState([]);
  const [selectedBankAccount, setSelectedBankAccount] = useState('');

  // useMemo will re-calculate this data only when fileData changes
  const previewData = useMemo(() => {
    if (!fileData || !Array.isArray(fileData.headers) || !Array.isArray(fileData.preview_data)) {
      return { headers: [], rows: [] };
    }

    // Find the first empty header
    const firstEmptyHeaderIndex = fileData.headers.findIndex(h => !h || String(h).trim() === '');
    
    // Determine the number of columns to display
    let columnsToDisplay = firstEmptyHeaderIndex === -1 ? fileData.headers.length : firstEmptyHeaderIndex;
    columnsToDisplay = Math.min(columnsToDisplay, MAX_PREVIEW_COLUMNS);

    // Slice the headers and row data based on the calculated number
    const slicedHeaders = fileData.headers.slice(0, columnsToDisplay);
    const slicedRows = (fileData.preview_data || []).map(row => {
      // Create a new row object with only the desired columns
      const newRow = {};
      slicedHeaders.forEach((header) => {
        newRow[header] = row ? row[header] : undefined;
      });
      return newRow;
    });

    return { headers: slicedHeaders, rows: slicedRows.slice(0, 5) }; // Also limit rows for preview
  }, [fileData]);

  useEffect(() => {
    if (fileData?.suggested_mapping) {
      setMapping(fileData.suggested_mapping);
      setStatementFormat(fileData.suggested_mapping.statement_format || 'single_amount_crdr');
    }
  }, [fileData]);

  useEffect(() => {
    const fetchBankAccounts = async () => {
      if (!selectedClient) {
        setBankAccounts([]);
        setSelectedBankAccount('');
        return;
      }
      try {
        const response = await axios.get(`${API}/clients/${selectedClient}/bank-accounts`);
        setBankAccounts(response.data);
      } catch (error) {
        toast.error("Failed to fetch bank accounts for client.");
        setBankAccounts([]);
      }
    };
    fetchBankAccounts();
  }, [selectedClient]);

  const handleConfirmMapping = async () => {
    if (!selectedClient) {
      toast.error('Please select a client');
      return;
    }

    if (!selectedBankAccount) {
      toast.error('Please select a bank account');
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

      const response = await axios.post(
        `${API}/confirm-mapping/${fileData.file_id}?client_id=${selectedClient}&bank_account_id=${selectedBankAccount}`,
        mappingData
      );
      
      onSuccess(response.data.statement_id);      
      toast.success('Mapping confirmed and statement processed!');

    } catch (error) {
      toast.error('Failed to confirm mapping. Please try again.');
      console.error('Mapping error:', error);
      onClose();
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen || !fileData) return null;
  //Change Upload statement modal size here
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-6xl max-h-[98vh] overflow-y-auto">
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
          {/* --- ADD THIS ENTIRE JSX BLOCK --- */}
          <div className="space-y-2">
            <Label>Select Bank Account</Label>
            <Select 
              value={selectedBankAccount} 
              onValueChange={setSelectedBankAccount}
              disabled={!selectedClient || bankAccounts.length === 0}
            >
              <SelectTrigger>
                <SelectValue placeholder="First, select a client" />
              </SelectTrigger>
              <SelectContent>
                {bankAccounts.map(account => (
                  <SelectItem key={account.id} value={account.id}>
                    {account.ledger_name} ({account.bank_name})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {/* --- END OF ADDITION --- */}

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
// --- FIND AND REPLACE THE ENTIRE ClientManagement COMPONENT ---
const ClientManagement = () => {
  const [clients, setClients] = useState([]);
  const [loading, setLoading] = useState(false);
  
  // State for Create Modal
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newClient, setNewClient] = useState({ name: '' });
  
  // --- START: New state for Edit Modal ---
  const [showEditModal, setShowEditModal] = useState(false);
  const [editingClient, setEditingClient] = useState(null); // Will hold the client object being edited
  // --- END: New state for Edit Modal ---

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
      fetchClients(); // Refresh list
    } catch (error) {
      toast.error('Failed to create client');
    } finally {
      setLoading(false);
    }
  };

  // --- START: New handler functions for Edit ---
  const handleOpenEditModal = (client) => {
    setEditingClient(client); // Set the client to be edited
    setShowEditModal(true);   // Open the modal
  };

  const handleUpdateClient = async () => {
    if (!editingClient || !editingClient.name.trim()) {
      toast.error('Client name is required');
      return;
    }
    setLoading(true);
    try {
      await axios.put(`${API}/clients/${editingClient.id}`, { name: editingClient.name });
      toast.success('Client updated successfully!');
      setShowEditModal(false);
      setEditingClient(null);
      fetchClients(); // Refresh list
    } catch (error) {
      toast.error('Failed to update client');
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
                <p>Patterns: <span className="font-bold">{client.ledger_rule_count}</span></p>
                <p>Statements: <span className="font-bold">{client.bank_statement_count}</span></p>
                <p>Bank Accounts: <span className="font-bold">{client.bank_account_count}</span></p>
              </div>
              <div className="flex gap-2 mt-4">
                <Button variant="outline" size="sm" onClick={() => handleOpenEditModal(client)}>
                  <Edit className="w-4 h-4 mr-1" />
                  Edit
                </Button>
                <Link to={`/clients/${client.id}`}>
                  <Button variant="outline" size="sm">
                  <Eye className="w-4 h-4 mr-1" />
                  View
                </Button>
                </Link>
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
    {/* --- START: Add the new Edit Client Modal --- */}
      <Dialog open={showEditModal} onOpenChange={setShowEditModal}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Client Name</DialogTitle>
            <DialogDescription>
              Update the name for: <span className="font-bold">{editingClient?.name}</span>
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <Label>New Client Name</Label>
              <Input
                value={editingClient?.name || ''}
                onChange={(e) => setEditingClient(prev => ({ ...prev, name: e.target.value }))}
                placeholder="Enter new client name"
              />
            </div>
          </div>
          <div className="flex justify-end gap-4 pt-4">
            <Button variant="outline" onClick={() => setShowEditModal(false)}>
              Cancel
            </Button>
            <Button onClick={handleUpdateClient} disabled={loading}>
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                'Save Changes'
              )}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
      {/* --- END: Add the new Edit Client Modal --- */}

    </div>
  );
};

// Client Details Page Component
// In App.js

// --- REPLACE the entire existing ClientDetailsPage component with this ---
// --- FIND AND REPLACE THE ENTIRE ClientDetailsPage COMPONENT ---
const ClientDetailsPage = () => {
  const { clientId } = useParams();
  const [client, setClient] = useState(null);
  const [bankAccounts, setBankAccounts] = useState([]);
  const [statements, setStatements] = useState([]);
  const [statementToDelete, setStatementToDelete] = useState(null);
  const [showAddAccountModal, setShowAddAccountModal] = useState(false);

  // New state for Tally ledger history
  const [showEditAccountModal, setShowEditAccountModal] = useState(false);
  const [accountToEdit, setAccountToEdit] = useState(null);
  const [accountToDelete, setAccountToDelete] = useState(null);

  const [knownLedgers, setKnownLedgers] = useState([]);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [ledgersData, setLedgersData] = useState({ ledgers: [], total_pages: 1 });
  const [ledgersLoading, setLedgersLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');

  const fetchClientData = useCallback(async () => {
    // This part only fetches non-paginated data once
    setLoading(true);
    try {
      const clientRes = await axios.get(`${API}/clients/${clientId}`);
      setClient(clientRes.data);
      const accountsRes = await axios.get(`${API}/clients/${clientId}/bank-accounts`);
      setBankAccounts(accountsRes.data);
      const statementsRes = await axios.get(`${API}/clients/${clientId}/statements`);
      setStatements(statementsRes.data);
    } catch (error) {
      toast.error("Failed to fetch main client details.");
    } finally {
      setLoading(false);
    }
  }, [clientId]);

  const fetchLedgers = useCallback(async (page, search) => {
    // This function fetches the paginated ledgers
    setLedgersLoading(true);
    try {
      const params = new URLSearchParams({
        page: page,
        limit: 12, // Display 12 items per page
        search: search || '',
      });
      const response = await axios.get(`${API}/clients/${clientId}/ledgers?${params.toString()}`);
      setLedgersData(response.data);
    } catch (error) {
      toast.error("Failed to fetch Tally ledgers.");
    } finally {
      setLedgersLoading(false);
    }
  }, [clientId]);

  useEffect(() => {
    fetchClientData();
  }, [fetchClientData]);

  // --- START OF ADDITION (2 of 3): New useEffect for fetching ledgers ---
  useEffect(() => {
    // Debounce the search query to avoid API calls on every keystroke
    const handler = setTimeout(() => {
      fetchLedgers(currentPage, searchQuery);
    }, 300); // 300ms delay

    return () => {
      clearTimeout(handler);
    };
  }, [currentPage, searchQuery, fetchLedgers]);
  // --- END OF ADDITION (2 of 3) ---

  const handleAccountCreated = (newAccount) => {
    setBankAccounts(prev => [...prev, newAccount]);
    setShowAddAccountModal(false);
  };

  const handleOpenEditModal = (account) => {
    setAccountToEdit(account);
    setShowEditAccountModal(true);
  };

  const handleAccountUpdated = (updatedAccount) => {
    setBankAccounts(prev => prev.map(acc => acc.id === updatedAccount.id ? updatedAccount : acc));
    setShowEditAccountModal(false);
  };

  const handleDeleteAccount = async () => {
    if (!accountToDelete) return;
    try {
      await axios.delete(`${API}/bank-accounts/${accountToDelete.id}`);
      toast.success(`Bank account "${accountToDelete.bank_name}" deleted.`);
      setBankAccounts(prev => prev.filter(acc => acc.id !== accountToDelete.id));
      setAccountToDelete(null); // Close the dialog
    } catch (error) {
      const errorMessage = error.response?.data?.detail || "Failed to delete bank account.";
      toast.error(errorMessage);
    }
  };

  const handleDeleteStatement = async () => {
    if (!statementToDelete) return;
    try {
      await axios.delete(`${API}/statements/${statementToDelete.id}`);
      toast.success(`Statement "${statementToDelete.filename}" deleted.`);
      setStatements(prev => prev.filter(s => s.id !== statementToDelete.id));
      setStatementToDelete(null);
    } catch (error) {
      toast.error("Failed to delete statement.");
    }
  };

  if (loading) return <div>Loading client details...</div>;
  if (!client) return <div>Client not found.</div>;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">{client.name}</h1>
        <p className="text-slate-600 mt-2">Manage client statements, accounts, and ledger history.</p>
      </div>
      
{/* Processed Statements Card */}
      <Card className="border-0 shadow-sm">
        {/* ... (The content of this card remains exactly the same) ... */}
        <CardHeader>
          <CardTitle>Processed Statements</CardTitle>
          <CardDescription>Review or delete processed statements for this client.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {statements.length > 0 ? (
              statements.map(stmt => {
  const isCompleted = stmt.status === "Completed";

  const getPercentageColor = (percentage) => {
    const p = typeof percentage === 'number' ? percentage : 0;
    if (p === 100) return "bg-green-100 text-green-800 border-green-200";
    if (p >= 75) return "bg-teal-100 text-teal-800 border-teal-200";
    if (p >= 40) return "bg-amber-100 text-amber-800 border-amber-200";
    if (p > 0) return "bg-orange-100 text-orange-800 border-orange-200";
    return "bg-red-100 text-red-800 border-red-200";
  };

  return (
    <div key={stmt.id} className="p-4 border rounded-lg bg-slate-50/50 flex flex-col gap-2">
      {/* --- TOP ROW: Filename and Badges --- */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <p className="font-semibold text-slate-800 truncate">{stmt.filename}</p>
          <Badge 
            variant={isCompleted ? "default" : "outline"}
            className={isCompleted 
              ? "bg-green-100 text-green-800 border-green-200" 
              : "border-amber-400 text-amber-700"}
          >
            {stmt.status}
          </Badge>
          <Badge variant="outline" className={getPercentageColor(stmt.completion_percentage)}>
            {stmt.completion_percentage.toFixed(2)}%
          </Badge>
        </div>
      </div>

      {/* --- MIDDLE ROW (METADATA) --- */}
      <div className="text-xs text-slate-500 flex flex-wrap items-center gap-x-3 gap-y-1">
        <span>Uploaded: <span className="font-medium text-slate-600">{new Date(stmt.upload_date).toLocaleDateString()}</span></span>
        {stmt.bank_ledger_name && <span className="text-slate-300">|</span>}
        {stmt.bank_ledger_name && <span>Account: <span className="font-medium text-slate-600">{stmt.bank_ledger_name}</span></span>}
        {stmt.statement_period && <span className="text-slate-300">|</span>}
        {stmt.statement_period && <span>Period: <span className="font-medium text-slate-600">{stmt.statement_period}</span></span>}
        <span className="text-slate-300">|</span>
        <span>Matched: <span className="font-medium text-slate-600">{stmt.matched_transactions} / {stmt.total_transactions}</span></span>
      </div>
      
      {/* --- BOTTOM ROW (PROGRESS BAR & ACTIONS) --- */}
      <div className="flex items-center gap-4 mt-1">
        <Progress value={stmt.completion_percentage} className="h-2" />
        <div className="flex gap-2 flex-shrink-0">
          <Link to={`/statements/${stmt.id}`}>
            <Button variant="outline" size="sm" className="h-8">
              <Eye className="w-4 h-4 mr-1" /> View
            </Button>
          </Link>
          <Button variant="destructive" size="sm" className="h-8" onClick={() => setStatementToDelete(stmt)}>
            <Trash2 className="w-4 h-4 mr-1" /> Delete
          </Button>
        </div>
      </div>
    </div>
  );
})
            ) : (
              <p className="text-center text-slate-500 py-4">No statements have been processed for this client yet.</p>
            )}
          </div>
        </CardContent>
      </Card>

      {/* 3. Tally Ledger History Card */}
      <Card className="border-0 shadow-sm">
        <CardHeader>
          <CardTitle>Tally Ledger History</CardTitle>
          <CardDescription>Upload and manage the curated list of ledgers and sample narrations from Tally.</CardDescription>
        </CardHeader>
        <CardContent>
          {/* --- START OF ADDITION (3 of 3): New Search and Controls --- */}
          <div className="flex justify-between items-center mb-4">
            <div className="relative w-1/3">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
              <Input
                placeholder="Search ledgers..."
                className="pl-9"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setCurrentPage(1); // Reset to first page on new search
                }}
              />
            </div>
            <Button onClick={() => setShowUploadModal(true)}>
              <Upload className="w-4 h-4 mr-2" /> Upload Tally Day Book
            </Button>
          </div>

          {/* New Grid Layout */}
          {ledgersLoading ? (
            <p className="text-center py-8 text-slate-500">Loading ledgers...</p>
          ) : ledgersData.ledgers.length > 0 ? (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {ledgersData.ledgers.map(ledger => (
                  <Link to={`/clients/${clientId}/ledgers/${ledger.id}`} key={ledger.id}>
                    <div className="p-4 border rounded-lg hover:bg-slate-50 hover:shadow-md transition-all h-full">
                      <p className="font-semibold text-slate-800 truncate" title={ledger.ledger_name}>{ledger.ledger_name}</p>
                      <div className="flex items-center justify-between mt-2 text-sm text-slate-600">
                        <span>
                          {ledger.rule_count > 0 ? (
                            <CheckCircle2 className="w-4 h-4 inline mr-1 text-green-500" />
                          ) : (
                            <AlertCircle className="w-4 h-4 inline mr-1 text-amber-500" />
                          )}
                          {ledger.rule_count} rule(s)
                        </span>
                        <span>{ledger.sample_count} sample(s)</span>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>

              {/* Pagination Controls */}
              <div className="flex items-center justify-between mt-6">
                <span className="text-sm text-slate-600">
                  Page {currentPage} of {ledgersData.total_pages}
                </span>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={() => setCurrentPage(p => p - 1)} disabled={currentPage === 1}>Previous</Button>
                  <Button variant="outline" size="sm" onClick={() => setCurrentPage(p => p + 1)} disabled={currentPage === ledgersData.total_pages}>Next</Button>
                </div>
              </div>
            </>
          ) : (
            <p className="text-center text-slate-500 py-8">No ledgers found. Try adjusting your search or upload a Tally Day Book.</p>
          )}
          {/* --- END OF ADDITION (3 of 3) --- */}
        </CardContent>
      </Card>
                 
      {/* 2. Bank Accounts Card */}
      <Card className="border-0 shadow-sm">
        <CardHeader>
          <CardTitle>Bank Accounts</CardTitle>
          <CardDescription>Manage the bank accounts associated with this client.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex justify-end mb-4">
            <Button onClick={() => setShowAddAccountModal(true)}>
              <Plus className="w-4 h-4 mr-2" />
              Add Bank Account
            </Button>
          </div>
          <div className="space-y-4">
            {bankAccounts.length > 0 ? (
              bankAccounts.map(account => (
                <div key={account.id} className="p-4 border rounded-lg flex justify-between items-center">
                  <div>
                    <p className="font-bold">{account.bank_name}</p>
                    <p className="text-sm text-slate-500">{account.ledger_name}</p>
                  </div>
                  {/* --- START OF ADDITION (3 of 3): Edit and Delete Buttons --- */}
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" onClick={() => handleOpenEditModal(account)}>
                      <Edit className="w-4 h-4 mr-1" /> Edit
                    </Button>
                    <Button variant="destructive" size="sm" onClick={() => setAccountToDelete(account)}>
                      <Trash2 className="w-4 h-4 mr-1" /> Delete
                    </Button>
                  </div>
                  {/* --- END OF ADDITION (3 of 3) --- */}
                </div>
              ))
            ) : (
              <p className="text-center text-slate-500 py-4">No bank accounts added yet.</p>
            )}
          </div>
        </CardContent>
      </Card>
      
      <AddBankAccountModal
        isOpen={showAddAccountModal}
        onClose={() => setShowAddAccountModal(false)}
        clientId={clientId}
        onSuccess={handleAccountCreated}
      />

      {/* Tally Upload Modal - ADDED */}
      <TallyUploadModal 
        isOpen={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        clientId={clientId}
        onSuccess={() => {
            setShowUploadModal(false);
            fetchData(); // Re-fetch all data to update the summary list
        }}
      />

      <Dialog open={!!statementToDelete} onOpenChange={(isOpen) => !isOpen && setStatementToDelete(null)}>
        {/* ... (The content of this dialog remains exactly the same) ... */}
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Are you absolutely sure?</DialogTitle>
            <DialogDescription>
              This action cannot be undone. This will permanently delete the statement
              <span className="font-bold"> "{statementToDelete?.filename}"</span>.
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-4 pt-4">
            <Button variant="outline" onClick={() => setStatementToDelete(null)}>Cancel</Button>
            <Button variant="destructive" onClick={handleDeleteStatement}>Confirm Delete</Button>
          </div>
        </DialogContent>
      </Dialog>

      <EditBankAccountModal
        isOpen={showEditAccountModal}
        onClose={() => setShowEditAccountModal(false)}
        account={accountToEdit}
        onSuccess={handleAccountUpdated}
      />

      <Dialog open={!!accountToDelete} onOpenChange={(isOpen) => !isOpen && setAccountToDelete(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Bank Account?</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete the account <span className="font-bold">"{accountToDelete?.bank_name}"</span>? 
              This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-4 pt-4">
            <Button variant="outline" onClick={() => setAccountToDelete(null)}>Cancel</Button>
            <Button variant="destructive" onClick={handleDeleteAccount}>Confirm Delete</Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};
// --- END OF REPLACEMENT ---
// --- ADD THIS ENTIRE NEW MODAL COMPONENT ---
const TallyUploadModal = ({ isOpen, onClose, clientId, onSuccess }) => {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [showHelp, setShowHelp] = useState(false);

  const handleFileChange = (event) => {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      toast.error("Please select a file to upload.");
      return;
    }
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API}/clients/${clientId}/upload-ledger-history`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      toast.success(response.data.message);
      onSuccess();
    } catch (error) {
      const errorMessage = error.response?.data?.detail || "Failed to upload Tally history.";
      toast.error(errorMessage);
    } finally {
      setIsUploading(false);
      setFile(null); // Reset file input
    }
  };
  
  // Clean up when modal closes
  const handleClose = () => {
    setFile(null);
    setShowHelp(false);
    onClose();
  }

  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Upload Tally Day Book</DialogTitle>
          <DialogDescription>
            Upload the exported Excel file to populate the known ledgers and samples.
          </DialogDescription>
        </DialogHeader>
        <div className="py-4 space-y-4">
          <div>
            <Label>Tally Export File (.xlsx)</Label>
            <div className="flex items-center gap-2 mt-2">
              <Input type="file" accept=".xlsx,.xls" onChange={handleFileChange} />
              <Button variant="outline" size="icon" onClick={() => setShowHelp(true)}>
                <HelpCircle className="w-4 h-4" />
              </Button>
            </div>
             {file && <p className="text-sm text-slate-500 mt-2">Selected: {file.name}</p>}
          </div>
        </div>
        <div className="flex justify-end gap-4 pt-4 border-t">
          <Button variant="outline" onClick={handleClose}>Cancel</Button>
          <Button onClick={handleUpload} disabled={isUploading || !file}>
            {isUploading ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Uploading...</> : "Upload File"}
          </Button>
        </div>
        
        {/* Tally Configuration Help Dialog */}
        <Dialog open={showHelp} onOpenChange={setShowHelp}>
          <DialogContent className="max-w-3xl">
            <DialogHeader>
              <DialogTitle>Tally Export Configuration</DialogTitle>
              <DialogDescription>
                To get the correct format, please use these settings when exporting the Day Book report from Tally.
              </DialogDescription>
            </DialogHeader>
            <div className="grid grid-cols-2 gap-6 mt-4 items-start">
              <div className="space-y-3">
                <h3 className="font-semibold">Key Settings:</h3>
                <ul className="list-disc list-inside space-y-1 text-sm">
                  <li><strong>Report Type:</strong> Ledger Accounts</li>
                  <li><strong>Show Narrations:</strong> Yes</li>
                  <li><strong>Format of Report:</strong> Condensed</li>
                  <li><strong>Show Voucher No:</strong> No (Recommended)</li>
                  <li><strong>File Format:</strong> Excel (Spreadsheet)</li>
                </ul>
                <p className="text-xs text-slate-500 pt-4">
                  Ensure all other "Show..." options are generally set to "No" to produce the cleanest output file for parsing.
                </p>
              </div>
              <div className="border rounded-lg overflow-hidden">
                <img src="/tally-config.png" alt="Tally Configuration Screen" className="w-full h-auto" />
              </div>
            </div>
          </DialogContent>
        </Dialog>

      </DialogContent>
    </Dialog>
  );
};
// --- END OF NEW MODAL COMPONENT ---

// --- ADD THIS ENTIRE NEW MODAL COMPONENT ---

const AddBankAccountModal = ({ isOpen, onClose, clientId, onSuccess }) => {
  const [bankName, setBankName] = useState('');
  const [ledgerName, setLedgerName] = useState('');
  const [contraList, setContraList] = useState('');
  const [filterList, setFilterList] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!bankName || !ledgerName) {
      toast.error("Bank Name and Ledger Name are required.");
      return;
    }
    setLoading(true);
    try {
      const payload = {
        client_id: clientId,
        bank_name: bankName,
        ledger_name: ledgerName,
        // Convert comma-separated strings to arrays of strings, filtering out empty entries
        contra_list: contraList.split(',').map(s => s.trim()).filter(Boolean),
        filter_list: filterList.split(',').map(s => s.trim()).filter(Boolean),
      };
      
      const response = await axios.post(`${API}/bank-accounts`, payload);
      toast.success("Bank account created successfully!");
      onSuccess(response.data); // Pass the new account data back to the parent
    } catch (error) {
      toast.error("Failed to create bank account.");
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add New Bank Account</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label>Bank Name</Label>
            <Input value={bankName} onChange={(e) => setBankName(e.target.value)} placeholder="e.g., IDBI Bank" />
          </div>
          <div>
            <Label>Ledger Name</Label>
            <Input value={ledgerName} onChange={(e) => setLedgerName(e.target.value)} placeholder="e.g., IDBI Bank_12467" />
          </div>
          <div>
            <Label>Contra Ledgers (comma-separated)</Label>
            <Textarea value={contraList} onChange={(e) => setContraList(e.target.value)} placeholder="e.g., ICICI Bank_..., Cash" />
          </div>
          <div>
            <Label>Filter List (comma-separated)</Label>
            <Textarea value={filterList} onChange={(e) => setFilterList(e.target.value)} placeholder="e.g., IDBI 4003" />
          </div>
        </div>
        <div className="flex justify-end gap-4 pt-4">
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={handleSubmit} disabled={loading}>
            {loading ? 'Saving...' : 'Save Account'}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

// --- START OF ADDITION (1 of 3) ---
const EditBankAccountModal = ({ isOpen, onClose, account, onSuccess }) => {
  const [bankName, setBankName] = useState('');
  const [ledgerName, setLedgerName] = useState('');
  const [contraList, setContraList] = useState('');
  const [filterList, setFilterList] = useState('');
  const [loading, setLoading] = useState(false);

  // Pre-fill the form when the modal opens or the account prop changes
  useEffect(() => {
    if (account) {
      setBankName(account.bank_name || '');
      setLedgerName(account.ledger_name || '');
      // Convert arrays back to comma-separated strings for the textarea
      setContraList((account.contra_list || []).join(', '));
      setFilterList((account.filter_list || []).join(', '));
    }
  }, [account]);

  const handleSubmit = async () => {
    if (!bankName || !ledgerName) {
      toast.error("Bank Name and Ledger Name are required.");
      return;
    }
    setLoading(true);
    try {
      const payload = {
        client_id: account.client_id, // Use the client_id from the existing account
        bank_name: bankName,
        ledger_name: ledgerName,
        contra_list: contraList.split(',').map(s => s.trim()).filter(Boolean),
        filter_list: filterList.split(',').map(s => s.trim()).filter(Boolean),
      };
      
      const response = await axios.put(`${API}/bank-accounts/${account.id}`, payload);
      toast.success("Bank account updated successfully!");
      onSuccess(response.data); // Pass the updated account data back to the parent
    } catch (error) {
      toast.error("Failed to update bank account.");
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit Bank Account</DialogTitle>
          <DialogDescription>
            Update the details for: <span className="font-bold">{account?.bank_name}</span>
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div>
            <Label>Bank Name</Label>
            <Input value={bankName} onChange={(e) => setBankName(e.target.value)} placeholder="e.g., IDBI Bank" />
          </div>
          <div>
            <Label>Ledger Name</Label>
            <Input value={ledgerName} onChange={(e) => setLedgerName(e.target.value)} placeholder="e.g., IDBI Bank_12467" />
          </div>
          <div>
            <Label>Contra Ledgers (comma-separated)</Label>
            <Textarea value={contraList} onChange={(e) => setContraList(e.target.value)} placeholder="e.g., ICICI Bank_..., Cash" />
          </div>
          <div>
            <Label>Filter List (comma-separated)</Label>
            <Textarea value={filterList} onChange={(e) => setFilterList(e.target.value)} placeholder="e.g., IDBI 4003" />
          </div>
        </div>
        <div className="flex justify-end gap-4 pt-4">
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={handleSubmit} disabled={loading}>
            {loading ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
// --- END OF ADDITION (1 of 3) ---
// --- ADD THIS ENTIRE CONSTANT ---
const REGEX_BUILDING_BLOCKS = [
  { label: "Starts With", snippet: "^", description: "start of the line" },
  { label: "Ends With", snippet: "$", description: "end of the line" },
  { label: "Word Boundary", snippet: "\\b", description: "a word boundary (whole word)" },
  { label: "Any Digit", snippet: "\\d", description: "any single digit (0-9)" },
  { label: "1+ Digits", snippet: "\\d+", description: "one or more digits" },
  { label: "Any Letter", snippet: "[A-Za-z]", description: "any single letter" },
  { label: "1+ Letters", snippet: "[A-Za-z]+", description: "one or more letters" },
  { label: "Anything", snippet: ".*", description: "any character, zero or more times" },
];
// --- END OF ADDITION ---
// --- FIND AND REPLACE THE ENTIRE LedgerCombobox COMPONENT ---
const LedgerCombobox = ({ ledgers, value, onValueChange }) => {
  const [open, setOpen] = React.useState(false);

  const triggerRef = useRef(null);
  const [popoverWidth, setPopoverWidth] = useState(0);

  useEffect(() => {
    if (triggerRef.current) {
      setPopoverWidth(triggerRef.current.offsetWidth);
    }
  }, []);

  // Check if the current typed value is an exact match to an existing ledger
  const isExactMatch = ledgers.some(ledger => ledger.toLowerCase() === value.toLowerCase());

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          ref={triggerRef}
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-full justify-between font-normal"
        >
          <span className="truncate">{value || "Select or type a ledger..."}</span>
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="p-0"
        style={{ width: popoverWidth }}
      >
        <Command
          // The filter remains to provide the search functionality
          filter={(itemValue, search) => itemValue.toLowerCase().includes(search.toLowerCase()) ? 1 : 0}
        >
          <CommandInput
            placeholder="Search or create ledger..."
            value={value}
            onValueChange={onValueChange}
          />
          <CommandList>
            {/* CommandEmpty is now simple and only shows when there are TRULY no options */}
            <CommandEmpty>No results found.</CommandEmpty>

            <CommandGroup>
              {/* --- THIS IS THE KEY FIX --- */}
              {/* Conditionally render the "Create" option as a regular item */}
              {/* It appears only if the user has typed something AND it's not an exact match */}
              {value && !isExactMatch && (
                <CommandItem
                  key={value}
                  value={value}
                  onSelect={(currentValue) => {
                    onValueChange(currentValue);
                    setOpen(false);
                  }}
                >
                  <Plus className="mr-2 h-4 w-4" />
                  Create "{value}"
                </CommandItem>
              )}

              {/* Then, render all the matching ledgers from the list */}
              {ledgers.map((ledger) => (
                <CommandItem
                  key={ledger}
                  value={ledger}
                  onSelect={(currentValue) => {
                    onValueChange(currentValue);
                    setOpen(false);
                  }}
                >
                  <Check className={`mr-2 h-4 w-4 ${value.toLowerCase() === ledger.toLowerCase() ? "opacity-100" : "opacity-0"}`} />
                  {ledger}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
};
// --- END OF REPLACEMENT ---
// --- REPLACE THE ENTIRE ClusterCard COMPONENT ---
const ClusterCard = ({ cluster, clientId, onRuleCreated, otherNarrations, onDetach, onMarkAsSuspense, knownLedgers }) => {
  const [editableRegex, setEditableRegex] = useState(cluster.suggested_regex || '');
  const [ledgerName, setLedgerName] = useState('');
  const [loading, setLoading] = useState(false);
  const [validation, setValidation] = useState({ 
    matchStatus: 'none',
    matchCount: 0,
    highlightedNarrations: []
  });
  const [falsePositiveCount, setFalsePositiveCount] = useState(0);
  
  // New state for the regex helpers
  const [showHelpers, setShowHelpers] = useState(false);
  const regexInputRef = useRef(null);

  const handleInsertSnippet = (snippet) => {
    const input = regexInputRef.current;
    if (!input) return;

    const start = input.selectionStart;
    const end = input.selectionEnd;
    const text = input.value;
    
    const newText = text.substring(0, start) + snippet + text.substring(end);
    setEditableRegex(newText);

    // Focus the input and move the cursor to the end of the inserted snippet
    input.focus();
    setTimeout(() => {
      input.setSelectionRange(start + snippet.length, start + snippet.length);
    }, 0);
  };

  // In App.js, inside the ClusterCard component

// --- FIND AND REPLACE THE ENTIRE decodeRegex FUNCTION ---
  const decodeRegex = (regex) => {
    if (!regex || !regex.trim()) return [];

    const keywordMatches = [...regex.matchAll(/\\b([A-Za-z0-9_ -]+)\\b/g)];

    if (keywordMatches.length === 0) {
      return [[{ type: 'error', text: "Could not find any whole words to explain in this pattern." }]];
    }

    const explanation = [];

    // --- START: MODIFIED LOGIC ---
    // Process the first keyword
    explanation.push([
      { type: 'intro', text: 'The text must contain ' },
      { type: 'keyword_punctuation', text: 'the word "' },
      { type: 'keyword_text', text: keywordMatches[0][1] },
      { type: 'keyword_punctuation', text: '"' }
    ]);

    // Process all subsequent keywords
    for (let i = 1; i < keywordMatches.length; i++) {
      explanation.push([
        { type: 'connector', text: '   followed (anywhere later) by ' },
        { type: 'keyword_punctuation', text: 'the word "' },
        { type: 'keyword_text', text: keywordMatches[i][1] },
        { type: 'keyword_punctuation', text: '"' }
      ]);
    }
    // --- END: MODIFIED LOGIC ---
    return explanation;
  };

  const getTokenColor = (type) => {
    switch (type) {
      case 'wildcard': return 'text-blue-400';
      case 'boundary':
      case 'anchor':
      case 'escape': return 'text-green-400';
      case 'literal': return 'text-orange-400';
      case 'grouping': return 'text-purple-400';
      case 'quantifier': return 'text-red-400';
      default: return 'text-slate-400';
    }
  };

  useEffect(() => {
    const testRegex = (regexStr, text) => {
      if (!text) return false;
      try { return new RegExp(regexStr, 'i').test(text); } catch (e) { return false; }
    };
    
    const getHighlightedHtml = (regexStr, text) => {
      if (!text) return { html: '', matched: false };
      try {
        const re = new RegExp(regexStr, 'i');
        const match = text.match(re);
        if (!match || !match[0] || !regexStr) return { html: text.replace(/</g, '&lt;'), matched: false };
        const escapedText = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        const highlightedHtml = escapedText.replace(match[0], `<span class="bg-green-200 font-bold px-1 rounded">${match[0]}</span>`);
        return { html: highlightedHtml, matched: true };
      } catch (e) {
        return { html: text.replace(/</g, '&lt;'), matched: false };
      }
    };

    let fpCount = 0;
    if (editableRegex && otherNarrations) {
      for (const other of otherNarrations) {
        if (testRegex(editableRegex, other)) { fpCount++; }
      }
    }
    setFalsePositiveCount(fpCount);

    const newHighlightedResults = cluster.transactions.map(t => getHighlightedHtml(editableRegex, t.Narration));
    const currentMatchCount = newHighlightedResults.filter(result => result.matched).length;

    let status = 'none';
    if (currentMatchCount > 0 && currentMatchCount === cluster.transactions.length) status = 'all';
    else if (currentMatchCount > 0) status = 'partial';

    setValidation({
      matchStatus: status,
      matchCount: currentMatchCount,
      highlightedNarrations: newHighlightedResults.map(result => result.html)
    });
  }, [editableRegex, cluster.transactions, otherNarrations]);

  const handleCreateRule = async () => {
    if (!ledgerName.trim()) { toast.error("Ledger name is required."); return; }
    if (validation.matchStatus === 'none') { toast.error("Cannot create a rule with a pattern that doesn't match."); return; }
    if (falsePositiveCount > 0) { toast.error("Cannot create a rule with a pattern that has false positives."); return; }
    try { new RegExp(editableRegex); } catch (e) { toast.error("Invalid Regex syntax."); return; }

    setLoading(true);
    try {
      const payload = {
        client_id: clientId,
        ledger_name: ledgerName,
        regex_pattern: editableRegex,
        sample_narrations: cluster.transactions.map(t => t.Narration),
      };
      await axios.post(`${API}/ledger-rules`, payload);
      toast.success(`Rule for "${ledgerName}" created!`);
      onRuleCreated();
    } catch (error) { toast.error("Failed to create rule."); } 
    finally { setLoading(false); }
  };

  const formatCurrency = (amount) => {
    if (amount === null || typeof amount === 'undefined') return '';
    const cleanedAmount = String(amount).replace(/,/g, '');
    const num = Number(cleanedAmount);
    return isNaN(num) ? '' : `₹${num.toLocaleString('en-IN')}`;
  };

  return (
    <div className="p-4 border rounded-lg bg-slate-50 space-y-3">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-4">
          <p className="text-sm font-semibold">Define Regex Pattern</p>
          <Button size="sm" variant="outline" onClick={() => onMarkAsSuspense(cluster.transactions)}>
            <HelpCircle className="w-4 h-4 mr-2" /> Mark All as Suspense
          </Button>
        </div>
        <div className="flex items-center gap-2">
          {(() => {
            const percentage = Math.round((validation.matchCount / cluster.transactions.length) * 100);
            switch (validation.matchStatus) {
              case 'all': return <span className="text-sm font-bold text-green-600 flex items-center"><CheckCircle2 className="w-4 h-4 mr-1"/> All Matched</span>;
              case 'partial': return <span className="text-sm font-bold text-yellow-600">{`Partial Match [${validation.matchCount}/${cluster.transactions.length}]`}</span>;
              default: return <span className="text-sm font-bold text-red-600 flex items-center"><AlertCircle className="w-4 h-4 mr-1"/> No Match</span>;
            }
          })()}
        </div>
      </div>
      
      <div>
        <div className="flex items-center gap-2">
          <Input
            ref={regexInputRef}
            className="font-mono text-xs bg-white flex-grow"
            value={editableRegex}
            onChange={(e) => setEditableRegex(e.target.value)}
            placeholder="Enter Regex Pattern..."
          />
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="outline" size="icon" className="h-9 w-9 flex-shrink-0" title="Decode Regex">
                <Brain className="h-4 w-4" />
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader><DialogTitle>Regex Decoder</DialogTitle></DialogHeader>
              
              {/* --- START: FINAL DECODER DISPLAY --- */}
              <div className="mt-2 p-4 bg-slate-800 text-slate-300 rounded-md font-mono text-sm">
                <p className="font-bold text-white">Your Pattern:</p>
                <p className="break-words text-slate-400 mb-3">{editableRegex}</p>
                <p className="font-bold text-white">Explanation:</p>
                <div className="mt-2">
                  {decodeRegex(editableRegex).map((line, lineIndex) => (
                    <div key={lineIndex}>
                      {line.map((part, partIndex) => (
                        <span key={partIndex} className={
                          part.type === 'keyword_text' ? 'text-green-400 font-bold' :
                          part.type === 'keyword_punctuation' ? 'text-pink-400 font-semibold' :
                          part.type === 'connector' ? 'text-amber-400' :
                          'text-slate-300'
                        }>
                          {part.text}
                        </span>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
              {/* --- END: FINAL DECODER DISPLAY --- */}
            </DialogContent>
          </Dialog>
          <Button variant="outline" size="icon" className="h-9 w-9 flex-shrink-0" title="Regex Helpers" onClick={() => setShowHelpers(!showHelpers)}>
            <Zap className="h-4 w-4" />
          </Button>
        </div>
        
         {/* This is the smoothly animating ribbon */}
        <div 
          className={`
            transition-all duration-300 ease-in-out overflow-hidden
            ${showHelpers 
              ? 'max-h-40 mt-2 p-2 border rounded-md bg-slate-100 flex flex-wrap gap-2' 
              : 'max-h-0 mt-0 p-0 border-none invisible'
            }
          `}
        >
          {REGEX_BUILDING_BLOCKS.map(block => (
            <Button key={block.label} size="sm" variant="outline" className="bg-white" onClick={() => handleInsertSnippet(block.snippet)} title={block.description}>
              {block.label}
            </Button>
          ))}
        </div>
      </div>

      {falsePositiveCount > 0 && (
        <div className="p-3 my-2 bg-yellow-100 border-l-4 border-yellow-400 text-yellow-800 text-sm rounded-r-md flex items-center gap-3">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <div><span className="font-bold">Warning:</span> This pattern incorrectly matches <span className="font-extrabold">{falsePositiveCount}</span> other transaction(s).</div>
        </div>
      )}

      <p className="text-sm font-semibold">Sample Transactions ({cluster.transactions.length} items):</p>
      
      {/* THE FIX: Replaced ScrollArea with a div and added overflow-y-auto */}
      <div className="p-2 border rounded-md bg-white max-h-96 overflow-y-auto">
        <div className="text-xs ">
          {cluster.transactions.map((transaction, i) => {
            const isCredit = (transaction['CR/DR'] || '').startsWith('CR');
            return (
              <div key={i} className="flex items-center gap-2 p-1 group text-[13px] leading-snug">
                {/* Action Buttons (unchanged) */}
                <div className="flex-shrink-0 flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
                   <Button variant="ghost" size="icon" className="w-6 h-6" onClick={() => onMarkAsSuspense([transaction])} title="Mark as Suspense">
                      <HelpCircle className="w-3 h-3 text-slate-500" />
                   </Button>
                   <Button variant="ghost" size="icon" className="w-6 h-6" onClick={() => onDetach(transaction, cluster.cluster_id)} title="Detach from cluster">
                      <X className="w-3 h-3 text-slate-500" />
                   </Button>
                </div>

                {/* NEW: Fixed-width container for the date badge */}
                <div className="w-24 flex-shrink-0">
                  <Badge variant="outline" className="font-mono">
                    {(transaction.Date || '').split(' ')[0]}
                  </Badge>
                </div>

                {/* Narration (now correctly aligned) */}
                <div className="truncate flex-grow" dangerouslySetInnerHTML={{ __html: validation.highlightedNarrations[i] }} />

                {/* NEW: Container for right-side badges */}
                <div className="flex items-center gap-2 flex-shrink-0">
                  <Badge variant="outline" className="font-mono">{formatCurrency(transaction.Amount)}</Badge>
                  <Badge 
                    className={`h-5 font-semibold border-0 bg-slate-800 ${isCredit 
                      ? 'text-green-400' 
                      : 'text-red-400'}`}
                  >
                    {isCredit ? 'Credit' : 'Debit'}
                  </Badge>
                </div>
              </div>
            );
          })}
        </div>
      </div>
         
      <div className="flex items-center gap-4 pt-2">
      {/* --- THIS IS THE REPLACEMENT --- */}
      <LedgerCombobox 
        ledgers={knownLedgers || []}
        value={ledgerName}
        onValueChange={setLedgerName}
      />
        <Button onClick={handleCreateRule} disabled={loading || validation.matchStatus === 'none' || falsePositiveCount > 0 || !ledgerName.trim()}>
          <Plus className="w-4 h-4 mr-2" />{loading ? 'Creating...' : 'Create Rule'}
        </Button>
      </div>
    </div>
  );
};
// --- END OF REPLACEMENT ---


// --- ADD THIS ENTIRE NEW COMPONENT ---

// --- FIND AND REPLACE THE ENTIRE ClassifiedTransactionsTable COMPONENT ---
const ClassifiedTransactionsTable = ({
  transactions,
  onFlagAsIncorrect,
  selectedTxIds,
  onToggleRow,
  // NOTE: onToggleAll is no longer passed here
}) => {
  const transactionsToShow = transactions.filter(
    t => t.matched_ledger !== "Suspense" || t.user_confirmed === true
  );

  // NOTE: isAllSelected logic is moved to the parent component

  if (transactionsToShow.length === 0) {
    return <p className="text-center text-slate-500 py-4">No transactions were matched to existing rules.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm table-fixed">
        {/* THE REAL THEAD IS NOW INVISIBLE. It only serves to set column widths. */}
        <thead>
          <tr className="border-b">
            <th className="py-0 px-2 w-24"></th>
            <th className="py-0 px-2 w-12"></th>
            <th className="py-0 px-2"></th>
            <th className="py-0 px-2 w-32"></th>
            <th className="py-0 px-2 w-24"></th>
            <th className="py-0 px-2 w-48"></th>
            <th className="py-0 px-2 w-24"></th>
          </tr>
        </thead>
        <tbody>
          {transactionsToShow.map(transaction => (
            <tr
              key={transaction._tempId || (transaction.Narration + transaction.Amount + Math.random())}
              className={`border-b hover:bg-slate-50 ${selectedTxIds.has(transaction._tempId) ? 'bg-blue-50' : ''}`}
            >
              <td className="p-2 whitespace-nowrap">{transaction.Date}</td>
              <td className="p-2 text-center">
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={selectedTxIds.has(transaction._tempId)}
                  onChange={() => onToggleRow(transaction._tempId)}
                />
              </td>
              <td className="p-2 max-w-sm hyphenate">{transaction.Narration}</td>
              <td className="p-2 text-right font-mono">{transaction.Amount?.toLocaleString()}</td>
              <td className="p-2">{transaction['CR/DR']}</td>
              <td className="p-2 truncate">{transaction.matched_ledger}</td>
              <td className="p-2 text-center">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onFlagAsIncorrect(transaction)}
                  title="Flag as incorrect match"
                >
                  <X className="w-4 h-4 text-red-500" />
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
// --- END OF REPLACEMENT ---
// --- END OF REPLACEMENT ---

// --- FIND AND REPLACE THE ENTIRE generateSimpleRegex FUNCTION ---
const generateSimpleRegex = (narration) => {
  if (!narration) return '.*';

  // A more robust list of common banking/transaction noise words
  const noise = new Set([
    'NEFT', 'IMPS', 'UPI', 'RTGS', 'BY', 'TO', 'FOR', 'THE', 'AND', 'PAYMENT',
    'TRANSFER', 'FUND', 'DEBIT', 'CREDIT', 'TRF', 'REF', 'CHECK', 'CHEQUE',
    'WITHDRAWAL', 'ATM', 'CASH'
  ]);
  
  // Split narration into potential keywords
  const keywords = narration.toUpperCase().split(/[^A-Z0-9]+/);

  // Find the first keyword that passes our stricter filter criteria
  const meaningfulKeyword = keywords.find(kw => {
    if (!kw || kw.length < 4) {
      return false; // Ignore empty or short words
    }
    if (noise.has(kw)) {
      return false; // Ignore common noise words
    }
    // A simple test to see if the word is mostly numeric (likely an ID)
    const numbers = (kw.match(/\d/g) || []).length;
    if (numbers > kw.length / 2) {
      return false; // Ignore if more than half the characters are digits
    }
    return true; // This looks like a good keyword
  });

  if (meaningfulKeyword) {
    // Escape any special characters in the keyword for the regex
    const escapedKeyword = meaningfulKeyword.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
    return `.*\\b${escapedKeyword}\\b.*`;
  }

  // If no good keyword is found after all filtering, return a generic pattern
  return '.*';
};

const BulkActionRibbon = ({
  selectedCount,
  knownLedgers,
  onClearSelection,
  onReassign,
  onMarkSuspense,
  onReclassify
}) => {
  const [targetLedger, setTargetLedger] = useState('');

  const handleReassign = () => {
    if (!targetLedger) {
      toast.error("Please select a ledger to re-assign.");
      return;
    }
    onReassign(targetLedger);
  };

  const isVisible = selectedCount > 0;

  return (
    <div
      className={`
        sticky top-[113px] z-20 bg-slate-100 border-b border-slate-300
        transition-all duration-300 ease-in-out overflow-hidden
        ${isVisible ? 'max-h-20 opacity-100' : 'max-h-0 opacity-0 border-none'}
      `}
    >
      <div className="p-2 flex items-center gap-4 text-sm">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClearSelection}>
            <X className="w-4 h-4" />
          </Button>
          <span className="font-bold">{selectedCount}</span> item(s) selected
        </div>

        <Separator orientation="vertical" className="h-8" />

        <div className="flex items-center gap-2">
          <span>Re-assign to:</span>
          {/* Increased width from w-52 to w-96 */}
          <div className="w-96"> 
            <LedgerCombobox ledgers={knownLedgers} value={targetLedger} onValueChange={setTargetLedger} />
          </div>
          {/* Added w-40 for uniform width */}
          <Button size="sm" onClick={handleReassign} disabled={!targetLedger} className="w-40 justify-center">Re-assign</Button>
        </div>

        <Separator orientation="vertical" className="h-8" />

        <Button size="sm" variant="outline" onClick={onMarkSuspense} className="w-40 justify-center">Mark as Suspense</Button>
        <Button size="sm" variant="outline" onClick={onReclassify} className="w-40 justify-center">Re-classify Selected</Button>
      </div>
    </div>
  );
};

const StickyTableHeader = ({ isAllSelected, onToggleAll }) => {
  return (
    // This div mimics the table row. It's NOT a real <thead>.
    <div className="flex w-full bg-slate-50 border-b border-slate-300 font-semibold text-sm">
      <div className="p-2 w-24">Date</div>
      <div className="p-2 w-12 text-center">
        <input
          type="checkbox"
          className="h-4 w-4"
          checked={isAllSelected}
          onChange={onToggleAll}
        />
      </div>
      <div className="p-2 flex-grow">Description</div>
      <div className="p-2 w-32 text-right">Amount</div>
      <div className="p-2 w-24">CR/DR</div>
      <div className="p-2 w-48">Matched Ledger</div>
      <div className="p-2 w-24 text-center">Actions</div>
    </div>
  );
};
const LedgerDistributionChart = ({ data }) => {
  // Define a color palette
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];
  const SPECIAL_COLORS = {
    'Unclassified': '#b0b0b0', // Gray for soft suspense
    'Manual Suspense': '#d0a000', // Amber/Gold for hard suspense
  };

  if (!data || data.length === 0) {
    return <div className="text-center text-sm text-slate-500">No data for chart.</div>;
  }

  return (
    <ResponsiveContainer width="100%" height={120}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          labelLine={false}
          outerRadius={50}
          fill="#8884d8"
          dataKey="value"
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={SPECIAL_COLORS[entry.name] || COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            background: 'white',
            border: '1px solid #ddd',
            borderRadius: '0.5rem',
            fontSize: '12px',
          }}
          formatter={(value, name) => [`${value} transaction(s)`, name]}
        />
      </PieChart>
    </ResponsiveContainer>
  );
};

const StatementPageHeader = ({
  client,
  bankAccount,
  statement,
  stats
}) => {
  if (!client || !bankAccount || !statement || !stats) {
    return <div>Loading header...</div>;
  }

  const {
    totalCount,
    classifiedPercentage,
    hardSuspensePercentage,
    totalInflow,
    totalOutflow,
    softSuspenseCount,
    unclassifiedDebitCount,
    unclassifiedCreditCount,
    chartData
  } = stats;

  return (
    <div className="mb-8">
      {/* Breadcrumb Section */}
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-slate-800">
          {client.name} &gt; {bankAccount.ledger_name}
        </h2>
        <p className="text-sm text-slate-500">{statement.filename}</p>
        {/* We need to calculate statement_period, will do in parent component */}
      </div>

      {/* Stats Dashboard Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 border rounded-lg p-4 bg-slate-50/50">
        {/* Left Column: Progress */}
        <div className="lg:col-span-2 space-y-3">
          <div className="flex justify-between items-center">
            <h3 className="font-semibold text-slate-700">Classification Progress</h3>
            <span className="text-sm font-bold text-slate-800">
              {((classifiedPercentage || 0) + (hardSuspensePercentage || 0)).toFixed(1)}% Complete
            </span>
          </div>
          {/* Stacked Progress Bar */}
          <div className="w-full h-4 bg-slate-200 rounded-full flex overflow-hidden">
            <div
              className="bg-green-500 h-full"
              style={{ width: `${classifiedPercentage}%` }}
              title={`Matched: ${classifiedPercentage.toFixed(1)}%`}
            />
            <div
              className="bg-yellow-500 h-full"
              style={{ width: `${hardSuspensePercentage}%` }}
              title={`Manual Suspense: ${hardSuspensePercentage.toFixed(1)}%`}
            />
          </div>
          <div className="flex justify-between text-xs text-slate-600">
            <span>Total Inflow: <span className="font-bold">₹{totalInflow.toLocaleString('en-IN')}</span></span>
            <span>Total Outflow: <span className="font-bold">₹{totalOutflow.toLocaleString('en-IN')}</span></span>
            <span>Pending Review: <span className="font-bold">{softSuspenseCount} ({unclassifiedDebitCount} Dr, {unclassifiedCreditCount} Cr)</span></span>
          </div>
        </div>
        {/* Right Column: Pie Chart */}
        <div className="space-y-2">
           <h3 className="font-semibold text-slate-700 text-center">Ledger Distribution</h3>
           <LedgerDistributionChart data={chartData} />
        </div>
      </div>
    </div>
  );
};
// Statement Classification Page Component
const StatementDetailsPage = () => {
  const { statementId } = useParams();

  const [statement, setStatement] = useState(null);
  const [client, setClient] = useState(null);
  const [bankAccount, setBankAccount] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);

  const [loading, setLoading] = useState(true); // Renamed from 'loading'
  const [classifying, setClassifying] = useState(false);
  const [pageStats, setPageStats] = useState(null);

  const [isVoucherModalOpen, setIsVoucherModalOpen] = useState(false);
  const [voucherData, setVoucherData] = useState(null);
  const [detachedTransactions, setDetachedTransactions] = useState([]);
  const [knownLedgers, setKnownLedgers] = useState([]);
  const [selectedTxIds, setSelectedTxIds] = useState(new Set());

  const classifiedTxns = useMemo(() => {
    return classificationResult?.classified_transactions || [];
  }, [classificationResult]);


  useEffect(() => {
    const fetchInitialData = async () => {
      setLoading(true);
      try {
        // Step 1: Fetch the primary statement FIRST and wait for it.
        const statementRes = await axios.get(`${API}/statements/${statementId}`);
        const stmtData = statementRes.data;
        setStatement(stmtData);

        // Check if we have the necessary IDs before proceeding.
        if (!stmtData.client_id || !stmtData.bank_account_id) {
            throw new Error("Statement data is missing required client or bank account IDs.");
        }

        // Step 2: Now that we have the IDs, fetch all other data in parallel.
        const [clientRes, bankAccountRes, classificationRes, knownLedgersRes] = await Promise.all([
          axios.get(`${API}/clients/${stmtData.client_id}`),
          axios.get(`${API}/bank-accounts/${stmtData.bank_account_id}`),
          axios.post(`${API}/classify-transactions/${statementId}`),
          axios.get(`${API}/clients/${stmtData.client_id}/known-ledgers`),
        ]);

        setClient(clientRes.data);
        setBankAccount(bankAccountRes.data);
        setKnownLedgers(knownLedgersRes.data);
        
        // Tag transactions with temp IDs for UI state management
        let counter = 0;
        const tagWithId = (t) => ({ ...t, _tempId: counter++ });
        const taggedClassified = classificationRes.data.classified_transactions.map(tagWithId);
        const taggedClusters = classificationRes.data.unmatched_clusters.map(cluster => ({
          ...cluster,
          transactions: cluster.transactions.map(tagWithId)
        }));
        
        setClassificationResult({
          ...classificationRes.data,
          classified_transactions: taggedClassified,
          unmatched_clusters: taggedClusters
        });

      } catch (error) {
        toast.error("Failed to load statement details. The record may be incomplete.");
        console.error("Initial data fetch error:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchInitialData();
  }, [statementId]);

  // --- START: NEW STATS CALCULATION LOGIC ---
  useEffect(() => {
    if (!classificationResult) return;

    const allTransactions = [
      ...classificationResult.classified_transactions,
      ...classificationResult.unmatched_clusters.flatMap(c => c.transactions)
    ];
    const totalCount = allTransactions.length;
    if (totalCount === 0) return;

    let classifiedCount = 0;
    let hardSuspenseCount = 0;
    let softSuspenseCount = 0;
    let totalInflow = 0;
    let totalOutflow = 0;
    let unclassifiedDebitCount = 0;
    let unclassifiedCreditCount = 0;
    const ledgerCounts = {};

    for (const tx of classificationResult.classified_transactions) {
      const isCredit = (tx['CR/DR'] || '').startsWith('CR');
      const amount = parseFloat(String(tx.Amount || '0').replace(/,/g, ''));

      if (isCredit) totalInflow += amount;
      else totalOutflow += amount;

      if (tx.matched_ledger === 'Suspense' && tx.user_confirmed) {
        hardSuspenseCount++;
        ledgerCounts['Manual Suspense'] = (ledgerCounts['Manual Suspense'] || 0) + 1;
      } else if (tx.matched_ledger !== 'Suspense') {
        classifiedCount++;
        ledgerCounts[tx.matched_ledger] = (ledgerCounts[tx.matched_ledger] || 0) + 1;
      }
    }
    
    for (const tx of classificationResult.unmatched_clusters.flatMap(c => c.transactions)) {
        softSuspenseCount++;
        const isCredit = (tx['CR/DR'] || '').startsWith('CR');
        const amount = parseFloat(String(tx.Amount || '0').replace(/,/g, ''));
        if (isCredit) {
            totalInflow += amount;
            unclassifiedCreditCount++;
        } else {
            totalOutflow += amount;
            unclassifiedDebitCount++;
        }
    }
    if (softSuspenseCount > 0) {
        ledgerCounts['Unclassified'] = softSuspenseCount;
    }

    setPageStats({
      totalCount,
      classifiedPercentage: (classifiedCount / totalCount) * 100,
      hardSuspensePercentage: (hardSuspenseCount / totalCount) * 100,
      totalInflow,
      totalOutflow,
      softSuspenseCount,
      unclassifiedDebitCount,
      unclassifiedCreditCount,
      chartData: Object.entries(ledgerCounts).map(([name, value]) => ({ name, value })),
    });

  }, [classificationResult]);
  // --- END: NEW STATS CALCULATION LOGIC --

  const handleToggleRow = (txId) => {
    setSelectedTxIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(txId)) {
        newSet.delete(txId);
      } else {
        newSet.add(txId);
      }
      return newSet;
    });
  };
  
  const handleToggleAll = () => {
    setSelectedTxIds(prev => {
      // If not everything is selected, select all. Otherwise, clear selection.
      if (prev.size < classifiedTxns.length) {
        return new Set(classifiedTxns.map(t => t._tempId));
      } else {
        return new Set();
      }
    });
  };
   const handleBulkUpdate = async (type, payload = {}) => {
    if (selectedTxIds.size === 0) {
      toast.info("No transactions selected.");
      return;
    }
    
    // Create a new version of the transactions list with the updates applied
    const updatedTransactions = classifiedTxns.map(tx => {
      // If this transaction is not selected, return it as is
      if (!selectedTxIds.has(tx._tempId)) {
        return tx;
      }
      
      // If it is selected, apply the change
      if (type === 'reassign') {
        return { ...tx, matched_ledger: payload.targetLedger, user_confirmed: true };
      }
      
      if (type === 'markSuspense') {
        return { ...tx, matched_ledger: 'Suspense', user_confirmed: true };
      }
      
      return tx; // Fallback
    });
    
    const newResult = {
      ...classificationResult,
      classified_transactions: updatedTransactions,
    };
    
    // 1. Optimistically update the UI so it feels instant
    setClassificationResult(newResult);
    
    // 2. Persist the changes to the database
    await saveClassificationState(newResult);
    
    // 3. Clean up the selection and notify the user
    setSelectedTxIds(new Set());
    toast.success(`${selectedTxIds.size} transaction(s) have been updated.`);
  };

  const handleBulkReclassify = async () => {
    if (selectedTxIds.size === 0) {
      toast.info("No transactions selected.");
      return;
    }

    // 1. Gather the full transaction objects that are selected
    const transactionsToReclassify = classifiedTxns.filter(tx => selectedTxIds.has(tx._tempId));
    
    try {
      // 2. Call the new backend endpoint
      const response = await axios.post(
        `${API}/statements/${statementId}/reclassify-subset`,
        { transactions: transactionsToReclassify }
      );
      
      const reclassifiedResults = response.data;

      // 3. Merge the results back into the main list
      const resultMap = new Map(reclassifiedResults.map(tx => [tx._tempId, tx]));
      const updatedTransactions = classifiedTxns.map(originalTx => 
        resultMap.has(originalTx._tempId) ? resultMap.get(originalTx._tempId) : originalTx
      );

      const newResult = {
        ...classificationResult,
        classified_transactions: updatedTransactions,
      };

      // 4. Optimistically update the UI and persist the new state
      setClassificationResult(newResult);
      await saveClassificationState(newResult);
      
      // 5. Clean up and notify
      setSelectedTxIds(new Set());
      toast.success(`${reclassifiedResults.length} transaction(s) were re-classified and saved.`);

    } catch (error) {
      toast.error("Failed to re-classify transactions.");
      console.error("Bulk re-classify error:", error);
    }
  };
  // --- START: NEW UNIFIED DATA PROCESSING ---
  const runClassification = useCallback(async (isForced = false) => {
    // 1. Add a "guard clause" to prevent running if the statement isn't loaded yet.
    if (!statement) {
      return; 
    }
    setClassifying(true);
    try {
      try {
        const ledgersRes = await axios.get(`${API}/clients/${statement.client_id}/known-ledgers`);
        setKnownLedgers(ledgersRes.data);
      } catch (err) {
        toast.error("Could not fetch known ledgers list.");
        setKnownLedgers([]); // Set to empty array on failure
      }
      const url = isForced 
        ? `${API}/classify-transactions/${statementId}?force_reclassify=true`
        : `${API}/classify-transactions/${statementId}`;
      
      const response = await axios.post(url);

      // Add a temporary, unique frontend ID to every transaction object for reliable state updates.
      let counter = 0;
      const tagWithId = (t) => ({ ...t, _tempId: counter++ });

      const taggedClassified = response.data.classified_transactions.map(tagWithId);
      const taggedClusters = response.data.unmatched_clusters.map(cluster => ({
        ...cluster,
        transactions: cluster.transactions.map(tagWithId)
      }));

      setClassificationResult({
        ...response.data,
        classified_transactions: taggedClassified,
        unmatched_clusters: taggedClusters
      });
      
      if (isForced) {
        setDetachedTransactions([]); // Clear detached items on a forced re-classify
      }

    } catch (error) {
      toast.error("Failed to run classification.");
    } finally {
      setClassifying(false);
    }
  }, [statementId, statement]);

  const saveClassificationState = async (resultToSave) => {
    if (!resultToSave) return;
    
    // Create a "clean" version of the data by removing our temporary IDs before sending to backend.
    const cleanData = resultToSave.classified_transactions.map(({ _tempId, ...rest }) => rest);

    try {
      await axios.post(`${API}/statements/${statementId}/update-transactions`, {
        processed_data: cleanData,
      });
    } catch (error) {
      toast.error("Failed to save progress.");
    }
  };

  const handleMarkAsSuspense = async (transactionsToMark) => {
    if (!classificationResult) return;
    const idsToUpdate = new Set(transactionsToMark.map(t => t._tempId));

    // Work on immutable copies of current state
    const prevClassified = classificationResult.classified_transactions.map(t => ({ ...t }));
    const prevClusters = classificationResult.unmatched_clusters.map(c => ({ ...c, transactions: c.transactions.map(tx => ({ ...tx })) }));

    // 1) Update already-classified transactions (mark confirmed + ensure ledger)



    for (let i = 0; i < prevClassified.length; i++) {
     
      const t = prevClassified[i];
      if (idsToUpdate.has(t._tempId)) {
        prevClassified[i] = { ...t, matched_ledger: t.matched_ledger || 'Suspense', user_confirmed: true };
        idsToUpdate.delete(t._tempId); // handled
      }
    }

    // 2) Move matching transactions from clusters into classified list
    const moved = [];
    const newClusters = prevClusters.map(cluster => {
      const remaining = [];
      for (const tx of cluster.transactions) {
        if (idsToUpdate.has(tx._tempId)) {
          moved.push({ ...tx, matched_ledger: 'Suspense', user_confirmed: true });
          idsToUpdate.delete(tx._tempId);
        } else {
          remaining.push(tx);
        }
      }
      return { ...cluster, transactions: remaining };
    }).filter(c => c.transactions.length > 0);

    // Log any ids that were not found (optional debug)
    if (idsToUpdate.size > 0) {
      console.warn('Some transactions to mark as Suspense were not found in current state:', Array.from(idsToUpdate));
    }

    const newClassified = [...prevClassified, ...moved];

    const newResult = {
      ...classificationResult,
      classified_transactions: newClassified,
      unmatched_clusters: newClusters
    };

    // Optimistic UI update
    setClassificationResult(newResult);

    // Persist the exact same new state to backend
    try {
      await saveClassificationState(newResult);
      toast.success(`${transactionsToMark.length} transaction(s) marked as Suspense.`);
    } catch (err) {
      toast.error('Failed to save Suspense changes. Changes may not be persisted.');
      console.error('saveClassificationState error:', err);
    }
  };

  // --- ADD THIS NEW FUNCTION ---
  const handleDetachTransaction = (transaction, clusterId) => {
    if (!classificationResult) return;

    // Remove the transaction from the matching cluster
    const newClusters = classificationResult.unmatched_clusters
      .map(cluster => {
        if (cluster.cluster_id !== clusterId) return cluster;
        return {
          ...cluster,
          transactions: cluster.transactions.filter(tx => tx._tempId !== transaction._tempId)
        };
      })
      .filter(c => c.transactions && c.transactions.length > 0);

    // Add to detachedTransactions (avoid duplicates)
    setDetachedTransactions(prev => {
      if (prev.some(t => t._tempId === transaction._tempId)) return prev;
      return [...prev, transaction];
    });

    // Update UI optimistically
    const newResult = { ...classificationResult, unmatched_clusters: newClusters };
    setClassificationResult(newResult);

    // Mark state dirty so save effect can persist if required
    toast.info('Transaction detached for re-clustering.');
  };
  // --- END ADDITION ---

  const handleFlagAsIncorrect = async (transactionToFlag) => {
    if (!classificationResult) return;

    // Work on immutable copies
    const prevClassified = classificationResult.classified_transactions.map(t => ({ ...t }));
    const prevClusters = classificationResult.unmatched_clusters.map(c => ({ ...c, transactions: c.transactions.map(tx => ({ ...tx })) }));

    // Find the transaction in classified list (prefer _tempId)
    const idx = prevClassified.findIndex(t => t._tempId === transactionToFlag._tempId
      || (t.Narration === transactionToFlag.Narration && String(t.Amount) === String(transactionToFlag.Amount) && t.Date === transactionToFlag.Date)
    );
    if (idx === -1) {
      toast.error("Could not locate transaction to flag as incorrect.");
      return;
    }

    const origTx = prevClassified[idx];

    // Build the version to persist: set matched_ledger -> 'Suspense' and remove user_confirmed
    const { user_confirmed, ...restFields } = origTx;
    const txForSave = { ...restFields, matched_ledger: 'Suspense' };
    // Also keep any frontend temp id when showing in UI
    const txForUI = { ...txForSave, _tempId: origTx._tempId };

    // Prepare payload state to send to backend: replace the transaction in classified list with txForSave (so backend updates it)
    const classifiedForSave = prevClassified.map((t, i) => i === idx ? txForSave : { ...t });

    const saveResult = {
      ...classificationResult,
      classified_transactions: classifiedForSave,
      // don't need to modify unmatched_clusters for save payload
    };

    // Prepare optimistic UI state: remove from classified list and add a small cluster containing the transaction
    const newClassifiedUI = prevClassified.filter((_, i) => i !== idx);
    const newCluster = {
      cluster_id: `manual-${Date.now()}-${Math.floor(Math.random()*1000)}`,
      suggested_regex: generateSimpleRegex(txForUI.Narration || ''),
      transactions: [txForUI]
    };
    const newUnmatchedClustersUI = [newCluster, ...prevClusters];

    const optimisticResult = {
      ...classificationResult,
      classified_transactions: newClassifiedUI,
      unmatched_clusters: newUnmatchedClustersUI
    };

    // Apply optimistic UI
    setClassificationResult(optimisticResult);

    try {
      // Persist the update (ensures backend updates existing entry to Suspense without user_confirmed)
      await saveClassificationState(saveResult);

      // Re-run classification on server to rebuild clusters correctly
      await runClassification();

      toast.success('Marked as incorrect and returned to unclassified clusters.');
    } catch (err) {
      console.error('handleFlagAsIncorrect error:', err);
      // Revert UI on failure
      setClassificationResult(classificationResult);
      toast.error('Failed to mark transaction as incorrect. Changes were not saved.');
    }
  };

  // --- ADD THIS ENTIRE FUNCTION ---
  const handleGenerateVouchers = async () => {
    setClassifying(true);
    try {
      // Step 1: Always save the latest state before generating.
      await saveClassificationState(classificationResult);
      
      // Step 2: Fetch the counts and suggested filename for the modal.
      const response = await axios.get(`${API}/vouchers/${statementId}/summary`);
      setVoucherData(response.data);
      setIsVoucherModalOpen(true); // Open the modal
      
    } catch (error) {
      toast.error("Could not prepare voucher data. Please try again.");
    } finally {
      setClassifying(false);
    }
  };


  // --- ADD THIS LOGIC inside StatementDetailsPage, before the return statement ---
  const otherNarrations = useMemo(() => {
        if (!classificationResult) return [];
        // Get all narrations from transactions that were already successfully matched.
        return classificationResult.classified_transactions
            .filter(t => t.matched_ledger !== 'Suspense')
            .map(t => t.Narration);
    }, [classificationResult]);

  if (loading || !classificationResult || !pageStats) {
    return <div>Loading and processing statement...</div>;
  }
   const classifiedTxnsToShow = (classificationResult?.classified_transactions || []).filter(
    t => t.matched_ledger !== "Suspense" || t.user_confirmed === true
  );
  const isAllSelected = classifiedTxnsToShow.length > 0 && selectedTxIds.size === classifiedTxnsToShow.length;
  // --- END OF FIX ---
  
  return (
    <div className="space-y-8">
      {/* --- START OF ADDITION (3 of 3): Render the new header --- */}
      <StatementPageHeader
        client={client}
        bankAccount={bankAccount}
        statement={statement}
        stats={pageStats}
      />
      {/* --- END OF ADDITION (3 of 3) --- */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Classification Review</h1>
          <p className="text-slate-600 mt-2">
            {(pageStats.totalCount - pageStats.softSuspenseCount)} of {pageStats.totalCount} transactions classified.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={handleGenerateVouchers} disabled={classifying} className="bg-blue-600 hover:bg-blue-700">
            <Download className="w-4 h-4 mr-2" />
            Generate Vouchers
          </Button>
          <Button onClick={() => runClassification(true)} disabled={classifying}>
            <RefreshCw className={`w-4 h-4 mr-2 ${classifying ? 'animate-spin' : ''}`} />
            Re-classify
          </Button>
        </div>
      </div>

      {/* Unmatched Transaction Clusters Section */}
      <Card className="border-0 shadow-sm">
        <CardHeader>
          <CardTitle>Unmatched Transaction Clusters</CardTitle>
          <CardDescription>
            Review these groups of similar transactions and create new rules.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {classificationResult.unmatched_clusters.length > 0 ? (
            // --- THIS IS THE NEW, INTERACTIVE REPLACEMENT ---
            classificationResult.unmatched_clusters.map(cluster => (
              <ClusterCard 
                key={cluster.cluster_id} 
                cluster={cluster} 
                clientId={statement.client_id}
                onRuleCreated={() => runClassification(true)}
                otherNarrations={otherNarrations} // Pass the new prop down
                onDetach={handleDetachTransaction}
                onMarkAsSuspense={handleMarkAsSuspense} // <-- ADD THIS
                knownLedgers={knownLedgers} // <-- ADD THIS LINE

              />
            ))
            // --- END OF NEW REPLACEMENT ---
          ) : (
            <p className="text-center text-slate-500 py-4">Congratulations! All transactions were matched.</p>
          )}
        </CardContent>
      </Card>
      {/* --- START: ADD THIS ENTIRE JSX BLOCK --- */}
      {detachedTransactions.length > 0 && (
        <Card className="border-0 shadow-sm bg-yellow-50 border-yellow-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Filter className="w-5 h-5 text-yellow-700" />
              Detached Transactions for Re-Clustering
            </CardTitle>
            <CardDescription>
              These transactions have been removed from their clusters. You can create rules for them individually or re-cluster them.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* We will reuse the ClusterCard to display these, as it's already styled nicely */}
            <ClusterCard
                cluster={{
                    cluster_id: 'detached-group',
                    transactions: detachedTransactions,
                        suggested_regex: detachedTransactions.length > 0 
      ? generateSimpleRegex(detachedTransactions[0].Narration) 
      : ''

                }}
                clientId={statement.client_id}
                onRuleCreated={runClassification} // You can still create a rule from a single detached item
                otherNarrations={otherNarrations}
                onDetach={() => {}} // Detaching from this group does nothing
                onMarkAsSuspense={handleMarkAsSuspense} // <-- ADD THIS
                knownLedgers={knownLedgers} // <-- ADD THIS LINE

            />
          </CardContent>
        </Card>
      )}
      {/* Classified Transactions Table Section */}
      <div className="bg-white border-0 shadow-sm rounded-lg">
        {/* This div acts as the CardHeader, with controlled bottom padding (pb-4) to close the gap */}
        <div className="p-6 pb-4">
          <h3 className="text-2xl font-semibold leading-none tracking-tight">
            Classified Transactions
          </h3>
        </div>

        {/* The sticky container for the header and ribbon */}
        <div className="sticky top-16 z-20">
          <StickyTableHeader 
            isAllSelected={isAllSelected}
            onToggleAll={handleToggleAll}
          />
          <BulkActionRibbon
            selectedCount={selectedTxIds.size}
            knownLedgers={knownLedgers}
            onClearSelection={() => setSelectedTxIds(new Set())}
            onReassign={(ledger) => handleBulkUpdate('reassign', { targetLedger: ledger })}
            onMarkSuspense={() => handleBulkUpdate('markSuspense')}
            onReclassify={handleBulkReclassify}
          />
        </div>
        
        {/* The table now sits inside the same container, with no extra wrappers */}
        <ClassifiedTransactionsTable
          transactions={classifiedTxns}
          onFlagAsIncorrect={handleFlagAsIncorrect}
          selectedTxIds={selectedTxIds}
          onToggleRow={handleToggleRow}
        />
      </div>
      {/* Voucher Modal */}{/* --- ADD THIS COMPONENT AT THE END --- */}
      <DownloadVoucherModal 
        isOpen={isVoucherModalOpen}
        onClose={() => setIsVoucherModalOpen(false)}
        data={voucherData}
        statementId={statementId}
      />
      {/* --- END OF ADDITION --- */}
    </div>
  );
};

// --- END OF NEW COMPONENT ---
// --- ADD THIS ENTIRE NEW PAGE COMPONENT ---
const LedgerSamplesPage = () => {
  const { clientId, ledgerId } = useParams();
  const [ledger, setLedger] = useState(null);
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalSamples, setTotalSamples] = useState(0);
  const [sampleToDelete, setSampleToDelete] = useState(null);
  const limit = 100;

  const fetchSamples = useCallback(async (currentPage) => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/known-ledgers/${ledgerId}/samples?page=${currentPage}&limit=${limit}`);
      const { samples, total_samples } = response.data;
      
      // Since our ledger name isn't in this response, we can get it from the client summary
      if (!ledger) {
        const summaryRes = await axios.get(`${API}/clients/${clientId}/known-ledgers/summary`);
        const currentLedger = summaryRes.data.find(l => l.id === ledgerId);
        setLedger(currentLedger);
      }

      setSamples(samples);
      setTotalSamples(total_samples);
      setTotalPages(Math.ceil(total_samples / limit));
      setPage(currentPage);
    } catch (error) {
      toast.error("Failed to fetch ledger samples.");
    } finally {
      setLoading(false);
    }
  }, [ledgerId, clientId, ledger]);

  useEffect(() => {
    fetchSamples(1);
  }, [fetchSamples]);

  const handleDeleteSample = async () => {
    if (!sampleToDelete) return;

    // Optimistic UI update
    const originalSamples = samples;
    setSamples(prev => prev.filter(s => s !== sampleToDelete));
    
    try {
      await axios.delete(`${API}/known-ledgers/${ledgerId}/samples`, {
        data: sampleToDelete
      });
      toast.success("Sample deleted successfully.");
      // On successful delete, re-fetch to get accurate total count
      setTotalSamples(prev => prev - 1);

    } catch (error) {
      toast.error("Failed to delete sample.");
      // Revert UI on failure
      setSamples(originalSamples);
    } finally {
      setSampleToDelete(null); // Close the dialog
    }
  };

  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= totalPages) {
      fetchSamples(newPage);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Ledger Sample Review</h1>
        <p className="text-slate-600 mt-2">
          Reviewing samples for: <span className="font-bold">{ledger?.ledger_name || '...'}</span> ({totalSamples} total samples)
        </p>
      </div>

      <Card className="border-0 shadow-sm">
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-slate-50">
                  <th className="p-3 text-left font-semibold">Narration</th>
                  <th className="p-3 text-right font-semibold w-32">Amount</th>
                  <th className="p-3 text-left font-semibold w-24">Type</th>
                  <th className="p-3 text-center font-semibold w-24">Actions</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan="4" className="text-center p-8">Loading samples...</td></tr>
                ) : (
                  samples.map((sample, index) => (
                    <tr key={index} className="border-b hover:bg-slate-50">
                      <td className="p-3 max-w-xl truncate">{sample.narration}</td>
                      <td className="p-3 text-right font-mono">{sample.amount.toLocaleString('en-IN')}</td>
                      <td className="p-3">
                        <Badge variant={sample.type === 'Credit' ? 'default' : 'destructive'} className={sample.type === 'Credit' ? 'bg-green-600' : 'bg-red-600'}>
                          {sample.type}
                        </Badge>
                      </td>
                      <td className="p-3 text-center">
                        <Button variant="destructive" size="icon" className="h-8 w-8" onClick={() => setSampleToDelete(sample)}>
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
        {!loading && (
          <div className="p-4 border-t flex items-center justify-between">
            <span className="text-sm text-slate-600">
              Page {page} of {totalPages}
            </span>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={() => handlePageChange(page - 1)} disabled={page === 1}>Previous</Button>
              <Button variant="outline" size="sm" onClick={() => handlePageChange(page + 1)} disabled={page === totalPages}>Next</Button>
            </div>
          </div>
        )}
      </Card>

      {/* Delete Confirmation Dialog */}
      <Dialog open={!!sampleToDelete} onOpenChange={() => setSampleToDelete(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Are you sure?</DialogTitle>
            <DialogDescription>
              This will permanently delete this sample narration. This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-4 pt-4">
            <Button variant="outline" onClick={() => setSampleToDelete(null)}>Cancel</Button>
            <Button variant="destructive" onClick={handleDeleteSample}>Confirm Delete</Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};
// --- END OF NEW PAGE COMPONENT ---

// Navigation Component
const Navigation = () => {
  return (
    <nav className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between h-16">
            <Link to="/" className="flex items-center space-x-3">
            {/* The container now has NO padding */}
            <div 
              className="h-16 w-16 rounded-xl"
              style={{
                backgroundImage: `
                  linear-gradient(to right, #e2e8f0 1px, transparent 1px),
                  linear-gradient(to bottom, #e2e8f0 1px, transparent 1px)
                `,
                // This creates wider, rectangular cells
                backgroundSize: '15px 10px', 
              }}
            >
              {/* Padding is applied to the image itself to prevent shrinking */}
              <img src={logo} alt="Acutant Labs Logo" className="h-full w-full p-0.1" />
            </div>
            <span className="text-3xl font-bold text-slate-900">ABS Processor</span>
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


// --- FIND AND REPLACE THE ENTIRE RuleEditModal COMPONENT ---
const RuleEditModal = ({ isOpen, onClose, rule, clientId, onSuccess, additionalSamples = [] }) => {
  const [ledgerName, setLedgerName] = useState('');
  const [regexPattern, setRegexPattern] = useState('');
  const [loading, setLoading] = useState(false);
  const [testResults, setTestResults] = useState({ originals: [], additionals: [] });
  const isEditing = !!rule;
  
  const decodeRegex = (regex) => {
    if (!regex || !regex.trim()) return [];
    const keywordMatches = [...regex.matchAll(/\\b([A-Za-z0-9_ -]+)\\b/g)];
    if (keywordMatches.length === 0) {
      return [[{ type: 'error', text: "Could not find any whole words to explain in this pattern." }]];
    }
    const explanation = [];
    explanation.push([
      { type: 'intro', text: 'The text must contain ' },
      { type: 'keyword_punctuation', text: 'the word "' },
      { type: 'keyword_text', text: keywordMatches[0][1] },
      { type: 'keyword_punctuation', text: '"' }
    ]);
    for (let i = 1; i < keywordMatches.length; i++) {
      explanation.push([
        { type: 'connector', text: '   followed (anywhere later) by ' },
        { type: 'keyword_punctuation', text: 'the word "' },
        { type: 'keyword_text', text: keywordMatches[i][1] },
        { type: 'keyword_punctuation', text: '"' }
      ]);
    }
    return explanation;
  };
   useEffect(() => {
    if (rule) {
      setLedgerName(rule.ledger_name);
      setRegexPattern(rule.regex_pattern);
    } else {
      setLedgerName('');
      setRegexPattern('');
    }
  }, [rule]);

  // Enhanced Live validation useEffect
  useEffect(() => {
    const originalNarrations = rule?.sample_narrations || [];
    const originalSet = new Set(originalNarrations);

    const additionalNarrations = (additionalSamples || [])
      .map(s => s.narration)
      .filter(narration => !originalSet.has(narration)); // De-duplicate

    const runTest = (narration) => {
      const { html, matches } = getHighlightedHtml(regexPattern, narration);
      return { narration, html, matches };
    };

    setTestResults({
      originals: originalNarrations.map(runTest),
      additionals: additionalNarrations.map(runTest)
    });

  }, [regexPattern, rule, additionalSamples, getHighlightedHtml]); // Added dependencies
// --- END OF REPLACEMENT ---
  // Helper function for highlighting (reused from ClusterCard)
  const getHighlightedHtml = (regexStr, text) => {
    if (!text) return { html: '', matches: false };
    try {
      const re = new RegExp(regexStr, 'i');
      const match = text.match(re);
      if (!match || !match[0] || !regexStr) {
        return { html: text.replace(/</g, '&lt;'), matches: false };
      }
      const escapedText = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
      const highlightedHtml = escapedText.replace(match[0], `<span class="bg-green-200 font-bold px-1 rounded">${match[0]}</span>`);
      return { html: highlightedHtml, matches: true };
    } catch (e) {
      return { html: text.replace(/</g, '&lt;'), matches: false };
    }
  };

  
  const handleSubmit = async () => {
    if (!ledgerName.trim() || !regexPattern.trim()) {
      toast.error("Ledger Name and Regex Pattern are required.");
      return;
    }
    setLoading(true);
    try {
      const payload = { client_id: clientId, ledger_name: ledgerName, regex_pattern: regexPattern, sample_narrations: rule?.sample_narrations || [] };
      if (isEditing) {
        await axios.put(`${API}/ledger-rules/${rule.id}`, { ledger_name: ledgerName, regex_pattern: regexPattern });
        toast.success("Rule updated successfully!");
      } else {
        await axios.post(`${API}/ledger-rules`, payload);
        toast.success("Rule created successfully!");
      }
      onSuccess();
      onClose();
    } catch (error) {
      toast.error(`Failed to ${isEditing ? 'update' : 'create'} rule.`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{isEditing ? 'Edit Rule' : 'Create New Rule'}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div>
            <Label>Ledger Name</Label>
            <Input value={ledgerName} onChange={(e) => setLedgerName(e.target.value)} placeholder="e.g., Office Supplies" />
          </div>
          <div>
            <Label>Regex Pattern</Label>
            <Textarea value={regexPattern} onChange={(e) => setRegexPattern(e.target.value)} placeholder="e.g., .*STAPLES.*" className="font-mono" />
          </div>
          {/* --- START OF ADDITION: Regex Decoder --- */}
          <div className="mt-2 p-3 bg-slate-800 text-slate-300 rounded-md font-mono text-xs">
            <p className="font-bold text-white mb-1">Explanation:</p>
            <div>
              {decodeRegex(regexPattern).map((line, lineIndex) => (
                <div key={lineIndex}>
                  {line.map((part, partIndex) => (
                    <span key={partIndex} className={
                      part.type === 'keyword_text' ? 'text-green-400 font-bold' :
                      part.type === 'keyword_punctuation' ? 'text-pink-400 font-semibold' :
                      part.type === 'connector' ? 'text-amber-400' : 'text-slate-300'
                    }>
                      {part.text}
                    </span>
                  ))}
                </div>
              ))}
            </div>
          </div>
          {/* --- END OF ADDITION: Regex Decoder --- */}
            {(testResults.originals.length > 0 || testResults.additionals.length > 0) && (
            <>
              <Separator />
              <div className="space-y-2">
                <Label>Live Test Results</Label>
                <ScrollArea className="h-60 p-2 border rounded-md bg-slate-50">
                  <div className="text-xs space-y-3">
                    {/* Original Samples Section */}
                    {testResults.originals.length > 0 && (
                      <div>
                        <p className="font-semibold mb-1">Original Samples:</p>
                        {testResults.originals.map((result, index) => (
                          <div key={`orig-${index}`} className="flex items-start gap-2">
                            <div className="flex-shrink-0 pt-0.5">
                              {result.matches ? <Check className="w-4 h-4 text-green-600" /> : <X className="w-4 h-4 text-red-600" />}
                            </div>
                            <div className="flex-grow" dangerouslySetInnerHTML={{ __html: result.html }} />
                          </div>
                        ))}
                      </div>
                    )}
                    {/* Additional Samples Section */}
                    {testResults.additionals.length > 0 && (
                      <div>
                        <p className="font-semibold mt-2 mb-1">Additional Samples from History:</p>
                        {testResults.additionals.map((result, index) => (
                          <div key={`add-${index}`} className="flex items-start gap-2">
                            <div className="flex-shrink-0 pt-0.5">
                              {result.matches ? <Check className="w-4 h-4 text-green-600" /> : <X className="w-4 h-4 text-red-600" />}
                            </div>
                            <div className="flex-grow" dangerouslySetInnerHTML={{ __html: result.html }} />
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </ScrollArea>
              </div>
            </>
          )}
        </div>
        <div className="flex justify-end gap-4 pt-4 border-t">
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={handleSubmit} disabled={loading}>
            {loading ? 'Saving...' : 'Save Rule'}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
// --- END OF REPLACEMENT ---

// --- ADD THIS ENTIRE NEW PAGE COMPONENT ---
// In App.js

// --- FIND AND REPLACE THE ENTIRE PatternManagementPage COMPONENT ---
const PatternManagementPage = () => {
  const [clients, setClients] = useState([]);
  const [selectedClient, setSelectedClient] = useState('');
  const [rules, setRules] = useState([]);
  const [loading, setLoading] = useState(false);
  
  // State for rule health stats
  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [sortConfig, setSortConfig] = useState({ key: 'ledger_name', direction: 'ascending' });

  
  // Modal states
  const [showModal, setShowModal] = useState(false);
  const [currentRule, setCurrentRule] = useState(null);
  const [ruleToDelete, setRuleToDelete] = useState(null);

  useEffect(() => {
    const fetchClients = async () => {
      try {
        const response = await axios.get(`${API}/clients`);
        setClients(response.data);
      } catch (error) { toast.error('Failed to fetch clients'); }
    };
    fetchClients();
  }, []);

  const fetchRules = async (clientId) => {
    if (!clientId) {
      setRules([]);
      setStats(null); // Clear stats when client changes
      return;
    }
    setLoading(true);
    try {
      const response = await axios.get(`${API}/ledger-rules/${clientId}`);
      setRules(response.data);
    } catch (error) {
      toast.error('Failed to fetch rules for client.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRules(selectedClient);
  }, [selectedClient]);

  // Memoized sorting logic
  const sortedRules = useMemo(() => {
    let sortableItems = [...rules];
    if (sortConfig !== null) {
      sortableItems.sort((a, b) => {
        let aValue, bValue;

        if (sortConfig.key === 'total_matches' || sortConfig.key === 'last_used') {
          const aStat = stats?.[a.ledger_name];
          const bStat = stats?.[b.ledger_name];
          
          if (sortConfig.key === 'total_matches') {
            aValue = aStat?.total_matches || 0;
            bValue = bStat?.total_matches || 0;
          } else { // last_used
            aValue = aStat?.last_used ? new Date(aStat.last_used) : new Date(0); // Oldest date for null
            bValue = bStat?.last_used ? new Date(bStat.last_used) : new Date(0);
          }
        } else { // ledger_name
          aValue = a.ledger_name;
          bValue = b.ledger_name;
        }

        if (aValue < bValue) return sortConfig.direction === 'ascending' ? -1 : 1;
        if (aValue > bValue) return sortConfig.direction === 'ascending' ? 1 : -1;
        return 0;
      });
    }
    return sortableItems;
  }, [rules, sortConfig, stats]);

  const requestSort = (key) => {
    let direction = 'ascending';
    if (sortConfig.key === key && sortConfig.direction === 'ascending') {
      direction = 'descending';
    }
    setSortConfig({ key, direction });
  };

  const getSortIcon = (name) => {
    if (sortConfig.key !== name) return null;
    if (sortConfig.direction === 'ascending') return <ArrowUp className="w-4 h-4 inline ml-1" />;
    return <ArrowDown className="w-4 h-4 inline ml-1" />;
  };

  const handleCalculateStats = async () => {
    if (!selectedClient) return;
    setStatsLoading(true);
    try {
      const response = await axios.get(`${API}/clients/${selectedClient}/rule-stats`);
      setStats(response.data.stats);
      toast.success("Health stats calculated!");
    } catch (error) {
      toast.error("Failed to calculate stats.");
    } finally {
      setStatsLoading(false);
    }
  };

  const handleOpenModal = (rule = null) => {
    setCurrentRule(rule);
    setShowModal(true);
  };

  const handleDeleteRule = async () => {
    if (!ruleToDelete) return;
    try {
      await axios.delete(`${API}/ledger-rules/${ruleToDelete.id}`);
      toast.success("Rule deleted successfully!");
      setRuleToDelete(null);
      fetchRules(selectedClient);
    } catch (error) { toast.error("Failed to delete rule."); }
  };

   return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Pattern Management</h1>
        <p className="text-slate-600 mt-2">Create, view, edit, and delete ledger rules for each client.</p>
      </div>

      <Card className="border-0 shadow-sm">
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <div className="w-1/3">
            <Label>Select Client</Label>
            <Select onValueChange={setSelectedClient}>
              <SelectTrigger><SelectValue placeholder="Choose a client..." /></SelectTrigger>
              <SelectContent>
                {clients.map(client => <SelectItem key={client.id} value={client.id}>{client.name}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={handleCalculateStats} disabled={!selectedClient || statsLoading}>
              <Zap className="w-4 h-4 mr-2" />
              {statsLoading ? 'Calculating...' : 'Calculate Health Stats'}
            </Button>
            <Button onClick={() => handleOpenModal()} disabled={!selectedClient}>
              <Plus className="w-4 h-4 mr-2" /> Add New Rule
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {selectedClient ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="p-2 text-left font-semibold cursor-pointer hover:bg-slate-50" onClick={() => requestSort('ledger_name')}>
                      Ledger Name {getSortIcon('ledger_name')}
                    </th>
                    <th className="p-2 text-left font-semibold">Regex Pattern</th>
                    <th className="p-2 text-left font-semibold w-32 cursor-pointer hover:bg-slate-50" onClick={() => requestSort('total_matches')}>
                      Total Matches {getSortIcon('total_matches')}
                    </th>
                    <th className="p-2 text-left font-semibold w-32 cursor-pointer hover:bg-slate-50" onClick={() => requestSort('last_used')}>
                      Last Used {getSortIcon('last_used')}
                    </th>
                    <th className="p-2 text-center font-semibold w-32">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedRules.map(rule => (
                    <tr key={rule.id} className="border-b hover:bg-slate-50">
                      <td className="p-2 font-medium">{rule.ledger_name}</td>
                      <td className="p-2 font-mono text-xs">{rule.regex_pattern}</td>
                      <td className="p-2">{stats?.[rule.ledger_name]?.total_matches ?? '---'}</td>
                      <td className="p-2">{stats?.[rule.ledger_name]?.last_used ?? '---'}</td>
                      <td className="p-2 text-center">
                        <div className="flex gap-2 justify-center">
                          <Button variant="outline" size="icon" className="h-8 w-8" onClick={() => handleOpenModal(rule)}>
                            <Edit className="h-4 w-4" />
                          </Button>
                          <Button variant="destructive" size="icon" className="h-8 w-8" onClick={() => setRuleToDelete(rule)}>
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {loading && <p className="text-center py-4">Loading rules...</p>}
              {!loading && rules.length === 0 && <p className="text-center text-slate-500 py-8">No rules found for this client.</p>}
            </div>
          ) : (
            <p className="text-center text-slate-500 py-8">Please select a client to see their rules.</p>
          )}
        </CardContent>
      </Card>
      
      <RuleEditModal 
        isOpen={showModal}
        onClose={() => setShowModal(false)}
        rule={currentRule}
        clientId={selectedClient}
        onSuccess={() => fetchRules(selectedClient)}
      />

      <Dialog open={!!ruleToDelete} onOpenChange={() => setRuleToDelete(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Are you sure?</DialogTitle>
            <DialogDescription>
              This will permanently delete the rule for <span className="font-bold">"{ruleToDelete?.ledger_name}"</span>. This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-4 pt-4">
            <Button variant="outline" onClick={() => setRuleToDelete(null)}>Cancel</Button>
            <Button variant="destructive" onClick={handleDeleteRule}>Confirm Delete</Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};
// --- END OF REPLACEMENT ---
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
            <Route path="/patterns" element={<PatternManagementPage />} />
            <Route path="/clients/:clientId/ledgers/:ledgerId" element={<LedgerSamplesPage />} />
            <Route path="/clients/:clientId" element={<ClientDetailsPage />} />
            <Route path="/statements/:statementId" element={<StatementDetailsPage />} />
          </Routes>
        </main>
      </Router>
    </div>
  );
}

export default App;