import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useNavigate, useParams } from 'react-router-dom';
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
// --- END OF REPLACEMENT ---
// --- END OF ADDITION ---

// --- ADD THIS ENTIRE COMPONENT ---
const DownloadVoucherModal = ({ isOpen, onClose, data }) => {
  const [filename, setFilename] = useState('');
  const [includeReceipts, setIncludeReceipts] = useState(true);
  const [includePayments, setIncludePayments] = useState(true);
  const [includeContras, setIncludeContras] = useState(true);

  useEffect(() => {
    if (data?.suggested_filename) {
      setFilename(data.suggested_filename);
    }
  }, [data]);

  const handleDownload = () => {
    let combinedVouchers = [];
    if (includeReceipts) {
        combinedVouchers.push(...data.receipt_vouchers.map(t => ({...t, voucher_type: 'Receipt'})));
    }
    if (includePayments) {
        combinedVouchers.push(...data.payment_vouchers.map(t => ({...t, voucher_type: 'Payment'})));
    }
    if (includeContras) {
        combinedVouchers.push(...data.contra_vouchers.map(t => ({...t, voucher_type: 'Contra'})));
    }

    if (combinedVouchers.length === 0) {
      toast.error("No vouchers selected to download.");
      return;
    }

    // Use our new helper function to generate the CSV content
    const csvContent = generateTallyCSV(combinedVouchers, data.bank_ledger_name, data.statement_details);
    
    // Standard browser download logic
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    onClose();
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
                <div className="flex items-center space-x-2"><Switch id="receipts" checked={includeReceipts} onCheckedChange={setIncludeReceipts} /><Label htmlFor="receipts">Receipts ({data.receipt_vouchers.length})</Label></div>
                <div className="flex items-center space-x-2"><Switch id="payments" checked={includePayments} onCheckedChange={setIncludePayments} /><Label htmlFor="payments">Payments ({data.payment_vouchers.length})</Label></div>
                <div className="flex items-center space-x-2"><Switch id="contras" checked={includeContras} onCheckedChange={setIncludeContras} /><Label htmlFor="contras">Contras ({data.contra_vouchers.length})</Label></div>
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="filename">Filename</Label>
            <Input id="filename" value={filename} onChange={(e) => setFilename(e.target.value)} />
          </div>
        </div>
        <div className="flex justify-end gap-4 pt-4 border-t">
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={handleDownload}><Download className="w-4 h-4 mr-2" />Download CSV</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
// --- END OF ADDITION ---```

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
const ClientDetailsPage = () => {
  const { clientId } = useParams();
  const [client, setClient] = useState(null);
  const [bankAccounts, setBankAccounts] = useState([]);
  const [statements, setStatements] = useState([]); // State for statements
  const [statementToDelete, setStatementToDelete] = useState(null); // State for delete confirmation
  const [showAddAccountModal, setShowAddAccountModal] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const clientRes = await axios.get(`${API}/clients/${clientId}`);
        setClient(clientRes.data);

        const accountsRes = await axios.get(`${API}/clients/${clientId}/bank-accounts`);
        setBankAccounts(accountsRes.data);
        
        // Fetch statements for the client
        const statementsRes = await axios.get(`${API}/clients/${clientId}/statements`);
        setStatements(statementsRes.data);

      } catch (error) {
        toast.error("Failed to fetch client details.");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [clientId]);

  const handleAccountCreated = (newAccount) => {
    setBankAccounts(prev => [...prev, newAccount]);
    setShowAddAccountModal(false);
  };

  const handleDeleteStatement = async () => {
    if (!statementToDelete) return;
    try {
      await axios.delete(`${API}/statements/${statementToDelete.id}`);
      toast.success(`Statement "${statementToDelete.filename}" deleted.`);
      // Update UI instantly
      setStatements(prev => prev.filter(s => s.id !== statementToDelete.id));
      setStatementToDelete(null); // Close the dialog
    } catch (error) {
      toast.error("Failed to delete statement.");
    }
  };

  if (loading) {
    return <div>Loading client details...</div>;
  }

  if (!client) {
    return <div>Client not found.</div>;
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">{client.name}</h1>
        <p className="text-slate-600 mt-2">Manage client statements and bank accounts.</p>
      </div>
      
      {/* Processed Statements Card */}
      <Card className="border-0 shadow-sm">
        <CardHeader>
          <CardTitle>Processed Statements</CardTitle>
          <CardDescription>Review or delete processed statements for this client.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {statements.length > 0 ? (
              statements.map(stmt => {
  const isCompleted = stmt.total_transactions > 0 && stmt.matched_transactions === stmt.total_transactions;
  return (
    <div key={stmt.id} className="p-4 border rounded-lg flex justify-between items-center bg-slate-50/50">
      <div className="flex flex-col">
        <div className="flex items-center gap-3">
          <p className="font-semibold text-slate-800">{stmt.filename}</p>
          <Badge 
            variant={isCompleted ? "default" : "outline"}
            className={isCompleted 
              ? "bg-green-100 text-green-800 border-green-200" 
              : "border-amber-400 text-amber-700"}
          >
            {isCompleted ? "Completed" : "Needs Review"}
          </Badge>
        </div>
        <div className="text-sm text-slate-500 mt-1 flex flex-wrap items-center gap-x-4 gap-y-1">
          {stmt.bank_ledger_name && (
            <span>Account: <span className="font-medium text-slate-600">{stmt.bank_ledger_name}</span></span>
          )}
          {stmt.statement_period && (
            <span>Period: <span className="font-medium text-slate-600">{stmt.statement_period}</span></span>
          )}
          <span>Matched: <span className="font-medium text-slate-600">{stmt.matched_transactions} / {stmt.total_transactions}</span></span>
        </div>
      </div>

      <div className="flex gap-2">
        <Link to={`/statements/${stmt.id}`}>
          <Button variant="outline" size="sm">
            <Eye className="w-4 h-4 mr-1" />
            View
          </Button>
        </Link>
        <Button variant="destructive" size="sm" onClick={() => setStatementToDelete(stmt)}>
          <Trash2 className="w-4 h-4 mr-1" />
          Delete
        </Button>
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
      
      {/* Bank Accounts Card */}
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
                  <Button variant="outline" size="sm">Edit</Button>
                </div>
              ))
            ) : (
              <p className="text-center text-slate-500 py-4">No bank accounts added yet.</p>
            )}
          </div>
        </CardContent>
      </Card>
      
      {/* Add/Edit Bank Account Modal */}
      <AddBankAccountModal
        isOpen={showAddAccountModal}
        onClose={() => setShowAddAccountModal(false)}
        clientId={clientId}
        onSuccess={handleAccountCreated}
      />

      {/* Delete Confirmation Dialog */}
      <Dialog open={!!statementToDelete} onOpenChange={(isOpen) => !isOpen && setStatementToDelete(null)}>
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
    </div>
  );
};
// --- END OF REPLACEMENT ---


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


// --- DEFINITIVE REPLACEMENT for ClusterCard ---
// --- DEFINITIVE REPLACEMENT for ClusterCard ---
// In App.js

// --- REPLACE THE ENTIRE ClusterCard COMPONENT ---
// In App.js

// --- REPLACE THE ENTIRE ClusterCard COMPONENT ---
const ClusterCard = ({ cluster, clientId, onRuleCreated, otherNarrations, onDetach, narrationColumnName, onMarkAsSuspense }) => {
  const [editableRegex, setEditableRegex] = useState(cluster.suggested_regex || '');
  const [ledgerName, setLedgerName] = useState('');
  const [loading, setLoading] = useState(false);
  const [validation, setValidation] = useState({ 
    matchStatus: 'none',
    matchCount: 0,
    highlightedNarrations: []
  });
  const [falsePositiveCount, setFalsePositiveCount] = useState(0);

  useEffect(() => {
    if (!narrationColumnName) return;

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

    const newHighlightedResults = cluster.transactions.map(t => getHighlightedHtml(editableRegex, t[narrationColumnName]));
    const currentMatchCount = newHighlightedResults.filter(result => result.matched).length;

    let status = 'none';
    if (currentMatchCount > 0 && currentMatchCount === cluster.transactions.length) status = 'all';
    else if (currentMatchCount > 0) status = 'partial';

    setValidation({
      matchStatus: status,
      matchCount: currentMatchCount,
      highlightedNarrations: newHighlightedResults.map(result => result.html)
    });

  }, [editableRegex, cluster.transactions, otherNarrations, narrationColumnName]);

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
        sample_narrations: cluster.transactions.map(t => t[narrationColumnName]),
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
              case 'all': return <span className="text-sm font-bold text-green-600 flex items-center"><CheckCircle2 className="w-4 h-4 mr-1"/> All Match</span>;
              case 'partial': return <span className="text-sm font-bold text-yellow-600">{`Partial Match [${validation.matchCount}/${cluster.transactions.length}]`}</span>;
              default: return <span className="text-sm font-bold text-red-600 flex items-center"><AlertCircle className="w-4 h-4 mr-1"/> No Match</span>;
            }
          })()}
        </div>
      </div>
      
      <Textarea className="font-mono text-xs bg-white" value={editableRegex} onChange={(e) => setEditableRegex(e.target.value)} />

      {falsePositiveCount > 0 && (
        <div className="p-3 my-2 bg-yellow-100 border-l-4 border-yellow-400 text-yellow-800 text-sm rounded-r-md flex items-center gap-3">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <div><span className="font-bold">Warning:</span> This pattern incorrectly matches <span className="font-extrabold">{falsePositiveCount}</span> other transaction(s).</div>
        </div>
      )}

      <p className="text-sm font-semibold">Sample Transactions ({cluster.transactions.length} items):</p>
      
      <ScrollArea className="h-32 p-2 border rounded-md bg-white">
        <div className="text-xs space-y-2">
          {cluster.transactions.map((transaction, i) => {
            const rawCrDr = transaction['CR/DR'] || '';
            const isCredit = rawCrDr.trim().replace(/\./g, '').toUpperCase() === 'CR';
            
            return (
              <div key={i} className="flex items-center justify-between gap-1 p-1 group">
                <div className="flex-shrink-0 flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
                   <Button variant="ghost" size="icon" className="w-6 h-6" onClick={() => onMarkAsSuspense([transaction])} title="Mark as Suspense">
                      <HelpCircle className="w-3 h-3 text-slate-500" />
                   </Button>
                   <Button variant="ghost" size="icon" className="w-6 h-6" onClick={() => onDetach(transaction, cluster.cluster_id)} title="Detach from cluster">
                      <X className="w-3 h-3 text-slate-500" />
                   </Button>
                </div>

                <div className="truncate flex-grow" dangerouslySetInnerHTML={{ __html: validation.highlightedNarrations[i] }} />
                
                <div className="flex items-center gap-2 flex-shrink-0">
                  <Badge className={`h-5 font-semibold ${isCredit ? 'bg-green-100 text-green-800 border-green-200' : 'bg-red-100 text-red-800 border-red-200'}`}>
                    {isCredit ? 'Credit' : 'Debit'}
                  </Badge>
                  <Badge variant="outline" className="font-mono">{formatCurrency(transaction['Amount (INR)'])}</Badge>
                </div>
              </div>
            );
          })}
        </div>
      </ScrollArea>
      
      <div className="flex items-center gap-4 pt-2">
        <Input placeholder="Enter Ledger Name..." value={ledgerName} onChange={(e) => setLedgerName(e.target.value)} />
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
const ClassifiedTransactionsTable = ({ transactions, onFlagAsIncorrect }) => {
  const transactionsToShow = transactions.filter(
    t => t.matched_ledger !== "Suspense" || t.user_confirmed == true
  );

  if (transactionsToShow.length === 0) {
    return <p className="text-center text-slate-500 py-4">No transactions were matched to existing rules.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm table-fixed"> {/* Use table-fixed for precise column widths */}
        <thead>
          <tr className="border-b">
            <th className="p-2 text-left font-semibold w-40">Date</th> {/* Added fixed width */}
            <th className="p-2 text-left font-semibold">Description</th> {/* Takes remaining space */}
            <th className="p-2 text-right font-semibold w-32">Amount</th> {/* Added fixed width */}
            <th className="p-2 text-left font-semibold w-24">CR/DR</th> {/* Added fixed width */}
            <th className="p-2 text-left font-semibold w-48">Matched Ledger</th> {/* Shrunk width */}
            <th className="p-2 text-center font-semibold w-24">Actions</th> {/* Added fixed width */}
          </tr>
        </thead>
        <tbody>
          {transactionsToShow.map(transaction => (
            <tr key={transaction.Narration + transaction.Amount + Math.random()} className="border-b hover:bg-slate-50">
              <td className="p-2 whitespace-nowrap">{transaction.Date}</td>
              
              {/* --- THIS IS THE NEW DESCRIPTION CELL --- */}
              <td className="p-2 max-w-sm ">{transaction.Narration}</td>

              {/* --- END OF NEW DESCRIPTION CELL --- */}

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
// Statement Classification Page Component
const StatementDetailsPage = () => {
  const { statementId } = useParams();
  const [statement, setStatement] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);
  const [loading, setLoading] = useState(true); // Renamed from 'loading'
  const [classifying, setClassifying] = useState(false);
  const [isVoucherModalOpen, setIsVoucherModalOpen] = useState(false);
  const [voucherData, setVoucherData] = useState(null);
  const [detachedTransactions, setDetachedTransactions] = useState([]);

  
  // --- SIMPLIFIED runClassification ---
    const runClassification = useCallback(async (isForced = false) => {
      if (isForced) {
      setDetachedTransactions([]);
    }
    setClassifying(true);
    try {
      // Add the force_reclassify parameter if this is a forced action.
      const url = isForced 
        ? `${API}/classify-transactions/${statementId}?force_reclassify=true`
        : `${API}/classify-transactions/${statementId}`;
      
      const response = await axios.post(url);
      setClassificationResult(response.data);
    } catch (error) {
      toast.error("Failed to run classification.");
    } finally {
      setClassifying(false);
    }
  }, [statementId]);

  // --- SIMPLIFIED handleFlagAsIncorrect ---
    // --- ADD THIS NEW HELPER FUNCTION FIRST ---
  const saveClassificationState = async (newClassificationResult) => {
    if (!newClassificationResult) return;
    try {
      await axios.post(`${API}/statements/${statementId}/update-transactions`, {
        processed_data: newClassificationResult.classified_transactions,
      });
      // Optional: show a subtle success toast, or none at all for background saves
      // toast.success("Progress saved!");
    } catch (error) {
      toast.error("Failed to save progress. Please check your connection.");
    }
  };

  const handleMarkAsSuspense = (transactionsToMark) => {
    const narrationColumn = statement?.column_mapping?.narration_column;
    if (!narrationColumn) return;
    const narrationsToMark = new Set(transactionsToMark.map(t => t[narrationColumn]));

    let newResult; // To hold the new state
    setClassificationResult(prevResult => {
      if (!prevResult) return null;

      const newClassified = prevResult.classified_transactions.map(t => {
        if (narrationsToMark.has(t.Narration)) {
          return { ...t, user_confirmed: true };
        }
        return t;
      });

      const newClusters = prevResult.unmatched_clusters.map(cluster => ({
        ...cluster,
        transactions: cluster.transactions.filter(t => !narrationsToMark.has(t[narrationColumn]))
      })).filter(cluster => cluster.transactions.length > 0);

      newResult = { // Assign the new state to our variable
        ...prevResult,
        classified_transactions: newClassified,
        unmatched_clusters: newClusters,
      };
      return newResult;
    });
    
    // Call the save function immediately after the state update is queued
    setTimeout(() => saveClassificationState(newResult), 0);
    toast.info(`${transactionsToMark.length} transaction(s) marked as Suspense.`);
  };

  // --- FIND AND REPLACE ONLY the handleFlagAsIncorrect function ---
  const handleFlagAsIncorrect = (transactionToFlag) => {
    const narrationColumn = statement?.column_mapping?.narration_column;
    if (!narrationColumn) return;

    let newResult;
    setClassificationResult(prevResult => {
      if (!prevResult) return null;

      // Step 1: Modify the state, DON'T delete.
      // Find the transaction and reset its status to a "pending" suspense item.
      const newClassified = prevResult.classified_transactions.map(t => {
        if (t.Narration === transactionToFlag.Narration) {
          const resetTransaction = { ...t };
          resetTransaction.matched_ledger = 'Suspense';
          delete resetTransaction.user_confirmed;
          delete resetTransaction.matched_pattern_id;
          return resetTransaction;
        }
        return t;
      });

      // Step 2: Re-build the cluster view based on the new reality.
      // Find the transaction we just reset.
      const transactionToReCluster = newClassified.find(
        t => t.Narration === transactionToFlag.Narration
      );

      // Find its original raw data for the cluster card.
      const originalTransaction = (statement.raw_data || []).find(
        raw => raw[narrationColumn] === transactionToReCluster.Narration
      );

      // If we found it, create a new cluster for it.
      let newClusters = prevResult.unmatched_clusters;
      if (originalTransaction) {
        const newCluster = {
          cluster_id: `flagged-${Math.random()}`,
          transactions: [originalTransaction],
          suggested_regex: generateSimpleRegex(originalTransaction[narrationColumn])

        };
        newClusters = [newCluster, ...prevResult.unmatched_clusters];
      }

      newResult = {
        ...prevResult,
        classified_transactions: newClassified,
        unmatched_clusters: newClusters,
      };
      return newResult;
    });

    // Step 3: Save the FULL, modified list back to the database.
    setTimeout(() => saveClassificationState(newResult), 0);
    toast.info("Transaction moved back to 'Unmatched' for review.");
  };
// --- END OF REPLACEMENT ---

  // --- ADD THIS ENTIRE FUNCTION ---
  const handleGenerateVouchers = async () => {
    setClassifying(true); // Reuse the existing loading state for user feedback
    try {
      // Step A: Commit the current state to the database.
      await axios.post(`${API}/statements/${statementId}/update-transactions`, {
        processed_data: classificationResult.classified_transactions,
      });
      toast.success("Current progress saved!");

      // Step B: Fetch the formatted voucher data from the backend.
      const response = await axios.post(`${API}/generate-vouchers/${statementId}`);
      setVoucherData(response.data);
      setIsVoucherModalOpen(true); // Open the modal on success
      
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to generate vouchers. Please try again.");
    } finally {
      setClassifying(false);
    }
  };
  const handleDetachTransaction = (transactionToDetach, sourceClusterId) => {
    setClassificationResult(prevResult => {
      // Guard against cases where there's no data
      if (!prevResult) return null;

      const newClusters = prevResult.unmatched_clusters.map(cluster => {
        // Find the cluster that the transaction is coming from
        if (cluster.cluster_id === sourceClusterId) {
          // Return a new cluster object with the transaction filtered out
          return {
            ...cluster,
            // Replace the incorrect id() call with a direct reference check.
            transactions: cluster.transactions.filter(
              t => t !== transactionToDetach 
            ),
          };
        }
        // Leave other clusters unchanged
        return cluster;
      }).filter(cluster => cluster.transactions.length > 0); // Remove any clusters that are now empty

      // Update the main classification result state
      return {
        ...prevResult,
        unmatched_clusters: newClusters,
      };
    });

    // Add the detached transaction to our "holding pen" state
    setDetachedTransactions(prevDetached => [...prevDetached, transactionToDetach]);

    toast.info("Transaction detached from cluster.");
  };
  // --- END OF NEW HANDLER ---
  useEffect(() => {
  const fetchStatement = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/statements/${statementId}`);
      setStatement(response.data);
    } catch (error) {
      toast.error("Failed to fetch statement details.");
      setStatement(null); // Clear statement on error
    } finally {
      setLoading(false);
    }
  };
  fetchStatement();
  }, [statementId]); // This effect only runs when the statementId changes

  // Effect 2: Run the classification ONLY when the statement has been successfully fetched.
  useEffect(() => {
  if (statement) {
    // We wrap runClassification in an async IIFE to handle the final setLoading
    (async () => {
      await runClassification();
      setLoading(false); // Turn off the main loading indicator AFTER classification is done
    })();
    }
  }, [statement, runClassification]); // Dependency array is correct
  // --- ADD THIS LOGIC inside StatementDetailsPage, before the return statement ---
  const otherNarrations = useMemo(() => {
        if (!classificationResult) return [];
        // Get all narrations from transactions that were already successfully matched.
        return classificationResult.classified_transactions
            .filter(t => t.matched_ledger !== 'Suspense')
            .map(t => t.Narration);
             console.log("Generated 'otherNarrations':", narrations); // <-- ADD THIS LOG

    }, [classificationResult]);

  if (loading || !classificationResult) {
    return <div>Loading classification results...</div>;
  }
  
  return (
    <div className="space-y-8">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Classification Review</h1>
          <p className="text-slate-600 mt-2">
            {classificationResult.matched_transactions} of {classificationResult.total_transactions} transactions matched.
          </p>
        </div>
         <div className="flex items-center gap-2">
          <Button onClick={handleGenerateVouchers} disabled={classifying} className="bg-blue-600 hover:bg-blue-700">
            <Download className={`w-4 h-4 mr-2 ${classifying ? 'animate-spin' : ''}`} />
            Download Vouchers
          </Button>
          {/* --- END OF ADDITION --- */}
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
                onRuleCreated={runClassification}
                otherNarrations={otherNarrations} // Pass the new prop down
                onDetach={handleDetachTransaction}
                onMarkAsSuspense={handleMarkAsSuspense} // <-- ADD THIS
                narrationColumnName={statement?.column_mapping?.narration_column}
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
      ? generateSimpleRegex(detachedTransactions[0][statement?.column_mapping?.narration_column]) 
      : ''

                }}
                clientId={statement.client_id}
                onRuleCreated={runClassification} // You can still create a rule from a single detached item
                otherNarrations={otherNarrations}
                onDetach={() => {}} // Detaching from this group does nothing
                onMarkAsSuspense={handleMarkAsSuspense} // <-- ADD THIS
                narrationColumnName={statement?.column_mapping?.narration_column}

            />
          </CardContent>
        </Card>
      )}
      {/* --- END: ADD THIS ENTIRE JSX BLOCK --- */}
      {/* Classified Transactions Table Section */}
      <Card className="border-0 shadow-sm">
        <CardHeader>
          <CardTitle>Classified Transactions</CardTitle>
        </CardHeader>
        <CardContent>
        <ClassifiedTransactionsTable
          transactions={classificationResult.classified_transactions}
          onFlagAsIncorrect={handleFlagAsIncorrect}
        />
        </CardContent>
      </Card>
      {/* Voucher Modal */}{/* --- ADD THIS COMPONENT AT THE END --- */}
      <DownloadVoucherModal 
        isOpen={isVoucherModalOpen}
        onClose={() => setIsVoucherModalOpen(false)}
        data={voucherData}
      />
      {/* --- END OF ADDITION --- */}
    </div>
  );
};

// --- END OF NEW COMPONENT ---
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
            <span className="text-xl font-bold text-slate-900">ABS Processor</span>
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
const RuleEditModal = ({ isOpen, onClose, rule, clientId, onSuccess }) => {
  const [ledgerName, setLedgerName] = useState('');
  const [regexPattern, setRegexPattern] = useState('');
  const [loading, setLoading] = useState(false);
  const [validationResults, setValidationResults] = useState([]);
  const isEditing = !!rule;

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

  // Pre-fill form when a rule is passed in
  useEffect(() => {
    if (rule) {
      setLedgerName(rule.ledger_name);
      setRegexPattern(rule.regex_pattern);
    } else {
      setLedgerName('');
      setRegexPattern('');
    }
  }, [rule]);

  // Live validation useEffect
  useEffect(() => {
    if (isEditing && rule?.sample_narrations) {
      const results = rule.sample_narrations.map(narration => {
        const { html, matches } = getHighlightedHtml(regexPattern, narration);
        return { narration, html, matches };
      });
      setValidationResults(results);
    } else {
      setValidationResults([]);
    }
  }, [regexPattern, rule, isEditing]);

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

          {isEditing && validationResults.length > 0 && (
            <>
              <Separator />
              <div className="space-y-2">
                <Label>Live Test Against Sample Narrations</Label>
                <ScrollArea className="h-40 p-2 border rounded-md bg-slate-50">
                  <div className="text-xs space-y-2">
                    {validationResults.map((result, index) => (
                      <div key={index} className="flex items-start gap-2">
                        <div className="flex-shrink-0 pt-0.5">
                          {result.matches ? (
                            <Check className="w-4 h-4 text-green-600" />
                          ) : (
                            <X className="w-4 h-4 text-red-600" />
                          )}
                        </div>
                        <div className="flex-grow" dangerouslySetInnerHTML={{ __html: result.html }} />
                      </div>
                    ))}
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
            <Route path="/clients/:clientId" element={<ClientDetailsPage />} />
            <Route path="/statements/:statementId" element={<StatementDetailsPage />} />
          </Routes>
        </main>
      </Router>
    </div>
  );
}

export default App;