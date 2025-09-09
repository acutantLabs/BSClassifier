from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import select, func, outerjoin


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

from starlette.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
# --- ADD THESE IMPORTS ---
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from sqlalchemy import select # Make sure to add this import at the top of the file


# Import our new modules
import models
import database
# --- END OF IMPORTS TO ADD ---
import os
import logging
import json
import re
import io

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import openai
import requests
from collections import defaultdict, Counter

# OpenAI setup
openai.api_key = os.environ.get('OPENAI_API_KEY')
LOCAL_LLM_ENDPOINT = os.environ.get('LOCAL_LLM_ENDPOINT')

# Create the main app without a prefix
# --- ADD THIS ENTIRE BLOCK OF CODE ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    An asynchronous context manager to handle startup and shutdown events.
    """
    # This part of the code will run ONCE when the server starts.
    logger.info("Server is starting up...")
    
    yield  # The server runs after this 'yield'
    
    # This part of the code will run ONCE when the server shuts down.
    # We no longer need to close the database client here, as each
    # session is managed by the get_db dependency.
    logger.info("Server is shutting down...")


# Now, define the 'app' variable that uses the function above
app = FastAPI(title="Tally Statement Processor", version="1.0.0", lifespan=lifespan)
# --- END OF BLOCK TO ADD ---
# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")
# Add this new function
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code here runs on startup
    yield
    # Code here runs on shutdown


# Pydantic models for Client data
class ClientBase(BaseModel):
    """Defines the common attributes for a client."""
    name: str

class ClientCreate(ClientBase):
    """The model used when CREATING a new client via the API."""
    pass

class ClientUpdate(ClientBase):
    """Model used for updating an existing client's name."""
    pass

class ClientModel(ClientBase):
    """A simple model for a single client's details."""
    id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ClientModelWithCounts(ClientBase):
    """The full model representing a Client as returned from the API."""
    id: str
    created_at: datetime
    updated_at: datetime

    # Add the new count fields
    ledger_rule_count: int
    bank_statement_count: int
    bank_account_count: int

    class Config:
        from_attributes = True

# Pydantic models for LedgerRule data
class LedgerRuleBase(BaseModel):
    """Defines the common attributes for a ledger rule."""
    client_id: str
    ledger_name: str
    regex_pattern: str
    sample_narrations: List[str] = []

class LedgerRule(LedgerRuleBase):
    """The model used when RETURNING a ledger rule from the API."""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        # --- THIS IS THE FIX ---
        from_attributes = True

class BankStatement(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_id: str
    filename: str
    upload_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    column_mapping: Dict[str, Optional[str]]
    statement_format: str  # "single_amount_crdr" or "separate_credit_debit"
    raw_data: List[Dict]
    processed_data: Optional[List[Dict]] = None

class BankStatement(BaseModel):
    """
    Pydantic model for returning a BankStatement object from the API.
    """
    id: str
    client_id: str
    filename: str
    upload_date: datetime
    column_mapping: Dict[str, Optional[str]]
    statement_format: str
    
    # We include the raw_data and processed_data fields
    raw_data: List[Dict]
    processed_data: Optional[List[Dict]] = None

    class Config:
        from_attributes = True # Ensures it's compatible with our SQLAlchemy model

class ColumnMapping(BaseModel):
    date_column: str
    narration_column: str
    amount_column: Optional[str] = None
    credit_column: Optional[str] = None
    debit_column: Optional[str] = None
    balance_column: Optional[str] = None
    crdr_column: Optional[str] = None
    statement_format: str

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    headers: List[str]
    preview_data: List[Dict]
    suggested_mapping: Dict[str, str]

class TransactionCluster(BaseModel):
    cluster_id: str
    narrations: List[str]
    suggested_regex: str
    keyword_patterns: List[str]
    confidence_score: float

class AIRegexRequest(BaseModel):
    narrations: List[str]
    existing_regex: Optional[str] = None
    use_local_llm: bool = False

class TallyVoucherData(BaseModel):
    client_id: str
    bank_ledger_name: str
    transactions: List[Dict]
    excluded_ledgers: List[str] = []
# Pydantic models for BankAccount data
class BankAccountBase(BaseModel):
    """Base model for bank account, used for creation and updates."""
    client_id: str
    bank_name: str
    ledger_name: str
    contra_list: Optional[List[str]] = []
    filter_list: Optional[List[str]] = []

class BankAccountCreate(BankAccountBase):
    """Model used when creating a new bank account."""
    pass

class BankAccountUpdate(BankAccountBase):
    """Model used when updating an existing bank account."""
    pass

class BankAccountModel(BankAccountBase):
    """Full model representing a BankAccount as returned from the API."""
    id: str
    created_at: datetime
    updated_at: datetime
    class Config:
        # --- THIS IS THE FIX ---
        from_attributes = True


# Utility Functions
def detect_date_columns(headers: List[str]) -> List[str]:
    """Detect potential date columns"""
    date_keywords = ['date', 'dt', 'txn_date', 'transaction_date', 'value_date']
    return [h for h in headers if any(keyword in h.lower() for keyword in date_keywords)]

def detect_narration_columns(headers: List[str]) -> List[str]:
    """Detect potential narration/description columns"""
    narration_keywords = ['narration', 'description', 'desc', 'particulars', 'details', 'reference']
    return [h for h in headers if any(keyword in h.lower() for keyword in narration_keywords)]

def detect_amount_columns(headers: List[str]) -> List[str]:
    """Detect potential amount columns"""
    amount_keywords = ['amount', 'amt', 'value', 'debit', 'credit', 'dr', 'cr', 'balance', 'bal']
    return [h for h in headers if any(keyword in h.lower() for keyword in amount_keywords)]

def suggest_column_mapping(headers: List[str]) -> Dict[str, str]:
    """Suggest column mapping based on headers"""
    mapping = {}
    
    date_cols = detect_date_columns(headers)
    if date_cols:
        mapping['date_column'] = date_cols[0]
    
    narration_cols = detect_narration_columns(headers)
    if narration_cols:
        mapping['narration_column'] = narration_cols[0]
    
    amount_cols = detect_amount_columns(headers)
    
    # Check for separate credit/debit columns
    credit_cols = [h for h in headers if 'credit' in h.lower() or h.lower() == 'cr']
    debit_cols = [h for h in headers if 'debit' in h.lower() or h.lower() == 'dr']
    
    if credit_cols and debit_cols:
        mapping['statement_format'] = 'separate_credit_debit'
        mapping['credit_column'] = credit_cols[0]
        mapping['debit_column'] = debit_cols[0]
    else:
        mapping['statement_format'] = 'single_amount_crdr'
        if amount_cols:
            mapping['amount_column'] = amount_cols[0]
        
        # Look for CR/DR indicator column
        crdr_cols = [h for h in headers if 'cr' in h.lower() and 'dr' in h.lower()]
        if crdr_cols:
            mapping['crdr_column'] = crdr_cols[0]
    
    # Balance column
    balance_cols = [h for h in headers if 'balance' in h.lower() or 'bal' in h.lower()]
    if balance_cols:
        mapping['balance_column'] = balance_cols[0]
    
    return mapping

def generate_regex_from_narrations(narrations: List[str]) -> str:
    """Generate regex pattern from similar narrations"""
    if not narrations:
        return ""
    
    # Find common patterns
    word_counts = Counter()
    for narration in narrations:
        # Extract potential keywords (non-numeric, significant words)
        words = re.findall(r'\b[A-Za-z]{3,}\b', narration.upper())
        word_counts.update(words)
    
    # Get most common words
    common_words = [word for word, count in word_counts.most_common(3) if count > 1]
    
    if common_words:
        # Create regex with common words
        pattern_parts = []
        for word in common_words:
            pattern_parts.append(f"(?=.*{re.escape(word)})")
        
        return "".join(pattern_parts) + ".*"
    
    # Fallback: use first few characters
    if narrations:
        first_narration = narrations[0]
        # Extract first significant word
        words = re.findall(r'\b[A-Za-z]{3,}\b', first_narration)
        if words:
            return f".*{re.escape(words[0])}.*"
    
    return ".*" + re.escape(narrations[0][:10]) + ".*"

def cluster_narrations(narrations: List[str], n_clusters: int = None) -> List[TransactionCluster]:
    """Cluster similar narrations using TF-IDF and K-means"""
    if len(narrations) < 2:
        if narrations:
            return [TransactionCluster(
                cluster_id=str(uuid.uuid4()),
                narrations=narrations,
                suggested_regex=generate_regex_from_narrations(narrations),
                keyword_patterns=[],
                confidence_score=0.5
            )]
        return []
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=100,
        ngram_range=(1, 2),
        lowercase=True
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(narrations)
        
        # Determine optimal number of clusters
        if n_clusters is None:
            n_clusters = min(max(2, len(narrations) // 5), 10)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Group narrations by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(narrations[idx])
        
        # Create cluster objects
        cluster_objects = []
        for cluster_id, cluster_narrations in clusters.items():
            suggested_regex = generate_regex_from_narrations(cluster_narrations)
            
            cluster_objects.append(TransactionCluster(
                cluster_id=str(uuid.uuid4()),
                narrations=cluster_narrations,
                suggested_regex=suggested_regex,
                keyword_patterns=[],
                confidence_score=0.7 if len(cluster_narrations) > 1 else 0.5
            ))
        
        return cluster_objects
    
    except Exception as e:
        logging.error(f"Clustering failed: {e}")
        # Fallback: create individual clusters
        return [TransactionCluster(
            cluster_id=str(uuid.uuid4()),
            narrations=[narration],
            suggested_regex=generate_regex_from_narrations([narration]),
            keyword_patterns=[],
            confidence_score=0.3
        ) for narration in narrations]

async def get_ai_improved_regex(narrations: List[str], existing_regex: str = None, use_local_llm: bool = False) -> str:
    """Get AI-improved regex pattern"""
    prompt = f"""
    You are a regex expert. Given these bank transaction narrations, create a precise regex pattern that matches all of them:
    
    Narrations:
    {chr(10).join(f"- {n}" for n in narrations[:10])}
    
    {"Current regex: " + existing_regex if existing_regex else ""}
    
    Requirements:
    1. Match ALL the provided narrations
    2. Be specific enough to avoid false positives
    3. Use word boundaries and case-insensitive patterns
    4. Focus on key identifying words, not variable parts like amounts or dates
    5. Return only the regex pattern, no explanation
    
    Regex pattern:
    """
    
    try:
        if use_local_llm and LOCAL_LLM_ENDPOINT:
            # Use local LLM
            response = requests.post(
                f"{LOCAL_LLM_ENDPOINT}/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": 200,
                    "temperature": 0.1,
                    "stop": ["\n\n"]
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["text"].strip()
        else:
            # Use OpenAI
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a regex expert helping with bank transaction pattern matching."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
    
    except Exception as e:
        logging.error(f"AI regex generation failed: {e}")
        # Fallback to basic regex generation
        return generate_regex_from_narrations(narrations)
    
    return generate_regex_from_narrations(narrations)

# API Routes

@api_router.get("/")
async def root():
    return {"message": "Tally Statement Processor API", "version": "1.0.0"}

# Client Management
@api_router.post("/clients", response_model=ClientModel)
async def create_client(client_data: ClientCreate, db: AsyncSession = Depends(database.get_db)):
    """Create a new client"""
    # Create a new SQLAlchemy model instance from the incoming data
    db_client = models.Client(**client_data.dict())
    
    # Add the new client to the session
    db.add(db_client)
    # Commit the transaction to save it to the database
    await db.commit()
    # Refresh the instance to get the default values from the DB (like created_at)
    await db.refresh(db_client)
    
    return db_client
# --- END OF REPLACEMENT ---

# --- CLEANER REPLACEMENT for get_clients ---
@api_router.get("/clients", response_model=List[ClientModelWithCounts])
async def get_clients(db: AsyncSession = Depends(database.get_db)):
    """Get all clients with their associated counts"""
    # (The complex query logic remains the same)
    ledger_rule_subquery = (
        select(models.LedgerRule.client_id, func.count(models.LedgerRule.id).label("ledger_rule_count"))
        .group_by(models.LedgerRule.client_id)
        .subquery()
    )
    # ... (other subqueries are the same) ...
    bank_statement_subquery = (
        select(models.BankStatement.client_id, func.count(models.BankStatement.id).label("bank_statement_count"))
        .group_by(models.BankStatement.client_id)
        .subquery()
    )
    bank_account_subquery = (
        select(models.BankAccount.client_id, func.count(models.BankAccount.id).label("bank_account_count"))
        .group_by(models.BankAccount.client_id)
        .subquery()
    )

    query = (
        select(
            models.Client,
            func.coalesce(ledger_rule_subquery.c.ledger_rule_count, 0).label("ledger_rule_count"),
            func.coalesce(bank_statement_subquery.c.bank_statement_count, 0).label("bank_statement_count"),
            func.coalesce(bank_account_subquery.c.bank_account_count, 0).label("bank_account_count"),
        )
        .outerjoin(ledger_rule_subquery, models.Client.id == ledger_rule_subquery.c.client_id)
        .outerjoin(bank_statement_subquery, models.Client.id == bank_statement_subquery.c.client_id)
        .outerjoin(bank_account_subquery, models.Client.id == bank_account_subquery.c.client_id)
        .order_by(models.Client.name)
    )
    
    result = await db.execute(query)
    
    # Process the results more elegantly
    clients_with_counts = []
    for row in result.mappings(): # .mappings() gives us dict-like rows
        client_data = row['Client'].__dict__
        client_data.update({
            "ledger_rule_count": row['ledger_rule_count'],
            "bank_statement_count": row['bank_statement_count'],
            "bank_account_count": row['bank_account_count'],
        })
        clients_with_counts.append(ClientModelWithCounts.model_validate(client_data))

    return clients_with_counts
# --- END OF CLEANER REPLACEMENT ---

# --- REPLACEMENT for get_client ---
@api_router.get("/clients/{client_id}", response_model=ClientModel)
async def get_client(client_id: str, db: AsyncSession = Depends(database.get_db)):
    """Get client by ID"""
    # Get a single client by its primary key (ID)
    client = await db.get(models.Client, client_id)
    
    # If the client doesn't exist, db.get() returns None
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
        
    return client
# --- END OF REPLACEMENT ---
# --- ADD THIS MISSING ENDPOINT ---
@api_router.put("/clients/{client_id}", response_model=ClientModel)
async def update_client(client_id: str, client_data: ClientUpdate, db: AsyncSession = Depends(database.get_db)):
    """Update an existing client's name"""
    db_client = await db.get(models.Client, client_id)
    if not db_client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Update the client object with the new data
    update_data = client_data.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_client, key, value)
        
    await db.commit()
    await db.refresh(db_client)
    return db_client
# --- END OF MISSING ENDPOINT ---
# File Upload and Processing
# --- REPLACEMENT for upload_statement ---
@api_router.post("/upload-statement", response_model=FileUploadResponse)
async def upload_statement(file: UploadFile = File(...), db: AsyncSession = Depends(database.get_db)):
    """Upload and analyze bank statement"""
    if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
        raise HTTPException(status_code=400, detail="Only Excel and CSV files are supported")
    
    try:
        content = await file.read()
        # --- START OF REPLACEMENT ---
        if file.filename.endswith('.csv'):
            # For CSV files, use read_csv with the index_col parameter
            df = pd.read_csv(io.BytesIO(content), index_col=0)
        else:
            # For Excel files, use read_excel with the index_col parameter
            df = pd.read_excel(io.BytesIO(content), index_col=0)
        # --- END OF REPLACEMENT ---
        
        df = df.dropna(how='all').reset_index(drop=True)
        headers = df.columns.tolist()
        preview_data = df.head(10).fillna('').to_dict('records')
        suggested_mapping = suggest_column_mapping(headers)
        
        # Store file data temporarily in the new table
        temp_file = models.TempFile(
            filename=file.filename,
            headers=headers,
            raw_data=df.fillna('').to_dict('records')
        )
        
        db.add(temp_file)
        await db.commit()
        await db.refresh(temp_file)
        
        return FileUploadResponse(
            file_id=temp_file.id, # Use the ID from the database
            filename=file.filename,
            headers=headers,
            preview_data=preview_data,
            suggested_mapping=suggested_mapping
        )
    
    except Exception as e:
        logging.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
# --- END OF REPLACEMENT ---

# --- FINAL, CORRECTED VERSION of confirm_column_mapping ---
@api_router.post("/confirm-mapping/{file_id}")
async def confirm_column_mapping(file_id: str, mapping: ColumnMapping, client_id: str = Query(...), db: AsyncSession = Depends(database.get_db)):
    """Confirm column mapping and process statement"""
    temp_file = await db.get(models.TempFile, file_id)
    if not temp_file:
        raise HTTPException(status_code=404, detail="File not found or has expired")
    
    try:
        # Create a permanent BankStatement record
        statement = models.BankStatement(
            client_id=client_id,
            filename=temp_file.filename,
            column_mapping=mapping.dict(),
            statement_format=mapping.statement_format,
            raw_data=temp_file.raw_data
        )
        
        db.add(statement)
        
        # --- THE CORRECT CODE ---
        await db.delete(temp_file)
        # --- END OF CORRECT CODE ---
        
        await db.commit()
        await db.refresh(statement)
        
        return {"message": "Statement processed successfully", "statement_id": statement.id}
    
    except Exception as e:
        await db.rollback()
        logging.error(f"Mapping confirmation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing mapping: {str(e)}")
# --- END OF REPLACEMENT ---


# Regex Pattern Management
# --- REPLACEMENT for create_regex_pattern ---
@api_router.post("/ledger-rules", response_model=LedgerRule)
async def create_ledger_rule(rule_data: LedgerRuleBase, db: AsyncSession = Depends(database.get_db)):
        """Create a new ledger rule"""
        db_rule = models.LedgerRule(**rule_data.dict())
        
        db.add(db_rule)
        await db.commit()
        await db.refresh(db_rule)
        
        return db_rule
# --- END OF REPLACEMENT ---

# --- REPLACEMENT for get_client_regex_patterns ---
@api_router.get("/ledger-rules/{client_id}", response_model=List[LedgerRule])
async def get_client_ledger_rules(client_id: str, db: AsyncSession = Depends(database.get_db)):
        """Get all ledger rules for a client"""
        query = select(models.LedgerRule).where(models.LedgerRule.client_id == client_id)
        
        result = await db.execute(query)
        rules = result.scalars().all()
        
        return rules
# --- END OF REPLACEMENT ---

# Transaction Classification

# --- REPLACEMENT for classify_transactions ---
@api_router.post("/classify-transactions/{statement_id}")
async def classify_transactions(statement_id: str, db: AsyncSession = Depends(database.get_db)):
    """Classify transactions using existing regex patterns"""
    statement = await db.get(models.BankStatement, statement_id)
    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")
    
    # Get client's regex patterns using the function we just refactored
    query = select(models.LedgerRule).where(models.LedgerRule.client_id == statement.client_id)
    result = await db.execute(query)
    patterns = result.scalars().all()
    
    classified_transactions = []
    unmatched_narrations = []
    
    # Ensure raw_data is a list of dicts
    raw_data = statement.raw_data if isinstance(statement.raw_data, list) else []
    for transaction in raw_data:
        narration = str(transaction.get(statement.column_mapping.get("narration_column"), ""))
        
        matched = False
        for pattern in patterns:
            try:
                if re.search(pattern.regex_pattern, narration, re.IGNORECASE):
                    # Create a copy to avoid modifying the original data
                    classified_transaction = transaction.copy()
                    classified_transaction["matched_ledger"] = pattern.ledger_name
                    classified_transaction["matched_pattern_id"] = pattern.id
                    classified_transactions.append(classified_transaction)
                    matched = True
                    break
            except re.error:
                continue # Skip invalid regex patterns
        
        if not matched:
            classified_transaction = transaction.copy()
            classified_transaction["matched_ledger"] = "Suspense A/c"
            classified_transactions.append(classified_transaction)
            if narration: # Only add non-empty narrations to the cluster list
                unmatched_narrations.append(narration)
    
    # Cluster unmatched narrations
    clusters = cluster_narrations(unmatched_narrations) if unmatched_narrations else []
    
    # Update statement with processed data
    statement.processed_data = classified_transactions
    await db.commit()
    
    return {
        "classified_transactions": classified_transactions,
        "unmatched_clusters": [cluster.dict() for cluster in clusters],
        "total_transactions": len(classified_transactions),
        "matched_transactions": len([t for t in classified_transactions if t["matched_ledger"] != "Suspense A/c"]),
        "unmatched_transactions": len(unmatched_narrations)
    }
# --- END OF REPLACEMENT ---
@api_router.post("/ai-improve-regex")
async def ai_improve_regex(request: AIRegexRequest):
    """Use AI to improve regex patterns"""
    improved_regex = await get_ai_improved_regex(
        request.narrations,
        request.existing_regex,
        request.use_local_llm
    )
    
    return {"improved_regex": improved_regex}

# Mock Tally API endpoints
@api_router.post("/tally/create-ledgers")
async def create_tally_ledgers(ledger_names: List[str]):
    """Mock endpoint for creating ledgers in Tally"""
    # This would integrate with actual Tally API
    return {
        "message": f"Successfully created {len(ledger_names)} ledgers in Tally",
        "ledgers": ledger_names,
        "status": "success"
    }

# Bank Account Management
@api_router.post("/bank-accounts", response_model=BankAccountModel)
async def create_bank_account(account_data: BankAccountCreate, db: AsyncSession = Depends(database.get_db)):
    """Create a new bank account for a client"""
    # Check if the client exists
    client = await db.get(models.Client, account_data.client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
        
    db_account = models.BankAccount(**account_data.dict())
    db.add(db_account)
    await db.commit()
    await db.refresh(db_account)
    return db_account

@api_router.get("/clients/{client_id}/bank-accounts", response_model=List[BankAccountModel])
async def get_client_bank_accounts(client_id: str, db: AsyncSession = Depends(database.get_db)):
    """Get all bank accounts for a specific client"""
    query = select(models.BankAccount).where(models.BankAccount.client_id == client_id)
    result = await db.execute(query)
    accounts = result.scalars().all()
    return accounts

@api_router.post("/tally/generate-vouchers")
async def generate_tally_vouchers(voucher_data: TallyVoucherData):
    """Generate Tally import files (Receipt, Payment, Contra vouchers)"""
    try:
        transactions = voucher_data.transactions
        bank_ledger = voucher_data.bank_ledger_name
        
        receipt_vouchers = []
        payment_vouchers = []
        contra_vouchers = []
        
        for transaction in transactions:
            ledger = transaction.get("matched_ledger", "Suspense A/c")
            amount = float(transaction.get("amount", 0))
            narration = transaction.get("narration", "")
            date = transaction.get("date", "")
            
            # Skip excluded ledgers
            if ledger in voucher_data.excluded_ledgers:
                continue
            
            # Determine voucher type based on amount and transaction type
            if amount > 0:  # Credit to bank (Receipt)
                receipt_vouchers.append({
                    "Date": date,
                    "Voucher Type": "Receipt",
                    "Ledger Name": ledger,
                    "Amount": amount,
                    "Narration": narration,
                    "Bank Ledger": bank_ledger
                })
            else:  # Debit from bank (Payment)
                payment_vouchers.append({
                    "Date": date,
                    "Voucher Type": "Payment", 
                    "Ledger Name": ledger,
                    "Amount": abs(amount),
                    "Narration": narration,
                    "Bank Ledger": bank_ledger
                })
        
        return {
            "receipt_vouchers": receipt_vouchers,
            "payment_vouchers": payment_vouchers,
            "contra_vouchers": contra_vouchers,
            "summary": {
                "total_receipts": len(receipt_vouchers),
                "total_payments": len(payment_vouchers),
                "total_contras": len(contra_vouchers)
            }
        }
    
    except Exception as e:
        logging.error(f"Voucher generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating vouchers: {str(e)}")

# ADD this to server.py
@api_router.get("/statements/{statement_id}", response_model=BankStatement)
async def get_statement(statement_id: str, db: AsyncSession = Depends(database.get_db)):
    statement = await db.get(models.BankStatement, statement_id)
    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")
    return statement
#Include any new endpoints above this line
# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
