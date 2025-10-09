from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import select, func, outerjoin
from sqlalchemy.orm import selectinload # <-- ADD THIS
from typing import List, Dict, Optional, Any, Union, Tuple

import openpyxl
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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
from pandas import NaT
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

class UpdateTransactionsRequest(BaseModel):
    processed_data: List[Dict[str, Any]]

class ClientUpdate(ClientBase):
    """Model used for updating an existing client's name."""
    pass

class RuleStat(BaseModel):
    """Defines the health statistics for a single rule."""
    total_matches: int
    last_used: Optional[str] = None

class RuleStatsResponse(BaseModel):
    """The response model for the rule stats endpoint."""
    stats: Dict[str, RuleStat]

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

class StatementMetadata(BaseModel):
    """A summary model for a bank statement, used for listings."""
    id: str
    filename: str
    upload_date: datetime
    total_transactions: int
    matched_transactions: int
    bank_ledger_name: Optional[str] = None
    statement_period: Optional[str] = None
    status: str  # New field for "Completed" or "Needs Review"
    completion_percentage: float # New field for the percentage

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

class LedgerRuleUpdate(BaseModel):
    """Model for updating an existing ledger rule."""
    ledger_name: str
    regex_pattern: str

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

class KnownLedgerSummaryWithRuleCount(BaseModel):
    """A summary of a known ledger, including its rule count."""
    id: str
    ledger_name: str
    sample_count: int
    rule_count: int

class PaginatedLedgersResponse(BaseModel):
    """Response model for paginated known ledgers."""
    total_ledgers: int
    total_pages: int
    ledgers: List[KnownLedgerSummaryWithRuleCount]

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

class DashboardStats(BaseModel):
    """Defines the metrics for the main dashboard."""
    total_clients: int
    statements_processed: int
    regex_patterns: int
    success_rate: float

class VoucherGenerationRequest(BaseModel):
    """Defines the user's selection for voucher export."""
    include_receipts: bool = True
    include_payments: bool = True
    include_contras: bool = True
    filename: str

class KnownLedgerSummary(BaseModel):
    """A summary of a known ledger for listing purposes."""
    id: str
    ledger_name: str
    sample_count: int

class SampleModel(BaseModel):
    """Defines the structure of a single transaction sample."""
    narration: str
    amount: float
    type: str # "Credit" or "Debit"

class PaginatedSamplesResponse(BaseModel):
    """Response model for paginated samples."""
    total_samples: int
    samples: List[SampleModel]

class ReclassifySubsetRequest(BaseModel):
    """Defines the request body for re-classifying a subset of transactions."""
    transactions: List[Dict[str, Any]]

# Utility Functions
def detect_date_columns(headers: List[str]) -> List[str]:
    """Detect potential date columns"""
    # --- REPLACEMENT for the return statement ---
    normalized_headers = [h.lower().replace(" ", "").replace("_", "") for h in headers]
    date_keywords = ['date', 'dt', 'txndate', 'transactiondate', 'valuedate'] # Also normalized

    matched_headers = []
    for i, normalized_header in enumerate(normalized_headers):
        if any(keyword in normalized_header for keyword in date_keywords):
            matched_headers.append(headers[i]) # Append the ORIGINAL header
        
    return matched_headers

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

def extract_keywords(text: str, devalued_keywords: set) -> set:
    """Extracts meaningful keywords from a narration string."""
    # This logic is ported from your regex_generator_v_4_clear_input.html
    tokens = re.split(r'[^A-Z0-9@]+', text.upper())
    meaningful_tokens = set()
    for token in tokens:
        if (
            token and
            len(token) > 2 and
            not re.match(r'\d{4,}', token) and
            not token.isdigit() and
            "**" not in token and
            token not in devalued_keywords
        ):
            meaningful_tokens.add(token)
    return meaningful_tokens

# In server.py

# --- FIND AND REPLACE THE ENTIRE get_dashboard_stats FUNCTION ---
@api_router.get("/dashboard-stats", response_model=DashboardStats)
async def get_dashboard_stats(db: AsyncSession = Depends(database.get_db)):
    """Calculates and returns key statistics for the main dashboard."""
    
    # --- THE FIX: Run queries sequentially, not in parallel ---
    total_clients = await db.scalar(select(func.count(models.Client.id)))
    statements_processed = await db.scalar(select(func.count(models.BankStatement.id)))
    regex_patterns = await db.scalar(select(func.count(models.LedgerRule.id)))
    # --- END OF FIX ---

    # Calculate success rate (this part of the logic is fine)
    total_transactions = 0
    matched_transactions = 0
    
    stmt_query = select(models.BankStatement.processed_data)
    result = await db.execute(stmt_query)
    all_processed_data = result.scalars().all()

    for processed_data in all_processed_data:
        if isinstance(processed_data, list):
            total_transactions += len(processed_data)
            matched_transactions += sum(1 for t in processed_data if t.get("matched_ledger") != "Suspense")
            
    success_rate = (matched_transactions / total_transactions * 100) if total_transactions > 0 else 0.0

    return DashboardStats(
        total_clients=total_clients or 0, # Add 'or 0' for safety
        statements_processed=statements_processed or 0,
        regex_patterns=regex_patterns or 0,
        success_rate=round(success_rate, 2)
    )
# --- END OF REPLACEMENT ---
# --- REPLACEMENT for generate_regex_from_narrations ---
async def generate_regex_from_narrations(narrations: List[str], db: AsyncSession) -> str:
    """
    Generate a precise, order-aware regex pattern from narrations using common, meaningful keywords.
    """
    if not narrations:
        return ""

    # Fetch devalued keywords for filtering
    result = await db.execute(select(models.DevaluedKeyword.keyword))
    devalued_keywords_set = set(result.scalars().all())

    # 1. Find keywords that are common to ALL narrations in the cluster
    keyword_sets = [extract_keywords(n, devalued_keywords_set) for n in narrations]
    if not any(keyword_sets):
        return ".*"  # Fallback if no meaningful keywords are found at all
    common_keywords = set.intersection(*keyword_sets)

    # 2. If no keywords are common across all items, the cluster is weak.
    # Fallback to a simple regex based on the first item's first keyword.
    if not common_keywords:
        first_narration_keywords = sorted(list(extract_keywords(narrations[0], devalued_keywords_set)))
        if first_narration_keywords:
            return f".*\\b{re.escape(first_narration_keywords[0])}\\b.*"
        return ".*"

    # 3. Get an ordered list of all tokens from the first narration to use as a template.
    first_narration_ordered_tokens = re.split(r'[^A-Z0-9@]+', narrations[0].upper())

    # 4. Build the final list by picking common keywords in the order they appear in the template.
    ordered_common_keywords = []
    seen = set()
    for token in first_narration_ordered_tokens:
        if token in common_keywords and token not in seen:
            ordered_common_keywords.append(token)
            seen.add(token) # Prevents adding a keyword more than once

    # 5. Construct the final regex pattern from the ordered list.
    if ordered_common_keywords:
        pattern = ".*".join(f"\\b{re.escape(k)}\\b" for k in ordered_common_keywords)
        return f".*{pattern}.*" # Wrap with wildcards for flexibility
    
    # This is a final fallback, which should now be rarely needed.
    return f".*\\b{re.escape(sorted(list(common_keywords))[0])}\\b.*"

# In server.py

# --- ADD THIS ENTIRE NEW HELPER FUNCTION ---
# --- DEFINITIVE REPLACEMENT for pre_cluster_by_shared_keywords ---
async def pre_cluster_by_shared_keywords(
    transactions: List[Dict[str, Any]], narration_column: str, db: AsyncSession
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: # <-- FIX 1: Correct type hint syntax
    """
    Finds high-confidence clusters based on shared keywords, ensuring no duplicates.
    Returns a tuple of (found_clusters, remaining_transactions).
    """
    result = await db.execute(select(models.DevaluedKeyword.keyword))
    devalued_keywords_set = set(result.scalars().all())
    
    keyword_map = defaultdict(list)
    for transaction in transactions:
        narration = str(transaction.get(narration_column, ""))
        keywords = extract_keywords(narration, devalued_keywords_set)
        for keyword in keywords:
            keyword_map[keyword].append(transaction)
            
    potential_clusters = []
    for keyword, mapped_transactions in keyword_map.items():
        if len(mapped_transactions) >= 2:
            potential_clusters.append(mapped_transactions)
            
    # Prioritize larger potential clusters first
    sorted_clusters = sorted(potential_clusters, key=len, reverse=True)
    
    final_clusters = []
    # --- FIX 2: Track using in-memory object ID, not the 'Srl' key ---
    processed_ids = set() 
    
    for cluster_group in sorted_clusters:
        # Filter out any transactions that have already been claimed by a larger cluster
        unclaimed_transactions = [
            t for t in cluster_group if id(t) not in processed_ids
        ]
        
        if len(unclaimed_transactions) >= 2:
            narrations = [str(t.get(narration_column, "")) for t in unclaimed_transactions]
            suggested_regex = await generate_regex_from_narrations(narrations, db)
            
            final_clusters.append({
                "cluster_id": str(uuid.uuid4()),
                "transactions": unclaimed_transactions,
                "suggested_regex": suggested_regex,
            })
            
            # Add the memory IDs of these transactions to the set so they can't be reused
            for t in unclaimed_transactions:
                processed_ids.add(id(t))

    # The remaining transactions are those whose memory IDs were never processed
    remaining_transactions = [
        t for t in transactions if id(t) not in processed_ids
    ]
    
    return final_clusters, remaining_transactions
# --- END OF REPLACEMENT ---
# --- FINAL REPLACEMENT for cluster_narrations ---
async def cluster_narrations(
    transactions: List[Dict[str, Any]], narration_column: str, db: AsyncSession, n_clusters: int = None
) -> List[Dict[str, Any]]:
    """Cluster similar transactions using TF-IDF and K-means, returning full transaction objects."""
    
    narrations = [str(t.get(narration_column, "")) for t in transactions]
    if not any(narrations):
        return []

    # Fetch devalued keywords for the vectorizer
    result = await db.execute(select(models.DevaluedKeyword.keyword))
    devalued_keywords = result.scalars().all()
    stop_words = [kw.lower() for kw in devalued_keywords] + ['english'] # FIX: Ensure all stop words are lowercase
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=100,
        ngram_range=(1, 2),
        lowercase=True
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(narrations)
        
        # Determine the optimal number of clusters if not specified
        if n_clusters is None:
            # Adjust clustering for smaller sample sizes
            num_samples = len(transactions)
        if n_clusters is None:
            if num_samples <= 5: # If 5 or fewer items, make each its own small cluster or group them tightly
                n_clusters = num_samples 
            else: # For larger groups, use the existing logic
                n_clusters = min(max(2, num_samples // 5), 10)
        # Ensure n_clusters is not zero if there are samples
        if num_samples > 0 and n_clusters == 0:
            n_clusters = 1
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Group full transaction objects by their assigned cluster label
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(transactions[idx])
        
        # Prepare the final list of cluster objects for the frontend
        cluster_objects = []
        for cluster_id, clustered_transactions in clusters.items():
            cluster_narrations_list = [str(t.get(narration_column, "")) for t in clustered_transactions]
            suggested_regex = await generate_regex_from_narrations(cluster_narrations_list, db)
            
            cluster_objects.append({
                "cluster_id": str(uuid.uuid4()),
                "transactions": clustered_transactions, # Return full objects
                "suggested_regex": suggested_regex,
            })
        return cluster_objects
    
    except Exception as e:
        logging.error(f"Clustering failed: {e}")
        # Fallback: Create a separate cluster for each transaction
        cluster_objects = []
        for transaction in transactions:
            narration = [str(transaction.get(narration_column, ""))]
            suggested_regex = await generate_regex_from_narrations(narration, db)
            cluster_objects.append({
                "cluster_id": str(uuid.uuid4()),
                "transactions": [transaction],
                "suggested_regex": suggested_regex
            })
        return cluster_objects


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
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))

        
        # --- END OF REPLACEMENT ---
        df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
        # Convert all data types in the DataFrame to strings to ensure JSON compatibility.

        df = df.dropna(how='all').reset_index(drop=True)

        # 1. Replace pandas' NaN/NaT with None, which is JSON-serializable as 'null'.
        # This is more robust than fillna('') as it preserves the "empty" nature.
        df = df.replace({np.nan: None, pd.NaT: None})

        # 2. Convert the entire clean DataFrame into a list of standard Python objects.
        # This is the key step that correctly converts numpy.int64 -> int, etc.
        sanitized_records = df.to_dict('records')

        headers = df.columns.tolist()
        preview_data = sanitized_records[:10] # Use the sanitized records for the preview
        suggested_mapping = suggest_column_mapping(headers)
        
        # Store the sanitized data temporarily in the new table
        temp_file = models.TempFile(
            filename=file.filename,
            headers=headers,
            raw_data=sanitized_records
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
def _clean_amount_string(value: Any) -> str:
    """
    Cleans a string to make it safely convertible to a number by removing
    commas and other non-numeric characters (except the decimal point).
    """
    if value is None:
        return "0"
    s = str(value)
    # Remove commas, then keep only digits, the first decimal point, and a leading minus sign.
    s = s.replace(",", "")
    # A simple regex to strip anything that's not a digit or a decimal point.
    # This is a safe way to handle currency symbols, etc.
    s = re.sub(r"[^0-9.-]", "", s)
    return s if s else "0"
# --- ADD THIS HELPER FUNCTION ---

def normalize_transaction_data(
    raw_data: List[Dict], mapping: "ColumnMapping"
) -> List[Dict]:
    """
    Normalizes raw transaction data from separate credit/debit columns
    into a single amount column and a CR/DR indicator column.
    """
    if mapping.statement_format != "separate_credit_debit":
        return raw_data  # No changes needed if not in the separate format

    normalized_data = []
    credit_col = mapping.credit_column
    debit_col = mapping.debit_column

    if not credit_col or not debit_col:
        # If columns are not defined, we cannot proceed with normalization.
        return raw_data

    for row in raw_data:
        new_row = row.copy()
        try:
            # Safely convert to numeric, coercing errors to NaN, then to 0.0
            credit_val = pd.to_numeric(_clean_amount_string(row.get(credit_col)), errors='coerce')
            debit_val = pd.to_numeric(_clean_amount_string(row.get(debit_col)), errors='coerce')
            credit_val = credit_val if pd.notna(credit_val) else 0.0
            debit_val = debit_val if pd.notna(debit_val) else 0.0
            
            # Logic to determine Amount and CR/DR
            if credit_val > 0:
                # Convert from numpy.float64 to a standard Python float
                new_row["Amount (INR)"] = float(credit_val)
                new_row["CR/DR"] = "CR"
            elif debit_val > 0:
                # Convert from numpy.float64 to a standard Python float
                new_row["Amount (INR)"] = float(debit_val)
                new_row["CR/DR"] = "DR"
            else:
                # Handle rows with 0 in both, or non-standard entries
                new_row["Amount (INR)"] = 0.0
                new_row["CR/DR"] = "N/A"

            # Remove original credit/debit columns to avoid duplication
            if credit_col in new_row:
                del new_row[credit_col]
            if debit_col in new_row:
                del new_row[debit_col]

            normalized_data.append(new_row)

        except (ValueError, TypeError):
            # If conversion fails for a row, append it as is but log it.
            logging.warning(f"Could not normalize row: {row}. Appending as is.")
            normalized_data.append(row)
            continue
            
    return normalized_data

@api_router.post("/statements/{statement_id}/reclassify-subset", response_model=List[Dict[str, Any]])
async def reclassify_subset(
    statement_id: str,
    request: ReclassifySubsetRequest,
    db: AsyncSession = Depends(database.get_db)
):
    """
    Re-applies all regex rules to a specific subset of transactions and returns the result
    without saving it to the database.
    """
    statement = await db.get(models.BankStatement, statement_id)
    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")

    # Fetch all rules for the client
    query = select(models.LedgerRule).where(models.LedgerRule.client_id == statement.client_id)
    result = await db.execute(query)
    patterns = result.scalars().all()

    # Create a list to hold the results
    reclassified_transactions = []

    # Iterate ONLY through the transactions provided in the request
    for transaction in request.transactions:
        # The frontend uses 'Narration' as the standardized key
        narration = str(transaction.get("Narration", ""))
        matched = False
        
        # Apply the same matching logic as the main classification function
        for pattern in patterns:
            if re.search(pattern.regex_pattern, narration, re.IGNORECASE):
                # Update the ledger and ensure user_confirmed is false
                transaction["matched_ledger"] = pattern.ledger_name
                transaction["user_confirmed"] = False 
                reclassified_transactions.append(transaction)
                matched = True
                break
        
        if not matched:
            # If no rule matches, it becomes a soft suspense item
            transaction["matched_ledger"] = "Suspense"
            transaction["user_confirmed"] = False
            reclassified_transactions.append(transaction)

    return reclassified_transactions

# --- REPLACEMENT for confirm_column_mapping ---
@api_router.post("/confirm-mapping/{file_id}")
async def confirm_column_mapping(file_id: str, mapping: ColumnMapping, client_id: str = Query(...), bank_account_id: str = Query(...), db: AsyncSession = Depends(database.get_db)):
    """Confirm column mapping and process statement"""
    temp_file = await db.get(models.TempFile, file_id)
    if not temp_file:
        raise HTTPException(status_code=404, detail="File not found or has expired")

    try:
        # --- START OF NEW LOGIC ---
        # Normalize the data before creating the statement object
        processed_raw_data = normalize_transaction_data(temp_file.raw_data, mapping)
        
        # If normalization occurred, we need to update the column mapping
        # to reflect the new standardized column names.
        final_mapping = mapping.dict()
        if mapping.statement_format == "separate_credit_debit":
            final_mapping["amount_column"] = "Amount (INR)"
            final_mapping["crdr_column"] = "CR/DR"
            # Set original columns to None as they no longer exist
            final_mapping["credit_column"] = None
            final_mapping["debit_column"] = None
        # --- END OF NEW LOGIC ---

        # Create a permanent BankStatement record
        statement = models.BankStatement(
            client_id=client_id,
            bank_account_id=bank_account_id,
            filename=temp_file.filename,
            column_mapping=final_mapping,  # Use the potentially updated mapping
            statement_format=mapping.statement_format,
            raw_data=processed_raw_data  # Use the normalized data
        )
        
        db.add(statement)
        await db.delete(temp_file)
        await db.commit()
        await db.refresh(statement)
        
        return {"message": "Statement processed successfully", "statement_id": statement.id}
    
    except Exception as e:
        await db.rollback()
        logging.error(f"Mapping confirmation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing mapping: {str(e)}")

# Regex Pattern Management
# --- REPLACEMENT for create_regex_pattern ---
@api_router.post("/ledger-rules", response_model=LedgerRule)
async def create_ledger_rule(rule_data: LedgerRuleBase, db: AsyncSession = Depends(database.get_db)):
        """Create a new ledger rule"""
        db_rule = models.LedgerRule(**rule_data.dict())
        
        db.add(db_rule)
        await db.commit()
        await db.refresh(db_rule)
        # --- START OF ADDITION ---
        # Now, create and save the corresponding feedback entry.
        feedback_entry = models.ClassificationFeedback(
            client_id=db_rule.client_id,
            ledger_name=db_rule.ledger_name,
            source_narrations=db_rule.sample_narrations,
            regex_pattern=rule_data.regex_pattern # Store the regex directly
        )
        # 3. Add both objects to the session.
        db.add(db_rule)
        db.add(feedback_entry)
    
        try:
            # 4. Commit the session ONCE to save both objects atomically.
            await db.commit()
        except Exception as e:
            # If anything fails, roll back both changes.
            await db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

        # 5. Refresh the rule object to load its generated ID and timestamps.
        await db.refresh(db_rule)
        
        # 6. Return the rule object. The session is still active, so no error will occur.
        return db_rule

# --- REPLACEMENT for get_client_regex_patterns ---
@api_router.get("/ledger-rules/{client_id}", response_model=List[LedgerRule])
async def get_client_ledger_rules(client_id: str, db: AsyncSession = Depends(database.get_db)):
        """Get all ledger rules for a client"""
        query = select(models.LedgerRule).where(models.LedgerRule.client_id == client_id)
        
        result = await db.execute(query)
        rules = result.scalars().all()
        
        return rules
# --- END OF REPLACEMENT ---
# --- ADD THIS ENTIRE HELPER FUNCTION ---
def create_standardized_transaction(transaction: Dict, mapping: Dict) -> Dict:
    """
    Creates a new, clean transaction dictionary containing only the essential
    columns under standardized key names.
    """
    standardized = {}
    # --- START: Date Normalization Logic ---
    raw_date_str = transaction.get(mapping.get("date_column"))
    if raw_date_str:
        # Use pandas for robust parsing, hinting that the day is first.
        # errors='coerce' will return NaT (Not a Time) for unparseable dates.
        parsed_date = pd.to_datetime(raw_date_str, dayfirst=True, errors='coerce')
        
        # Check if parsing was successful. pd.notna checks for NaT.
        if pd.notna(parsed_date):
            # Reformat to our single, consistent DD/MM/YYYY standard, stripping all time info.
            standardized["Date"] = parsed_date.strftime('%d/%m/%Y')
        else:
            # If parsing fails, use the original string as a safe fallback.
            standardized["Date"] = str(raw_date_str)
    else:
        standardized["Date"] = None # Handle cases where the date is missing.
    # --- END: Date Normalization Logic ---
    # Map essential columns using the provided mapping
    standardized["Narration"] = transaction.get(mapping.get("narration_column"))
    standardized["Amount"] = transaction.get(mapping.get("amount_column"))
    raw_cr_dr = transaction.get(mapping.get("crdr_column"), "")
    # Clean and normalize it immediately
    clean_cr_dr = str(raw_cr_dr).strip().replace(".", "").upper()
    standardized["CR/DR"] = clean_cr_dr

    # Only include Balance if it was mapped by the user
    if mapping.get("balance_column"):
        standardized["Balance"] = transaction.get(mapping.get("balance_column"))
        
    return standardized
# --- END OF ADDITION ---
# --- ADD THESE TWO NEW ENDPOINTS ---
@api_router.put("/ledger-rules/{rule_id}", response_model=LedgerRule)
async def update_ledger_rule(rule_id: str, rule_data: LedgerRuleUpdate, db: AsyncSession = Depends(database.get_db)):
    """Update a ledger rule by its ID."""
    db_rule = await db.get(models.LedgerRule, rule_id)
    if not db_rule:
        raise HTTPException(status_code=404, detail="Ledger rule not found")
    
    # Update the rule object with the new data
    update_data = rule_data.dict()
    for key, value in update_data.items():
        setattr(db_rule, key, value)
        
    await db.commit()
    await db.refresh(db_rule)
    return db_rule

@api_router.delete("/ledger-rules/{rule_id}", status_code=204)
async def delete_ledger_rule(rule_id: str, db: AsyncSession = Depends(database.get_db)):
    """Delete a ledger rule by its ID."""
    db_rule = await db.get(models.LedgerRule, rule_id)
    if not db_rule:
        raise HTTPException(status_code=404, detail="Ledger rule not found")
    
    await db.delete(db_rule)
    await db.commit()
    return None # Return None for 204 No Content response
# --- END OF ADDITION ---
# In server.py
@api_router.get("/clients/{client_id}/rule-stats", response_model=RuleStatsResponse)
async def get_rule_stats(client_id: str, db: AsyncSession = Depends(database.get_db)):
    """Calculates on-demand health statistics for all rules of a given client."""
    
    # Step 1: Fetch all statements for the client
    stmt_query = select(models.BankStatement).where(models.BankStatement.client_id == client_id)
    result = await db.execute(stmt_query)
    statements = result.scalars().all()

    # Step 2: Initialize a structure to hold our calculations
    # We use ledger_name as the key.
    stats = defaultdict(lambda: {"total_matches": 0, "last_used_dt": None})

    # Step 3: Iterate through all transactions of all statements
    for statement in statements:
        if not isinstance(statement.processed_data, list):
            continue
        
        for transaction in statement.processed_data:
            ledger_name = transaction.get("matched_ledger")
            
            # We only care about transactions matched to a specific rule, not Suspense
            if not ledger_name or ledger_name == "Suspense":
                continue

            # Increment the match counter
            stats[ledger_name]["total_matches"] += 1

            # Update the last used date
            date_str = transaction.get("Date")
            if date_str:
                try:
                    # Parse the date string into a datetime object for comparison
                    # This handles formats like 'DD/MM/YYYY HH:MM:SS'
                    transaction_dt = datetime.strptime(date_str.split(' ')[0], '%d/%m/%Y')
                    
                    if (stats[ledger_name]["last_used_dt"] is None or 
                        transaction_dt > stats[ledger_name]["last_used_dt"]):
                        stats[ledger_name]["last_used_dt"] = transaction_dt
                except (ValueError, TypeError):
                    continue # Ignore rows with unparseable dates

    # Step 4: Format the results for the final JSON response
    final_stats = {
        ledger: RuleStat(
            total_matches=data["total_matches"],
            last_used=data["last_used_dt"].strftime('%d %b %Y') if data["last_used_dt"] else None
        )
        for ledger, data in stats.items()
    }

    return RuleStatsResponse(stats=final_stats)
# --- END OF ADDITION ---

# --- FIND AND REPLACE THE ENTIRE classify-transactions FUNCTION ---
@api_router.post("/classify-transactions/{statement_id}")
async def classify_transactions(
    statement_id: str, 
    db: AsyncSession = Depends(database.get_db),
    force_reclassify: bool = Query(False, description="Force re-matching of rules against pending transactions")
):
    """
    Classify transactions. This endpoint is STATE-AWARE and now ONLY uses processed_data.
    """
    statement = await db.get(models.BankStatement, statement_id)
    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")

    mapping = statement.column_mapping
    narration_col = mapping.get("narration_column")
    
    classified_transactions = statement.processed_data if isinstance(statement.processed_data, list) else []

    if not classified_transactions: # First run
        # ... (This entire 'if not classified_transactions' block remains unchanged)
        query = select(models.LedgerRule).where(models.LedgerRule.client_id == statement.client_id)
        result = await db.execute(query)
        patterns = result.scalars().all()
        raw_data = statement.raw_data or []
        for transaction in raw_data:
            standardized_transaction = create_standardized_transaction(transaction, mapping)
            narration = str(transaction.get(narration_col, ""))
            matched = False
            for pattern in patterns:
                if re.search(pattern.regex_pattern, narration, re.IGNORECASE):
                    standardized_transaction["matched_ledger"] = pattern.ledger_name
                    classified_transactions.append(standardized_transaction)
                    matched = True
                    break
            if not matched:
                standardized_transaction["matched_ledger"] = "Suspense"
                classified_transactions.append(standardized_transaction)
        statement.processed_data = classified_transactions
        await db.commit()
        await db.refresh(statement)

    if force_reclassify: # Intelligent Merge
        # ... (This entire 'if force_reclassify' block remains unchanged)
        items_to_recheck = [t for t in classified_transactions if t.get("matched_ledger") == "Suspense" and not t.get("user_confirmed")]
        if items_to_recheck:
            query = select(models.LedgerRule).where(models.LedgerRule.client_id == statement.client_id)
            result = await db.execute(query)
            patterns = result.scalars().all()
            reclassification_updates = {}
            for item in items_to_recheck:
                for pattern in patterns:
                    if re.search(pattern.regex_pattern, item['Narration'], re.IGNORECASE):
                        reclassification_updates[item['Narration']] = pattern.ledger_name
                        break
            if reclassification_updates:
                for i, t in enumerate(classified_transactions):
                    if t['Narration'] in reclassification_updates:
                        classified_transactions[i]['matched_ledger'] = reclassification_updates[t['Narration']]
                statement.processed_data = classified_transactions
                await db.commit()
                await db.refresh(statement)

    # --- THIS IS THE CRITICAL CHANGE ---
    # We now identify the transactions to cluster directly from our master list.
    transactions_to_cluster = [
        t for t in classified_transactions
        if t.get("matched_ledger") == "Suspense" and not t.get("user_confirmed")
    ]
    # --- END OF CHANGE ---

    # The clustering functions will now receive standardized objects.
    # We pass the STANDARDIZED narration key "Narration" to them.
    pre_clusters, remaining_for_ml = await pre_cluster_by_shared_keywords(transactions_to_cluster, "Narration", db)
    
    debit_transactions = [t for t in remaining_for_ml if str(t.get("CR/DR","")).strip().upper() == "DR"]
    credit_transactions = [t for t in remaining_for_ml if str(t.get("CR/DR","")).strip().upper() == "CR"]
    
    debit_clusters = await cluster_narrations(debit_transactions, "Narration", db)
    credit_clusters = await cluster_narrations(credit_transactions, "Narration", db)
    final_clusters = pre_clusters + debit_clusters + credit_clusters
    
    return {
        "classified_transactions": classified_transactions,
        "unmatched_clusters": final_clusters,
        "total_transactions": len(statement.raw_data or []),
        "matched_transactions": len([t for t in classified_transactions if t.get("matched_ledger") != "Suspense"]),
        "unmatched_transactions": len(transactions_to_cluster)
    }
# --- END OF REPLACEMENT ---
# In server.py, after the classify_transactions function

# --- ADD THIS ENTIRE ENDPOINT ---
@api_router.post("/statements/{statement_id}/update-transactions")
async def update_transactions(statement_id: str, request: UpdateTransactionsRequest, db: AsyncSession = Depends(database.get_db)):
    """Updates the processed_data for a given statement using a canonical-key merge to avoid duplicates."""
    statement = await db.get(models.BankStatement, statement_id)
    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")

    existing = statement.processed_data or []
    incoming = request.processed_data or []

    def _normalize_narration(n):
        if not n:
            return ""
        s = str(n).strip().lower()
        s = re.sub(r'\s+', ' ', s)
        return s

    def _normalize_date(d):
        if not d:
            return ""
        s = str(d).split(' ')[0].strip()
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y", "%d-%b-%y"):
            try:
                return datetime.strptime(s, fmt).strftime("%d/%m/%Y")
            except Exception:
                continue
        return s.lower()

    def _normalize_amount(a):
        if a is None:
            return 0.0
        try:
            s = str(a).replace(',', '').strip()
            return float(s)
        except Exception:
            try:
                return float(re.sub(r'[^\d.-]', '', str(a)))
            except Exception:
                return 0.0

    def canonical_key(tx):
        amount = tx.get("Amount") if "Amount" in tx else tx.get("Amount (INR)") if "Amount (INR)" in tx else tx.get("amount") if "amount" in tx else tx.get("amt")
        return f"{_normalize_narration(tx.get('Narration') or tx.get('narration'))}||{_normalize_date(tx.get('Date') or tx.get('date'))}||{_normalize_amount(amount)}"

    # Build incoming map keyed by canonical key (last one wins for duplicates in incoming)
    incoming_map = {}
    for tx in incoming:
        incoming_map[canonical_key(tx)] = tx

    updated_count = 0
    appended_count = 0
    processed_incoming_keys = set()

    # Merge: prefer updating existing entries in-place (preserve order)
    merged = []
    for orig in existing:
        key = canonical_key(orig)
        if key in incoming_map:
            incoming_tx = incoming_map[key]
            # Merge: incoming fields override existing ones (but keep other fields)
            merged_tx = {**orig, **incoming_tx}

            # Safety rule: if transaction is (or becomes) Suspense, ensure user_confirmed is explicitly FALSE
            # unless the incoming payload explicitly set user_confirmed (in which case honor it).
            if str(merged_tx.get('matched_ledger', '')).strip().lower() == 'suspense':
                if 'user_confirmed' in incoming_tx:
                    merged_tx['user_confirmed'] = bool(incoming_tx['user_confirmed'])
                else:
                    merged_tx['user_confirmed'] = False

            # If incoming explicitly set user_confirmed (true/false), ensure it is honored regardless of ledger
            elif 'user_confirmed' in incoming_tx:
                merged_tx['user_confirmed'] = bool(incoming_tx['user_confirmed'])

            merged.append(merged_tx)
            processed_incoming_keys.add(key)
            updated_count += 1
        else:
            merged.append(orig)

    # Append any incoming transactions that did not match existing keys
    for key, tx in incoming_map.items():
        if key in processed_incoming_keys:
            continue
        # For new appended tx: ensure Suspense items are unconfirmed by default
        if str(tx.get('matched_ledger', '')).strip().lower() == 'suspense' and 'user_confirmed' not in tx:
            tx['user_confirmed'] = False
        merged.append(tx)
        appended_count += 1

    # Final deduplication pass to ensure no duplicate canonical keys remain (preserve first occurrence)
    seen = set()
    final_list = []
    for tx in merged:
        k = canonical_key(tx)
        if k in seen:
            continue
        seen.add(k)
        final_list.append(tx)

    statement.processed_data = final_list
    await db.commit()

    return {
        "message": "Transactions merged successfully",
        "updated": updated_count,
        "appended": appended_count,
        "final_count": len(final_list)
    }
# --- END OF ADDITION ---
# --- ADD THIS ENTIRE ENDPOINT ---

# --- ADD THIS ENTIRE HELPER FUNCTION ---
def generate_tally_rows(vouchers: List[Dict], bank_ledger_name: str, voucher_type: str) -> List[List[Any]]:
    """
    Generates a list of rows in the two-line Tally format for Excel export.
    """
    output_rows = []
    voucher_number = 1

    for t in vouchers:
        is_contra = voucher_type == 'Contra'
        
        # Safely parse amount from standardized data
        amount = float(str(t.get('Amount', '0')).replace(',', ''))
        
        # Format date from standardized data
        date_str = t.get('Date', '').split(' ')[0]
        try:
            formatted_date = datetime.strptime(date_str, '%d/%m/%Y').strftime('%d-%b-%Y')
        except ValueError:
            formatted_date = date_str

        narration = t.get('Narration', '')
        party_ledger = t.get('matched_ledger', 'Suspense')

        # Line 1: The Party Ledger Entry
        party_cr_dr = 'Dr' if is_contra else ('Cr' if voucher_type == 'Receipt' else 'Dr')
        row1 = [
            formatted_date, voucher_type, voucher_number, '', '', 
            party_ledger, f"{amount:.2f}", party_cr_dr,
            '', '', '', '', '', '', narration
        ]
        
        # Line 2: The Bank Ledger Entry
        bank_cr_dr = 'Cr' if is_contra else ('Dr' if voucher_type == 'Receipt' else 'Cr')
        row2 = [
            '', '', '', '', '', 
            bank_ledger_name, f"{amount:.2f}", bank_cr_dr,
            '', '', '', '', '', '', ''
        ]
        
        output_rows.append(row1)
        output_rows.append(row2)
        voucher_number += 1
        
    return output_rows
# --- END OF ADDITION ---

# In server.py

# --- FIND AND REPLACE the entire generate_vouchers function WITH THESE TWO ENDPOINTS ---

@api_router.get("/vouchers/{statement_id}/summary")
async def get_voucher_summary(statement_id: str, db: AsyncSession = Depends(database.get_db)):
    """Gets the counts of each voucher type to populate the download modal."""
    
    # --- THIS IS THE FIX ---
    # Eagerly load the related bank_account and client to prevent session errors.
    query = (
        select(models.BankStatement)
        .options(
            selectinload(models.BankStatement.bank_account),
            selectinload(models.BankStatement.client)
        )
        .where(models.BankStatement.id == statement_id)
    )
    result = await db.execute(query)
    statement = result.scalar_one_or_none()
    # --- END OF FIX ---

    if not statement: raise HTTPException(status_code=404, detail="Statement not found")
    
    # Now we can safely access the related objects because they were pre-loaded.
    bank_account = statement.bank_account
    client = statement.client

    if not bank_account: raise HTTPException(status_code=404, detail="Bank account not found")
    if not client: raise HTTPException(status_code=404, detail="Client not found")
    print("\n" + "="*50)
    print("INSIDE GET_VOUCHER_SUMMARY")
    print(f"  - Statement ID: {statement.id}")
    print(f"  - Raw processed_data from DB: {statement.processed_data}")
    
    processed_data = statement.processed_data or []
    
    print(f"  - Length of processed_data list being used: {len(processed_data)}")
    print("="*50 + "\n")
    # --- END: TEMPORARY DIAGNOSTIC CODE ---
    contra_list = set(bank_account.contra_list or [])
    
    receipts = 0
    payments = 0
    contras = 0

    for t in processed_data:
        cr_dr_val = str(t.get("CR/DR", "")).strip().upper()
        ledger = t.get("matched_ledger")

        if cr_dr_val.startswith("CR"):
            receipts += 1
        elif cr_dr_val.startswith("DR"):
            if ledger in contra_list:
                contras += 1
            else:
                payments += 1
    
    first_date_str = next((t.get("Date") for t in processed_data if t.get("Date")), None)
    month_year = "vouchers"
    if first_date_str:
        try:
            month_year = datetime.strptime(first_date_str.split(' ')[0], '%d/%m/%Y').strftime("%B_%Y")
        except ValueError:
            pass
            
    sanitized_ledger_name = bank_account.ledger_name.replace("/", "_").replace("\\", "_")
    suggested_filename = f"{client.name}_{sanitized_ledger_name}_{month_year}"

    return {
        "receipt_count": receipts,
        "payment_count": payments,
        "contra_count": contras,
        "suggested_filename": suggested_filename
    }

@api_router.post("/generate-vouchers/{statement_id}")
async def generate_vouchers(statement_id: str, request: VoucherGenerationRequest, db: AsyncSession = Depends(database.get_db)):
    """Generates a Tally-compatible XLSX file by populating a template."""
    statement = await db.get(models.BankStatement, statement_id)
    if not statement: raise HTTPException(status_code=404, detail="Statement not found")
    bank_account = await db.get(models.BankAccount, statement.bank_account_id)
    if not bank_account: raise HTTPException(status_code=404, detail="Bank account not found")

    processed_data = statement.processed_data or []
    contra_list = set(bank_account.contra_list or [])
    bank_ledger_name = bank_account.ledger_name
    
    # Sort data into lists
    receipt_data, payment_data, contra_data = [], [], []
    for t in processed_data:
        cr_dr_val = str(t.get("CR/DR", "")).strip().upper()
        ledger = t.get("matched_ledger")

        if cr_dr_val.startswith("CR"):
            receipt_data.append(t)
        elif cr_dr_val.startswith("DR"):
            if ledger in contra_list:
                contra_data.append(t)
            else:
                payment_data.append(t)

    # Load the template workbook
    template_path = "templates/AccountingVouchers.xlsx"
    try:
        workbook = openpyxl.load_workbook(template_path)
        template_sheet = workbook["Accounting Voucher"]
    except (FileNotFoundError, KeyError):
        raise HTTPException(status_code=500, detail="Tally template file not found or is missing 'Accounting Voucher' sheet.")

    # Process each selected voucher type
    if request.include_receipts and receipt_data:
        receipt_sheet = workbook.copy_worksheet(template_sheet)
        receipt_sheet.title = "Receipts"
        for row in generate_tally_rows(receipt_data, bank_ledger_name, "Receipt"):
            receipt_sheet.append(row)

    if request.include_payments and payment_data:
        payment_sheet = workbook.copy_worksheet(template_sheet)
        payment_sheet.title = "Payments"
        for row in generate_tally_rows(payment_data, bank_ledger_name, "Payment"):
            payment_sheet.append(row)

    if request.include_contras and contra_data:
        contra_sheet = workbook.copy_worksheet(template_sheet)
        contra_sheet.title = "Contras"
        for row in generate_tally_rows(contra_data, bank_ledger_name, "Contra"):
            contra_sheet.append(row)
    
    # Remove the original template sheet
    del workbook["Accounting Voucher"]

    # Move the "Read Me" sheet to the very end, if it exists
    if "Accounting Voucher (Read Me)" in workbook.sheetnames:
        read_me_sheet = workbook["Accounting Voucher (Read Me)"]
        workbook.move_sheet(read_me_sheet, offset=len(workbook.sheetnames))
    
    # Save to an in-memory buffer
    output_buffer = io.BytesIO()
    workbook.save(output_buffer)
    output_buffer.seek(0)
    
    final_filename = f"{request.filename}.xlsx"
    headers = {'Content-Disposition': f'attachment; filename="{final_filename}"'}
    return StreamingResponse(output_buffer, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
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


# --- FIND AND REPLACE THE ENTIRE get_client_statements FUNCTION ---
@api_router.get("/clients/{client_id}/statements", response_model=List[StatementMetadata])
async def get_client_statements(client_id: str, db: AsyncSession = Depends(database.get_db)):
    """Get all statement metadata for a specific client, including smart status."""
    
    query = (
        select(models.BankStatement, models.BankAccount.ledger_name)
        .join(models.BankAccount, models.BankStatement.bank_account_id == models.BankAccount.id)
        .where(models.BankStatement.client_id == client_id)
        .order_by(models.BankStatement.upload_date.desc())
    )
    result = await db.execute(query)
    
    metadata_list = []
    for stmt, bank_ledger_name in result.all():
        total = len(stmt.raw_data) if isinstance(stmt.raw_data, list) else 0
        
        # --- START: NEW CALCULATION LOGIC ---
        pending_count = 0
        matched_count = 0
        if isinstance(stmt.processed_data, list):
            for t in stmt.processed_data:
                # Count items that are 'Suspense' AND not confirmed by the user
                if t.get("matched_ledger") == "Suspense" and not t.get("user_confirmed"):
                    pending_count += 1
                # Count items that are not 'Suspense'
                if t.get("matched_ledger") != "Suspense":
                    matched_count += 1
        
        status = "Completed" if pending_count == 0 else "Needs Review"
        completion_percentage = 0.0
        if total > 0:
            completion_percentage = round(((total - pending_count) / total) * 100, 2)
        # --- END: NEW CALCULATION LOGIC ---
        # --- THIS IS THE RESTORED LOGIC ---
        statement_period = None
        if stmt.raw_data and stmt.column_mapping.get("date_column"):
            date_column = stmt.column_mapping["date_column"]
            dates = [
                pd.to_datetime(t.get(date_column), dayfirst=True, errors='coerce') 
                for t in stmt.raw_data if t.get(date_column)
            ]
            valid_dates = [d for d in dates if pd.notna(d)]
            if valid_dates:
                min_date = min(valid_dates)
                max_date = max(valid_dates)
                statement_period = f"{min_date.strftime('%d %b %Y')} - {max_date.strftime('%d %b %Y')}"
        # --- END OF RESTORED LOGIC ---

        metadata_list.append(StatementMetadata(
            id=stmt.id,
            filename=stmt.filename,
            upload_date=stmt.upload_date,
            total_transactions=total,
            matched_transactions=matched_count, # Use the calculated matched count
            bank_ledger_name=bank_ledger_name,
            statement_period=statement_period,
            status=status, # Add new status
            completion_percentage=completion_percentage # Add new percentage
        ))
    return metadata_list
# --- END OF REPLACEMENT ---

@api_router.delete("/statements/{statement_id}", status_code=204)
async def delete_statement(statement_id: str, db: AsyncSession = Depends(database.get_db)):
    """Delete a specific bank statement by its ID."""
    statement = await db.get(models.BankStatement, statement_id)
    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")
    
    await db.delete(statement)
    await db.commit()
    return None # Return None for 204 No Content response
# --- END OF ADDITION ---

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
            ledger = transaction.get("matched_ledger", "Suspense")
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


# Devalued Keyword Management
class KeywordCreate(BaseModel):
    keyword: str

@api_router.get("/keywords", response_model=List[str])
async def get_devalued_keywords(db: AsyncSession = Depends(database.get_db)):
    """Get all devalued keywords"""
    result = await db.execute(select(models.DevaluedKeyword.keyword).order_by(models.DevaluedKeyword.keyword))
    return result.scalars().all()

@api_router.post("/keywords")
async def add_devalued_keyword(keyword_data: KeywordCreate, db: AsyncSession = Depends(database.get_db)):
    """Add a new devalued keyword"""
    keyword = keyword_data.keyword.upper().strip()
    if not keyword:
        raise HTTPException(status_code=400, detail="Keyword cannot be empty")
    
    # Check if it already exists
    exists = await db.scalar(select(models.DevaluedKeyword).where(models.DevaluedKeyword.keyword == keyword))
    if exists:
        raise HTTPException(status_code=400, detail="Keyword already exists")

    db_keyword = models.DevaluedKeyword(keyword=keyword)
    db.add(db_keyword)
    await db.commit()
    return {"message": "Keyword added successfully", "keyword": keyword}
    
# --- END OF NEW ENDPOINT BLOCK ---
# --- ADD THIS ENTIRE BLOCK OF NEW ENDPOINTS ---

# Tally Ledger History Management
@api_router.post("/clients/{client_id}/upload-ledger-history")
async def upload_ledger_history(client_id: str, file: UploadFile = File(...), db: AsyncSession = Depends(database.get_db)):
    """
    Parses a Tally Day Book export (Excel format), extracts ledger and narration
    data, and intelligently merges it with existing known ledger data.
    """
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files are supported")

    try:
        content = await file.read()
        df = pd.read_excel(io.BytesIO(content), header=None)

        grouped_samples = defaultdict(list)
        
        # Iterate through the DataFrame rows
        for i, row in df.iterrows():
            # A transaction starts on a row with a valid date in the first column
            if pd.notna(row.iloc[0]) and isinstance(row.iloc[0], (datetime, pd.Timestamp)):
                if i + 1 < len(df): # Ensure there is a next row for the narration
                    
                    ledger_name = str(row.iloc[2]).strip()
                    narration = str(df.iloc[i + 1].iloc[2]).strip()
                    
                    debit_val = pd.to_numeric(row.iloc[4], errors='coerce')
                    credit_val = pd.to_numeric(row.iloc[5], errors='coerce')

                    amount = 0.0
                    trans_type = None

                    if pd.notna(debit_val) and debit_val > 0:
                        amount = debit_val
                        trans_type = "Debit"
                    elif pd.notna(credit_val) and credit_val > 0:
                        amount = credit_val
                        trans_type = "Credit"
                    
                    if ledger_name and narration and trans_type:
                        sample = {
                            "narration": narration,
                            "amount": float(amount),
                            "type": trans_type
                        }
                        grouped_samples[ledger_name].append(sample)
        
        if not grouped_samples:
            raise HTTPException(status_code=400, detail="No valid ledger data found in the uploaded file. Please check the format.")

        # --- Database Merging Logic ---
        MAX_SAMPLES_PER_LEDGER = 25000
        new_ledgers_created = 0
        ledgers_updated = 0

        for ledger_name, new_samples in grouped_samples.items():
            # Find if this ledger already exists for the client
            query = select(models.KnownLedger).where(
                (models.KnownLedger.client_id == client_id) &
                (models.KnownLedger.ledger_name == ledger_name)
            )
            result = await db.execute(query)
            db_ledger = result.scalar_one_or_none()

            if not db_ledger:
                db_ledger = models.KnownLedger(
                    client_id=client_id,
                    ledger_name=ledger_name,
                    samples=[]
                )
                db.add(db_ledger)
                new_ledgers_created += 1
            else:
                ledgers_updated += 1
            
            # Use a set for efficient de-duplication
            existing_fingerprints = {f"{s['narration']}|{s['amount']}|{s['type']}" for s in db_ledger.samples}
            
            unique_new_samples = []
            for sample in new_samples:
                fingerprint = f"{sample['narration']}|{sample['amount']}|{sample['type']}"
                if fingerprint not in existing_fingerprints:
                    unique_new_samples.append(sample)
                    existing_fingerprints.add(fingerprint)
            
            # Combine and apply FIFO cap
            combined_samples = db_ledger.samples + unique_new_samples
            if len(combined_samples) > MAX_SAMPLES_PER_LEDGER:
                combined_samples = combined_samples[-MAX_SAMPLES_PER_LEDGER:] # Keep the newest
            
            db_ledger.samples = combined_samples
        
        await db.commit()

        return {
            "message": "Ledger history uploaded and merged successfully.",
            "ledgers_found": len(grouped_samples),
            "new_ledgers_created": new_ledgers_created,
            "existing_ledgers_updated": ledgers_updated
        }

    except Exception as e:
        await db.rollback()
        logging.error(f"Tally history upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@api_router.get("/clients/{client_id}/known-ledgers", response_model=List[str])
async def get_known_ledgers_for_client(client_id: str, db: AsyncSession = Depends(database.get_db)):
    """Gets a simple, distinct list of known ledger names for a client."""
    query = select(models.KnownLedger.ledger_name).where(models.KnownLedger.client_id == client_id).distinct().order_by(models.KnownLedger.ledger_name)
    result = await db.execute(query)
    return result.scalars().all()

@api_router.get("/clients/{client_id}/ledgers", response_model=PaginatedLedgersResponse)
async def get_paginated_ledgers_for_client(
    client_id: str, 
    page: int = Query(1, ge=1),
    limit: int = Query(12, ge=1, le=100),
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(database.get_db)
):
    """Gets a paginated, searchable list of known ledgers with rule counts."""
    
    # 1. Subquery to count rules per ledger_name for the specific client
    rule_counts_subquery = (
        select(
            models.LedgerRule.ledger_name,
            func.count(models.LedgerRule.id).label("rule_count")
        )
        .where(models.LedgerRule.client_id == client_id)
        .group_by(models.LedgerRule.ledger_name)
        .subquery()
    )

    # 2. Base query to join ledgers with rule counts
    base_query = (
        select(
            models.KnownLedger.id,
            models.KnownLedger.ledger_name,
            func.json_array_length(models.KnownLedger.samples).label("sample_count"),
            func.coalesce(rule_counts_subquery.c.rule_count, 0).label("rule_count")
        )
        .outerjoin(
            rule_counts_subquery,
            models.KnownLedger.ledger_name == rule_counts_subquery.c.ledger_name
        )
        .where(models.KnownLedger.client_id == client_id)
    )

    # 3. Apply search filter if provided
    if search:
        base_query = base_query.where(models.KnownLedger.ledger_name.ilike(f"%{search}%"))

    # 4. Get total count for pagination before applying limit/offset
    count_query = select(func.count()).select_from(base_query.alias())
    total_ledgers = await db.scalar(count_query)
    total_pages = (total_ledgers + limit - 1) // limit

    # 5. Apply sorting, pagination and execute the final query
    final_query = base_query.order_by(models.KnownLedger.ledger_name).limit(limit).offset((page - 1) * limit)
    result = await db.execute(final_query)
    ledgers = result.mappings().all()

    return PaginatedLedgersResponse(
        total_ledgers=total_ledgers,
        total_pages=total_pages,
        ledgers=[KnownLedgerSummaryWithRuleCount(**row) for row in ledgers]
    )



@api_router.get("/clients/{client_id}/known-ledgers/summary", response_model=List[KnownLedgerSummary])
async def get_known_ledgers_summary(client_id: str, db: AsyncSession = Depends(database.get_db)):
    """Gets a summary of all known ledgers for a client, including their sample counts."""
    query = select(
        models.KnownLedger.id,
        models.KnownLedger.ledger_name,
        func.json_array_length(models.KnownLedger.samples).label("sample_count")
    ).where(models.KnownLedger.client_id == client_id).order_by(models.KnownLedger.ledger_name)
    
    result = await db.execute(query)
    # Use .mappings() to get dict-like rows
    return [KnownLedgerSummary(**row) for row in result.mappings().all()]


@api_router.get("/known-ledgers/{ledger_id}/samples", response_model=PaginatedSamplesResponse)
async def get_ledger_samples(ledger_id: str, page: int = Query(1, ge=1), limit: int = Query(100, ge=1, le=500), db: AsyncSession = Depends(database.get_db)):
    """Gets a paginated list of samples for a specific known ledger."""
    db_ledger = await db.get(models.KnownLedger, ledger_id)
    if not db_ledger:
        raise HTTPException(status_code=404, detail="Ledger not found")

    all_samples = db_ledger.samples
    total_samples = len(all_samples)
    
    # Paginate in Python
    start_index = (page - 1) * limit
    end_index = start_index + limit
    paginated_samples = all_samples[start_index:end_index]

    return PaginatedSamplesResponse(
        total_samples=total_samples,
        samples=paginated_samples
    )

@api_router.delete("/known-ledgers/{ledger_id}/samples")
async def delete_ledger_sample(ledger_id: str, sample_to_delete: SampleModel, db: AsyncSession = Depends(database.get_db)):
    """Deletes a specific sample from a known ledger's sample list."""
    db_ledger = await db.get(models.KnownLedger, ledger_id)
    if not db_ledger:
        raise HTTPException(status_code=404, detail="Ledger not found")

    original_count = len(db_ledger.samples)
    
    # Find and remove the matching sample
    # Note: This relies on exact match of the pydantic model
    sample_dict = sample_to_delete.dict()
    updated_samples = [s for s in db_ledger.samples if s != sample_dict]
    
    if len(updated_samples) == original_count:
        raise HTTPException(status_code=404, detail="Sample not found in ledger")

    db_ledger.samples = updated_samples
    await db.commit()

    return {"message": "Sample deleted successfully."}

# --- END OF NEW ENDPOINTS BLOCK ---
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
