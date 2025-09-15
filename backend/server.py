from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import select, func, outerjoin
from sqlalchemy.orm import selectinload # <-- ADD THIS
from typing import List, Dict, Optional, Any, Union, Tuple



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
    bank_ledger_name: Optional[str] = None # Added field
    statement_period: Optional[str] = None # Added field

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

        df = df.astype(str)

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
            credit_val = pd.to_numeric(row.get(credit_col), errors='coerce')
            debit_val = pd.to_numeric(row.get(debit_col), errors='coerce')
            credit_val = credit_val if pd.notna(credit_val) else 0.0
            debit_val = debit_val if pd.notna(debit_val) else 0.0
            
            # Logic to determine Amount and CR/DR
            if credit_val > 0:
                new_row["Amount (INR)"] = credit_val
                new_row["CR/DR"] = "CR"
            elif debit_val > 0:
                new_row["Amount (INR)"] = debit_val
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
    
    # Map essential columns using the provided mapping
    standardized["Date"] = transaction.get(mapping.get("date_column"))
    standardized["Narration"] = transaction.get(mapping.get("narration_column"))
    standardized["Amount"] = transaction.get(mapping.get("amount_column"))
    standardized["CR/DR"] = transaction.get(mapping.get("crdr_column"))
    
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
    Classify transactions. This endpoint is STATE-AWARE.
    - Default (force_reclassify=False): Fast load. Loads saved state and clusters pending items.
    - Force (force_reclassify=True): Intelligent Merge. Re-runs rules on pending items and merges results.
    """
    statement = await db.get(models.BankStatement, statement_id)

    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")

    mapping = statement.column_mapping
    narration_col = mapping.get("narration_column")
    
    # Step 1: Establish the "base of truth" from the database.
    classified_transactions = statement.processed_data if isinstance(statement.processed_data, list) else []

    # If it's the very first run, build the initial list from raw_data.
    if not classified_transactions:
        # This block is identical to our previous "first run" logic.
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
        await db.commit() # Save the initial state
        await db.refresh(statement)



    # Step 2: If a re-classification is forced, perform the "Intelligent Merge".
    if force_reclassify:
        # A: Identify items that are eligible for re-classification.
        items_to_recheck = [
            t for t in classified_transactions
            if t.get("matched_ledger") == "Suspense" and not t.get("user_confirmed")
        ]
        
        if items_to_recheck:
            # B: Fetch rules and create a map of new matches for efficiency.
            query = select(models.LedgerRule).where(models.LedgerRule.client_id == statement.client_id)
            result = await db.execute(query)
            patterns = result.scalars().all()
            reclassification_updates = {}
            
            for item in items_to_recheck:
                for pattern in patterns:
                    if re.search(pattern.regex_pattern, item['Narration'], re.IGNORECASE):
                        reclassification_updates[item['Narration']] = pattern.ledger_name
                        break # Found a match, move to the next item
            
            # C: Merge the updates back into our main list.
            if reclassification_updates:
                for i, t in enumerate(classified_transactions):
                    if t['Narration'] in reclassification_updates:
                        classified_transactions[i]['matched_ledger'] = reclassification_updates[t['Narration']]
                
                # D: Save the newly merged state back to the database.
                statement.processed_data = classified_transactions
                await db.commit()
                await db.refresh(statement)
                

    # Step 3: Determine what STILL needs to be clustered after any potential updates.
    pending_narrations_set = {
        t['Narration'] for t in classified_transactions
        if t.get("matched_ledger") == "Suspense" and not t.get("user_confirmed")
    }
    transactions_to_cluster = [
        t for t in (statement.raw_data or [])
        if t.get(narration_col) in pending_narrations_set
    ]

    # Step 4: Run the full clustering orchestration on the remaining pending items.
    pre_clusters, remaining_for_ml = await pre_cluster_by_shared_keywords(transactions_to_cluster, narration_col, db)
    # (The rest of the clustering logic remains the same)
    crdr_column = mapping.get("crdr_column")
    debit_transactions = [t for t in remaining_for_ml if str(t.get(crdr_column,"")).strip().upper() == "DR"]
    credit_transactions = [t for t in remaining_for_ml if str(t.get(crdr_column,"")).strip().upper() == "CR"]
    debit_clusters = await cluster_narrations(debit_transactions, narration_col, db)
    credit_clusters = await cluster_narrations(credit_transactions, narration_col, db)
    final_clusters = pre_clusters + debit_clusters + credit_clusters
    
    # Step 5: Return the final state.
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
    """Updates the processed_data for a given statement."""
    statement = await db.get(models.BankStatement, statement_id)
    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")
    
    statement.processed_data = request.processed_data
    await db.commit()
    
    return {"message": "Transactions updated successfully"}
# --- END OF ADDITION ---
# --- ADD THIS ENTIRE ENDPOINT ---

# --- REPLACEMENT for the entire generate_vouchers function ---
@api_router.post("/generate-vouchers/{statement_id}")
async def generate_vouchers(statement_id: str, db: AsyncSession = Depends(database.get_db)):
    """Sorts transactions and generates data for Tally vouchers."""
    statement = await db.get(models.BankStatement, statement_id)
    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")
    if not statement.bank_account_id:
        raise HTTPException(status_code=400, detail="Statement is not linked to a bank account. Please re-upload.")

    bank_account = await db.get(models.BankAccount, statement.bank_account_id)
    if not bank_account:
        raise HTTPException(status_code=404, detail="Bank account not found")

    client = await db.get(models.Client, statement.client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    processed_data = statement.processed_data if statement.processed_data else []
    contra_list = set(bank_account.contra_list or [])
    bank_ledger_name = bank_account.ledger_name
    
    crdr_column = statement.column_mapping.get("crdr_column")
    date_column = statement.column_mapping.get("date_column")

    receipt_vouchers, payment_vouchers, contra_vouchers = [], [], []
    first_date = None

    for transaction in processed_data:
        # --- START: ROBUST CR/DR CHECK ---
        raw_cr_dr = transaction.get(crdr_column)
        
        # Clean the string to handle variations like "Cr.", " cr ", "CR", etc.
        clean_cr_dr = ""
        if isinstance(raw_cr_dr, str):
            clean_cr_dr = raw_cr_dr.strip().replace(".", "").upper()
        # --- END: ROBUST CR/DR CHECK ---

        ledger = transaction.get("matched_ledger", "Suspense")
        
        if not first_date:
            try:
                date_str = transaction.get(date_column)
                if date_str:
                    # Optional cleanup: removed deprecated 'infer_datetime_format'
                    first_date = pd.to_datetime(date_str, dayfirst=True)
            except Exception:
                pass

        # Use the cleaned string for comparison
        if clean_cr_dr == "CR":
            receipt_vouchers.append(transaction)
        elif clean_cr_dr == "DR":
            if ledger in contra_list:
                contra_vouchers.append(transaction)
            else:
                payment_vouchers.append(transaction)

    month_year = first_date.strftime("%B_%Y") if first_date else "vouchers"
    sanitized_ledger_name = bank_ledger_name.replace("/", "_").replace("\\", "_")
    suggested_filename = f"{client.name}_{sanitized_ledger_name}_{month_year}.csv"

    # --- START: MODERN Pydantic USAGE ---
    # Use model_validate (replaces from_orm)
    pydantic_statement = BankStatement.model_validate(statement)
    # --- END: MODERN Pydantic USAGE ---

    return {
        "receipt_vouchers": receipt_vouchers,
        "payment_vouchers": payment_vouchers,
        "contra_vouchers": contra_vouchers,
        "suggested_filename": suggested_filename,
        # Use model_dump (replaces dict)
        "statement_details": pydantic_statement.model_dump(),
        "bank_ledger_name": bank_ledger_name,
    }


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

# --- ADD THIS ENTIRE BLOCK OF CODE ---
@api_router.get("/clients/{client_id}/statements", response_model=List[StatementMetadata])
async def get_client_statements(client_id: str, db: AsyncSession = Depends(database.get_db)):
    """Get all statement metadata for a specific client, including bank account info and period."""
    
    # Query BankStatement and join with BankAccount to get ledger_name
    query = (
        select(models.BankStatement, models.BankAccount.ledger_name)
        .join(models.BankAccount, models.BankStatement.bank_account_id == models.BankAccount.id)
        .where(models.BankStatement.client_id == client_id)
        .order_by(models.BankStatement.upload_date.desc())
    )
    result = await db.execute(query)
    
    metadata_list = []
    for stmt, bank_ledger_name in result.all(): # Fetch both statement object and ledger_name
        total = len(stmt.raw_data) if isinstance(stmt.raw_data, list) else 0
        matched = 0
        if isinstance(stmt.processed_data, list):
            matched = sum(1 for t in stmt.processed_data if t.get("matched_ledger") != "Suspense")
        
        # Determine statement period
        statement_period = None
        if stmt.raw_data and stmt.column_mapping.get("date_column"):
            date_column = stmt.column_mapping["date_column"]
            dates = []
            for transaction in stmt.raw_data:
                date_str = transaction.get(date_column)
                if date_str:
                    try:
                        # Use pandas to_datetime for robust parsing, dayfirst=True is common for Indian statements
                        dt = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
                        if pd.notna(dt):
                            dates.append(dt)
                    except Exception:
                        continue # Skip invalid dates
            
            if dates:
                min_date = min(dates)
                max_date = max(dates)
                statement_period = f"{min_date.strftime('%d %b %Y')} - {max_date.strftime('%d %b %Y')}"

        metadata_list.append(StatementMetadata(
            id=stmt.id,
            filename=stmt.filename,
            upload_date=stmt.upload_date,
            total_transactions=total,
            matched_transactions=matched,
            bank_ledger_name=bank_ledger_name, # Pass the fetched ledger name
            statement_period=statement_period    # Pass the calculated period
        ))
    return metadata_list

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
