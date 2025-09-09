from sqlalchemy import Column, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime, timezone
import uuid

# Create a Base class for our models to inherit from
Base = declarative_base()

# This model represents your clients (e.g., "Aurum", "Texas Hospicare")
class Client(Base):
    __tablename__ = 'clients'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True, index=True)
    # Add the new suspense ledger field with a sensible default
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Define the relationship to the new BankAccount model
    bank_accounts = relationship("BankAccount", back_populates="client", cascade="all, delete-orphan")
    ledger_rules = relationship("LedgerRule", back_populates="client", cascade="all, delete-orphan")

# This model is a direct replacement for each record in your Airtable.
class LedgerRule(Base):
    __tablename__ = 'ledger_rules'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    ledger_name = Column(String, nullable=False, index=True)
    regex_pattern = Column(String, nullable=False)
    sample_narrations = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    # Add the new updated_at field to track changes to the rule
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    client_id = Column(String, ForeignKey('clients.id'), nullable=False)
    client = relationship("Client", back_populates="ledger_rules")

    # --- ADD THIS CLASS to models.py ---
class TempFile(Base):
    __tablename__ = 'temp_files'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    headers = Column(JSON, nullable=False)
    raw_data = Column(JSON, nullable=False)
    upload_time = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # --- ADD THIS CLASS to models.py ---
class BankStatement(Base):
    __tablename__ = 'bank_statements'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String, ForeignKey('clients.id'), nullable=False)
    filename = Column(String, nullable=False)
    upload_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    column_mapping = Column(JSON, nullable=False)
    statement_format = Column(String, nullable=False)
    raw_data = Column(JSON, nullable=False)
    processed_data = Column(JSON, nullable=True)

    client = relationship("Client")

class BankAccount(Base):
    __tablename__ = 'bank_accounts'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    bank_name = Column(String, nullable=False)
    ledger_name = Column(String, nullable=False, unique=True) # e.g., "IDBI Bank_12467"
    contra_list = Column(JSON, nullable=True) # Stores a list of strings
    filter_list = Column(JSON, nullable=True) # Stores a list of strings
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Define the foreign key to link this account to a client
    client_id = Column(String, ForeignKey('clients.id'), nullable=False)
    client = relationship("Client", back_populates="bank_accounts")
# --- END OF NEW CLASS ---