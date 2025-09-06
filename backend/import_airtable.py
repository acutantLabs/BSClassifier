import os
import csv
import json
import asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Import our existing models so we can create LedgerRule objects
import models

# --- CONFIGURATION: PLEASE EDIT THIS ---
# 1. Make sure your CSV file is in the 'backend' directory, or provide the full path.
CSV_FILE_PATH = 'airtable_rules.csv' 
# --- END OF CONFIGURATION ---

async def main():
    """
    Main function to read the CSV and import data into PostgreSQL.
    """
    print("--- Starting Airtable Import Script ---")

    # 1. Get the database URL from the .env file
    from dotenv import load_dotenv
    load_dotenv()
    db_url = os.environ.get("POSTGRES_URL")
    if not db_url:
        print("ERROR: POSTGRES_URL not found in .env file. Exiting.")
        return

    # 2. Create a database connection and session
    engine = create_async_engine(db_url)
    AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    
    async with AsyncSessionLocal() as db:
        
        # 3. PRE-FETCH CLIENTS: Get all clients from the database to create a lookup map.
        # This is much faster than querying the DB for every row in the CSV.
        print("Fetching existing clients from the database...")
        result = await db.execute(select(models.Client))
        clients = result.scalars().all()
        client_map = {client.name: client.id for client in clients}
        print(f"Found {len(client_map)} clients: {list(client_map.keys())}")

        # 4. Read the CSV file
        print(f"Reading data from {CSV_FILE_PATH}...")
        try:
            with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rules_to_add = []
                
                for row in reader:
                    account_name = row.get('Account')
                    
                    # 5. FOREIGN KEY LOOKUP: Find the client_id from the account_name.
                    if account_name not in client_map:
                        print(f"WARNING: Skipping row. Client '{account_name}' not found in the database.")
                        continue # Skip this row and move to the next
                    
                    client_id = client_map[account_name]

                    # 6. MERGE COLUMNS: Combine Narration columns into a single JSON array.
                    sample_narrations = []
                    for i in range(1, 7):
                        narration_text = row.get(f'Narration {i}')
                        if narration_text and narration_text.strip():
                            sample_narrations.append(narration_text.strip())
                    
                    # 7. Create the SQLAlchemy object for the new rule
                    new_rule = models.LedgerRule(
                        client_id=client_id,
                        ledger_name=row.get('Ledger name'),
                        regex_pattern=row.get('Regex'),
                        sample_narrations=sample_narrations # The combined list
                    )
                    rules_to_add.append(new_rule)

                if rules_to_add:
                    print(f"Found {len(rules_to_add)} valid rules to import. Adding to session...")
                    db.add_all(rules_to_add)
                    await db.commit()
                    print("--- SUCCESS: All rules have been imported into the database! ---")
                else:
                    print("--- No valid rules found to import. ---")

        except FileNotFoundError:
            print(f"ERROR: The file was not found at '{CSV_FILE_PATH}'. Please check the path.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())