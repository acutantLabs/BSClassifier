import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# 1. Read the database connection URL from the .env file.
DATABASE_URL = os.environ.get("POSTGRES_URL")

if not DATABASE_URL:
    raise ValueError("POSTGRES_URL environment variable is not set.")

# 2. Create the core SQLAlchemy engine.
#    The 'asyncpg' part tells SQLAlchemy to use the asyncpg driver.
engine = create_async_engine(DATABASE_URL)

# 3. Create a SessionLocal class.
#    This will be a "factory" for creating new database sessions (conversations).
#    These settings are standard for FastAPI.
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession
)

# 4. Create a dependency function.
#    This is a special function that FastAPI will use to give us a
#    database session for each API request that needs one.
#    It automatically handles opening and closing the connection.
async def get_db():
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()