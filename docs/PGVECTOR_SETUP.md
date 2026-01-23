# PostgreSQL + pgvector Setup Guide

This guide explains how to configure Alpha Gen to use PostgreSQL with pgvector extension for persistent vector storage.

## Why pgvector?

- ✅ **True Persistence** - Data stored in your PostgreSQL database
- ✅ **ACID Compliance** - Transactions, backups, replication
- ✅ **Production Ready** - Battle-tested, mature ecosystem
- ✅ **Cost Effective** - No separate vector database service needed
- ✅ **Easy Integration** - Works with existing Postgres infrastructure

## Prerequisites

- PostgreSQL 11+ installed
- `pgvector` extension installed

## Installation

### 1. Install pgvector Extension

#### On Ubuntu/Debian:
```bash
sudo apt install postgresql-15-pgvector
```

#### On macOS (Homebrew):
```bash
brew install pgvector
```

#### From Source:
```bash
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### 2. Enable pgvector in Your Database

```sql
-- Connect to your database
psql -U your_user -d your_database

-- Enable the extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

## Configuration

### 1. Set Environment Variables

Update your `.env` file:

```bash
# Vector Store Configuration
VECTOR_STORE_PROVIDER=pgvector

# PostgreSQL Connection
POSTGRES_URL=postgresql://username:password@localhost:5432/alpha_gen

# Common Settings
VECTOR_STORE_COLLECTION=alpha_gen_docs
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 2. Connection String Format

```
postgresql://[user]:[password]@[host]:[port]/[database]
```

**Examples:**

```bash
# Local development
POSTGRES_URL=postgresql://postgres:password@localhost:5432/alpha_gen

# Remote server
POSTGRES_URL=postgresql://user:pass@db.example.com:5432/production_db

# With SSL
POSTGRES_URL=postgresql://user:pass@db.example.com:5432/db?sslmode=require

# Cloud providers
# AWS RDS
POSTGRES_URL=postgresql://admin:pass@mydb.abc123.us-east-1.rds.amazonaws.com:5432/alpha_gen

# Google Cloud SQL
POSTGRES_URL=postgresql://user:pass@/alpha_gen?host=/cloudsql/project:region:instance

# Azure Database
POSTGRES_URL=postgresql://user@server:pass@server.postgres.database.azure.com:5432/alpha_gen
```

## Database Setup

### Create Database and User

```sql
-- Create user first
CREATE USER alpha_gen_user WITH PASSWORD 'secure_password';

-- Create database with user as owner
CREATE DATABASE alpha_gen OWNER alpha_gen_user;

-- Connect to the database
\c alpha_gen

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant schema permissions (if needed)
GRANT ALL ON SCHEMA public TO alpha_gen_user;
```

### Verify Setup

```sql
-- Check extension
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Check tables (after first run)
\dt

-- View vector store collection
SELECT collection_name, COUNT(*) as doc_count
FROM langchain_pg_collection
GROUP BY collection_name;
```

## Usage

### Switching Between Chroma and pgvector

Alpha Gen automatically uses the provider specified in your environment:

```bash
# Use Chroma (default)
VECTOR_STORE_PROVIDER=chroma

# Use pgvector
VECTOR_STORE_PROVIDER=pgvector
POSTGRES_URL=postgresql://user:pass@localhost:5432/alpha_gen
```

### Testing the Connection

```python
from alpha_gen.core.rag import get_vector_store_manager

# This will use pgvector if configured
manager = get_vector_store_manager()

# Add test document
manager.add_documents(
    texts=["Test document"],
    metadatas=[{"source": "test"}]
)

# Search
results = manager.similarity_search("test", k=1)
print(results)
```

## Performance Tuning

### Create Indexes

For better performance with large datasets:

```sql
-- Create HNSW index (recommended for most cases)
CREATE INDEX ON langchain_pg_embedding 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Or IVFFlat index (faster build, slower search)
CREATE INDEX ON langchain_pg_embedding 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### Connection Pooling

For production, use connection pooling:

```bash
# Install pgbouncer
sudo apt install pgbouncer

# Configure in .env
POSTGRES_URL=postgresql://user:pass@localhost:6432/alpha_gen
```

## Backup and Restore

### Backup

```bash
# Full database backup
pg_dump -U alpha_gen_user alpha_gen > backup.sql

# Just vector data
pg_dump -U alpha_gen_user -t langchain_pg_* alpha_gen > vectors_backup.sql
```

### Restore

```bash
# Restore full database
psql -U alpha_gen_user alpha_gen < backup.sql

# Restore just vectors
psql -U alpha_gen_user alpha_gen < vectors_backup.sql
```

## Monitoring

### Check Vector Store Size

```sql
-- Total size
SELECT 
    pg_size_pretty(pg_total_relation_size('langchain_pg_embedding')) as total_size,
    COUNT(*) as document_count
FROM langchain_pg_embedding;

-- By collection
SELECT 
    c.name as collection,
    COUNT(e.*) as documents,
    pg_size_pretty(pg_total_relation_size('langchain_pg_embedding')) as size
FROM langchain_pg_collection c
LEFT JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
GROUP BY c.name;
```

### Query Performance

```sql
-- Enable timing
\timing on

-- Test similarity search
SELECT embedding <=> '[0.1, 0.2, ...]'::vector as distance
FROM langchain_pg_embedding
ORDER BY distance
LIMIT 10;
```

## Troubleshooting

### Extension Not Found

```
ERROR: extension "vector" is not available
```

**Solution:** Install pgvector extension (see Installation section)

### Connection Refused

```
ERROR: could not connect to server: Connection refused
```

**Solution:** Check PostgreSQL is running and connection string is correct

```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -U your_user -d your_database -h localhost
```

### Permission Denied

```
ERROR: permission denied for schema public
```

**Solution:** Grant proper permissions

```sql
GRANT ALL ON SCHEMA public TO your_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO your_user;
```

### Slow Queries

**Solution:** Create appropriate indexes (see Performance Tuning section)

## Migration from Chroma

To migrate existing Chroma data to pgvector:

```python
from alpha_gen.core.rag import get_vector_store_manager
from alpha_gen.core.config.settings import get_config

# 1. Export from Chroma
config = get_config()
config.vector_store.provider = "chroma"
chroma_manager = get_vector_store_manager()

# Get all documents (implement based on your needs)
# documents = chroma_manager.vector_store.get()

# 2. Import to pgvector
config.vector_store.provider = "pgvector"
pgvector_manager = get_vector_store_manager()

# Add documents
# pgvector_manager.add_documents(texts, metadatas)
```

## Best Practices

1. **Use Connection Pooling** - For production deployments
2. **Create Indexes** - After loading initial data
3. **Regular Backups** - Use pg_dump for backups
4. **Monitor Performance** - Track query times and index usage
5. **Secure Connections** - Use SSL for remote connections
6. **Environment Variables** - Never commit credentials to git

## Resources

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [LangChain pgvector Integration](https://python.langchain.com/docs/integrations/vectorstores/pgvector)

## Support

For issues specific to Alpha Gen's pgvector integration, please open an issue on GitHub.
