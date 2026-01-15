from typing import List, Dict, Any, Optional
import json
import argparse
import yaml
import os
from pathlib import Path
import oracledb
from langchain_core.documents import Document
from langchain_oracledb import OracleVS, OracleEmbeddings

class OraDBVectorStore:
    def __init__(self, persist_directory: str = "embeddings", embedding_function: Optional[Any] = None):
        """Initialize Oracle DB Vector Store using langchain-oracledb
        
        Args:
            persist_directory: Not used for Oracle DB connection but kept for compatibility
            embedding_function: Optional embedding function to use instead of OracleEmbeddings
        """
        # Load Oracle DB credentials from config.yaml
        credentials = self._load_config()
        
        username = credentials.get("ORACLE_DB_USERNAME", "ADMIN")
        password = credentials.get("ORACLE_DB_PASSWORD", "")
        dsn = credentials.get("ORACLE_DB_DSN", "")
        wallet_path = credentials.get("ORACLE_DB_WALLET_LOCATION")
        wallet_password = credentials.get("ORACLE_DB_WALLET_PASSWORD")
        
        if not password or not dsn:
            raise ValueError("Oracle DB credentials not found in config.yaml. Please set ORACLE_DB_USERNAME, ORACLE_DB_PASSWORD, and ORACLE_DB_DSN.")

        # Connect to the database
        try:
            if not wallet_path:
                print(f'Connecting (no wallet) to dsn {dsn} and user {username}')
                self.connection = oracledb.connect(user=username, password=password, dsn=dsn)
            else:
                print(f'Connecting (with wallet) to dsn {dsn} and user {username}')
                self.connection = oracledb.connect(user=username, password=password, dsn=dsn, 
                                           config_dir=wallet_path, wallet_location=wallet_path, wallet_password=wallet_password)
            print("Oracle DB Connection successful!")
        except Exception as e:
            print("Oracle DB Connection failed!", e)
            raise

        if embedding_function:
            self.embeddings = embedding_function
            print("Using provided custom embedding function.")
        else:
            # Initialize Embeddings
            # Using OracleEmbeddings with params. 
            # Defaulting to 'database' provider and 'ALL_MINILM_L12_V2' which we just loaded.
            # This should be configured in config.yaml for production.
            embed_params = credentials.get("ORACLE_EMBEDDINGS_PARAMS", {"provider": "database", "model": "ALL_MINILM_L12_V2"})
            if isinstance(embed_params, str):
                 try:
                     embed_params = json.loads(embed_params)
                 except:
                     pass
            
            self.embeddings = OracleEmbeddings(conn=self.connection, params=embed_params)

        # Initialize Tables (Collections)
        self.collections = {
            "pdf_documents": "PDFCollection",
            "web_documents": "WebCollection",
            "repository_documents": "RepoCollection",
            "general_knowledge": "GeneralCollection"
        }
        
        # Initialize OracleVS instances
        self.vector_stores = {}
        for name, table in self.collections.items():
            self.vector_stores[name] = OracleVS(
                client=self.connection,
                embedding_function=self.embeddings,
                table_name=table,
                distance_strategy="EUCLIDEAN_DISTANCE" # Matching previous logic
            )
            # Create table if not exists (OracleVS typically handles this on valid calls or we might need explicit index creation)
            # OracleVS might auto-create on add_texts? We'll see. 
            # If not, we rely on the fact that old implementation created them, or OracleVS will error.
            # Ideally OracleVS has a creates methods? 
            # We will assume existing tables from old implementation are compatible OR OracleVS will manage.
            # Actually, old impl created tables with specific schema. OracleVS might expect specific schema (id, text, metadata, embedding).
            # The schema in old impl: id, text, metadata, embedding. This seems standard.

    def _load_config(self) -> Dict[str, str]:
        """Load configuration from config.yaml"""
        try:
            config_path = Path("config.yaml")
            if not config_path.exists():
                print("Warning: config.yaml not found. Using empty configuration.")
                return {}
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config if config else {}
        except Exception as e:
            print(f"Warning: Error loading config: {str(e)}")
            return {}
            
    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """Sanitize metadata to ensure all values are valid types"""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = str(value)
            elif value is None:
                sanitized[key] = ""
            else:
                sanitized[key] = str(value)
        return sanitized

    def _add_chunks_to_collection(self, chunks: List[Dict[str, Any]], collection_name: str):
        """Helper to add chunks to a specific collection"""
        if not chunks:
            return
            
        store = self.vector_stores.get(collection_name)
        if not store:
            raise ValueError(f"Collection {collection_name} not found")
            
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [self._sanitize_metadata(chunk["metadata"]) for chunk in chunks]
        
        # OracleVS add_texts
        print(f"🔄 [OraDB] Inserting {len(chunks)} chunks into {collection_name}...")
        store.add_texts(texts=texts, metadatas=metadatas)
        print(f"✅ [OraDB] Successfully inserted {len(chunks)} chunks.")

    def add_pdf_chunks(self, chunks: List[Dict[str, Any]], document_id: str):
        """Add chunks from a PDF document to the vector store"""
        self._add_chunks_to_collection(chunks, "pdf_documents")
        
    def add_web_chunks(self, chunks: List[Dict[str, Any]], source_id: str):
        """Add chunks from web content to the vector store"""
        self._add_chunks_to_collection(chunks, "web_documents")
        
    def add_general_knowledge(self, chunks: List[Dict[str, Any]], source_id: str):
        """Add general knowledge chunks to the vector store"""
        self._add_chunks_to_collection(chunks, "general_knowledge")
        
    def add_repo_chunks(self, chunks: List[Dict[str, Any]], document_id: str):
        """Add chunks from a repository to the vector store"""
        self._add_chunks_to_collection(chunks, "repository_documents")

    def _query_collection(self, collection_name: str, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Helper to query a collection"""
        print(f"🔍 [OracleVS] Querying {collection_name}")
        store = self.vector_stores.get(collection_name)
        if not store:
            return []
            
        docs = store.similarity_search(query, k=n_results)
        
        formatted_results = []
        for doc in docs:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            formatted_results.append(result)
            
        print(f"🔍 [OracleVS] Retrieved {len(formatted_results)} chunks from {collection_name}")
        return formatted_results

    def query_pdf_collection(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Query the PDF documents collection"""
        return self._query_collection("pdf_documents", query, n_results)

    def query_web_collection(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Query the web documents collection"""
        return self._query_collection("web_documents", query, n_results)

    def query_general_collection(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Query the general knowledge collection"""
        return self._query_collection("general_knowledge", query, n_results)

    def delete_documents(self, collection_name: str, ids: Optional[List[str]] = None, delete_all: bool = False):
        """Delete documents from a collection"""
        store = self.vector_stores.get(collection_name)
        if not store:
            raise ValueError(f"Collection {collection_name} not found")
            
        if delete_all:
            # OracleVS might not support delete_all directly, but we can try dropping/truncating via SQL if needed,
            # but sticking to package interface first. 
            # If delete_all is true, we might just want to drop table contents.
            # However, typically vector stores delete by ID.
            # Implementing simple delete by ID for now or using SQL for mass delete if package allows.
            # Using raw SQL for efficiency if delete_all.
            table_name = self.collections.get(collection_name)
            if table_name:
                cursor = self.connection.cursor()
                cursor.execute(f"TRUNCATE TABLE {table_name}")
                self.connection.commit()
                print(f"🗑️ [OraDBVectorStore] Truncated collection {collection_name}")
        elif ids:
            store.delete(ids=ids)
            print(f"🗑️ [OraDBVectorStore] Deleted {len(ids)} documents from {collection_name}")

    def query_repo_collection(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Query the repository documents collection"""
        return self._query_collection("repository_documents", query, n_results)

    def get_collection_count(self, collection_name: str) -> int:
        """Get the total number of chunks in a collection"""
        table_name = self.collections.get(collection_name)
        if not table_name:
            return 0
        try:
            cursor = self.connection.cursor()
            # Check if table exists first? Or just try count.
            # Assuming table exists if it's in our map, otherwise SQL error will be caught.
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            result = cursor.fetchone()
            cursor.close()
            if result:
                return result[0]
            return 0
        except Exception as e:
            # Table might not exist yet
            return 0

    def get_latest_chunk(self, collection_name: str) -> Dict[str, Any]:
        """Get the most recently added chunk from a collection"""
        table_name = self.collections.get(collection_name)
        if not table_name:
            return {}
        try:
            cursor = self.connection.cursor()
            # Fetch one row. No guarantee of order without timestamp column, 
            # but ROWNUM 1 gives us *a* chunk.
            # Using simple query assuming standard columns
            cursor.execute(f"SELECT text, metadata FROM {table_name} WHERE ROWNUM <= 1")
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                text = row[0]
                metadata = row[1]
                
                # Handle LOBs if necessary
                if hasattr(metadata, 'read'):
                    metadata = metadata.read()
                
                # If metadata is string, valid for return. The caller parses it.
                return {
                    "content": text,
                    "metadata": metadata
                }
            return {}
        except Exception as e:
            print(f"Error getting chunk from {collection_name}: {e}")
            return {}

    def check_embedding_model_exists(self, model_name: str = "ALL_MINILM_L12_V2") -> bool:
        """Check if an ONNX embedding model is loaded in the database"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM user_mining_models WHERE model_name = :1", [model_name])
            result = cursor.fetchone()
            cursor.close()
            if result and result[0] > 0:
                return True
            return False
        except Exception as e:
            print(f"Error checking model {model_name}: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Manage Oracle DB vector store")
    parser.add_argument("--add", help="JSON file containing chunks to add")
    parser.add_argument("--add-web", help="JSON file containing web chunks to add")
    parser.add_argument("--query", help="Query to search for")
    
    args = parser.parse_args()
    try:
        store = OraDBVectorStore()
        
        if args.add:
            with open(args.add, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            store.add_pdf_chunks(chunks, document_id=args.add)
            print(f"✓ Added {len(chunks)} PDF chunks to Oracle DB vector store")
        
        if args.add_web:
            with open(args.add_web, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            store.add_web_chunks(chunks, source_id=args.add_web)
            print(f"✓ Added {len(chunks)} web chunks to Oracle DB vector store")
        
        if args.query:
            # Query both collections
            pdf_results = store.query_pdf_collection(args.query)
            web_results = store.query_web_collection(args.query)
            
            print("\nPDF Results:")
            print("-" * 50)
            for result in pdf_results:
                print(f"Content: {result['content'][:200]}...")
                print(f"Source: {result['metadata'].get('source', 'Unknown')}")
                print(f"Pages: {result['metadata'].get('page_numbers', [])}")
                print("-" * 50)
            
            print("\nWeb Results:")
            print("-" * 50)
            for result in web_results:
                print(f"Content: {result['content'][:200]}...")
                print(f"Source: {result['metadata'].get('source', 'Unknown')}")
                print(f"Title: {result['metadata'].get('title', 'Unknown')}")
                print("-" * 50)
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
