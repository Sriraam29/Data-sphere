import streamlit as st
import pandas as pd
import os
import json
import re
import datetime
import sqlalchemy
from sqlalchemy import create_engine, inspect, text, MetaData, Table
import pymongo
import pymysql
import psycopg2
import sqlite3
from utils import save_session_state

class DatabaseManager:
    def __init__(self):
        # Session state is fully initialised by initialize_session_state() in utils.py
        # before any module is instantiated. No duplicate defaults needed here.
        pass
            
    def _clear_available_databases(self):
        """Clear the list of available databases when changing database type"""
        if "available_databases" in st.session_state:
            st.session_state.available_databases = []
            
    def _get_available_databases(self, db_type, host, port, username, password):
        """Get list of available databases for the given database type and credentials"""
        try:
            if db_type == "PostgreSQL":
                # Connect to PostgreSQL server without specifying a database
                temp_conn_string = f"postgresql://{username}:{password}@{host}:{port}/postgres"
                engine = create_engine(temp_conn_string)
                try:
                    with engine.connect() as conn:
                        result = conn.execute(sqlalchemy.text("SELECT datname FROM pg_database WHERE datistemplate = false;"))
                        databases = [row[0] for row in result]
                finally:
                    engine.dispose()

                return databases
                
            elif db_type == "MySQL":
                # Connect to MySQL server without specifying a database
                temp_conn_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/"
                engine = create_engine(temp_conn_string)
                try:
                    with engine.connect() as conn:
                        result = conn.execute(sqlalchemy.text("SHOW DATABASES;"))
                        databases = [row[0] for row in result]
                finally:
                    engine.dispose()

                return databases
                
            elif db_type == "MongoDB":
                # Connect to MongoDB server without specifying a database
                if username and password:
                    temp_conn_string = f"mongodb://{username}:{password}@{host}:{port}/"
                else:
                    temp_conn_string = f"mongodb://{host}:{port}/"

                with pymongo.MongoClient(temp_conn_string) as client:
                    databases = client.list_database_names()
                return databases
                
            return []
            
        except Exception as e:
            st.error(f"Failed to list databases: {str(e)}")
            return []
    
    def create_connection_ui(self):
        """Create UI for setting up database connections"""
        st.subheader("Create New Database Connection")
        
        # Connection name
        connection_name = st.text_input("Connection Name", key="new_conn_name")
        
        # Connection method selection
        connection_method = st.radio(
            "Connection Method",
            ["Connect to Server", "Upload Database File"],
            key="connection_method"
        )
        
        # Handle file upload method
        if connection_method == "Upload Database File":
            self._file_upload_ui(connection_name)
            return
        
        # Database type selection
        db_type = st.selectbox(
            "Database Type",
            ["PostgreSQL", "MySQL", "SQLite", "MongoDB"],
            key="new_conn_type",
            on_change=self._clear_available_databases
        )
        
        # Connection details based on database type
        if db_type == "PostgreSQL":
            host = st.text_input("Host", value=os.getenv("PGHOST", "localhost"), key="pg_host")
            port = st.text_input("Port", value=os.getenv("PGPORT", "5432"), key="pg_port")
            username = st.text_input("Username", value=os.getenv("PGUSER", ""), key="pg_user")
            password = st.text_input("Password", type="password", value=os.getenv("PGPASSWORD", ""), key="pg_pass")
            
            # Add a button to list available databases
            if st.button("List Available Databases", key="list_pg_dbs_btn"):
                databases = self._get_available_databases("PostgreSQL", host, port, username, password)
                if databases:
                    # Store in session state
                    st.session_state.available_databases = databases
                    st.success(f"Found {len(databases)} databases")
            
            # Show database selection if available
            if "available_databases" in st.session_state and st.session_state.available_databases:
                col1, col2 = st.columns([3, 1])
                with col1:
                    database = st.selectbox(
                        "Select Database", 
                        options=st.session_state.available_databases,
                        key="pg_db"
                    )
                with col2:
                    if st.button("Clear List", key="clear_pg_dbs_btn"):
                        st.session_state.available_databases = []
                        st.rerun()
            else:
                database = st.text_input("Database", value=os.getenv("PGDATABASE", ""), key="pg_db_input", 
                                        help="Or click 'List Available Databases' to see all databases you can connect to")
            
            connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            
        elif db_type == "MySQL":
            host = st.text_input("Host", value="localhost", key="mysql_host")
            port = st.text_input("Port", value="3306", key="mysql_port")
            username = st.text_input("Username", key="mysql_user")
            password = st.text_input("Password", type="password", key="mysql_pass")
            
            # Add a button to list available databases
            if st.button("List Available Databases", key="list_mysql_dbs_btn"):
                databases = self._get_available_databases("MySQL", host, port, username, password)
                if databases:
                    # Store in session state
                    st.session_state.available_databases = databases
                    st.success(f"Found {len(databases)} databases")
            
            # Show database selection if available
            if "available_databases" in st.session_state and st.session_state.available_databases:
                col1, col2 = st.columns([3, 1])
                with col1:
                    database = st.selectbox(
                        "Select Database", 
                        options=st.session_state.available_databases,
                        key="mysql_db"
                    )
                with col2:
                    if st.button("Clear List", key="clear_mysql_dbs_btn"):
                        st.session_state.available_databases = []
                        st.rerun()
            else:
                database = st.text_input("Database", key="mysql_db_input", 
                                        help="Or click 'List Available Databases' to see all databases you can connect to")
            
            connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
            
        elif db_type == "SQLite":
            database_path = st.text_input("Database File Path", key="sqlite_path")
            connection_string = f"sqlite:///{database_path}"
            
        elif db_type == "MongoDB":
            host = st.text_input("Host", value="localhost", key="mongo_host")
            port = st.text_input("Port", value="27017", key="mongo_port")
            username = st.text_input("Username (optional)", key="mongo_user")
            password = st.text_input("Password (optional)", type="password", key="mongo_pass")
            
            # Add a button to list available databases
            if st.button("List Available Databases", key="list_mongo_dbs_btn"):
                databases = self._get_available_databases("MongoDB", host, port, username, password)
                if databases:
                    # Store in session state
                    st.session_state.available_databases = databases
                    st.success(f"Found {len(databases)} databases")
            
            # Show database selection if available
            if "available_databases" in st.session_state and st.session_state.available_databases:
                col1, col2 = st.columns([3, 1])
                with col1:
                    database = st.selectbox(
                        "Select Database", 
                        options=st.session_state.available_databases,
                        key="mongo_db"
                    )
                with col2:
                    if st.button("Clear List", key="clear_mongo_dbs_btn"):
                        st.session_state.available_databases = []
                        st.rerun()
            else:
                database = st.text_input("Database", key="mongo_db_input", 
                                        help="Or click 'List Available Databases' to see all databases you can connect to")
            
            if username and password:
                connection_string = f"mongodb://{username}:{password}@{host}:{port}/{database}"
            else:
                connection_string = f"mongodb://{host}:{port}/{database}"
        
        # Info message about listing databases
        if db_type in ["PostgreSQL", "MySQL", "MongoDB"] and ("available_databases" not in st.session_state or not st.session_state.available_databases):
            st.info("💡 You can click 'List Available Databases' to see and select from all databases you have access to.")
            
        # Save connection button
        if st.button("Test and Save Connection", key="test_and_save_conn_btn"):
            if not connection_name:
                st.error("Please provide a connection name.")
                return
            
            # Test connection
            try:
                if db_type == "PostgreSQL":
                    engine = create_engine(connection_string)
                    conn = engine.connect()
                    conn.close()
                elif db_type == "MySQL":
                    engine = create_engine(connection_string)
                    conn = engine.connect()
                    conn.close()
                elif db_type == "SQLite":
                    engine = create_engine(connection_string)
                    conn = engine.connect()
                    conn.close()
                elif db_type == "MongoDB":
                    client = pymongo.MongoClient(connection_string)
                    client.server_info()
                    client.close()
                
                # Prepare connection parameters — password stored here, NOT in connection_string
                connection_params = {}
                if db_type == "PostgreSQL":
                    connection_params = {
                        "host": host,
                        "port": port,
                        "database": database,
                        "username": username,
                        "password": password,
                    }
                elif db_type == "MySQL":
                    connection_params = {
                        "host": host,
                        "port": port,
                        "database": database,
                        "username": username,
                        "password": password,
                    }
                elif db_type == "MongoDB":
                    connection_params = {
                        "host": host,
                        "port": port,
                        "database": database,
                        "username": username,
                        "password": password,
                    }

                # Save connection details — no raw connection_string with plaintext password
                st.session_state.db_connections[connection_name] = {
                    "type": db_type,
                    "display_string": self._mask_password_in_connection_string(connection_string),
                    "connection_params": connection_params
                }
                
                # Save session state
                save_session_state()
                
                # Activate the connection
                st.session_state.connected_db = connection_name
                st.session_state.current_connection = st.session_state.db_connections[connection_name]
                
                # Get database schema
                self.get_database_schema()
                
                st.success(f"Connection to {db_type} database successful! Connection saved as '{connection_name}' and activated.")
                
                # Refresh the page to update all components
                st.rerun()
            except Exception as e:
                st.error(f"Failed to connect to the database: {str(e)}")
    
    def manage_connections_ui(self):
        """UI for managing existing database connections"""
        st.subheader("Manage Database Connections")
        
        if not st.session_state.db_connections:
            st.info("No database connections saved. Create a new connection to get started.")
            return
        
        # Display saved connections
        st.write("Saved Connections:")
        
        for conn_name, conn_details in st.session_state.db_connections.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{conn_name}** ({conn_details['type']})")
            
            with col2:
                if st.button("Connect", key=f"connect_{conn_name}"):
                    try:
                        self.connect_to_db(conn_name)
                        st.session_state.connected_db = conn_name
                        st.session_state.current_connection = conn_details
                        
                        # Get database schema
                        self.get_database_schema()
                        
                        st.success(f"Connected to {conn_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to connect: {str(e)}")
            
            with col3:
                if st.button("Delete", key=f"delete_{conn_name}"):
                    if st.session_state.connected_db == conn_name:
                        st.session_state.connected_db = None
                        st.session_state.current_connection = None
                        st.session_state.db_schema = None
                    
                    del st.session_state.db_connections[conn_name]
                    save_session_state()
                    st.success(f"Connection '{conn_name}' deleted.")
                    st.rerun()
        
        # Display current connection status
        if st.session_state.connected_db:
            st.success(f"✅ Currently connected to: {st.session_state.connected_db}")
        else:
            st.warning("⚠️ No active database connection.")
    
    def connect_to_db(self, connection_name):
        """Connect to a specific database by connection name"""
        if connection_name not in st.session_state.db_connections:
            raise ValueError(f"Connection '{connection_name}' not found.")
        
        conn_details = st.session_state.db_connections[connection_name]
        db_type = conn_details["type"]
        connection_string = self._build_connection_string(conn_details)
        
        try:
            if db_type in ["PostgreSQL", "MySQL", "SQLite"]:
                engine = create_engine(connection_string)
                with engine.connect() as conn:
                    # Test connection is successful if we get here
                    pass
            elif db_type == "MongoDB":
                with pymongo.MongoClient(connection_string) as client:
                    client.server_info()
            
            return True
        except Exception as e:
            raise Exception(f"Failed to connect: {str(e)}")
    
    def get_database_schema(self):
        """Get the schema of the currently connected database"""
        if not st.session_state.connected_db or not st.session_state.current_connection:
            return None
        
        db_type = st.session_state.current_connection["type"]
        connection_string = self._build_connection_string(st.session_state.current_connection)
        
        try:
            schema = {}
            
            if db_type in ["PostgreSQL", "MySQL", "SQLite"]:
                engine = create_engine(connection_string)
                inspector = inspect(engine)
                
                schema["tables"] = {}
                for table_name in inspector.get_table_names():
                    columns = inspector.get_columns(table_name)
                    primary_keys = inspector.get_pk_constraint(table_name)
                    foreign_keys = inspector.get_foreign_keys(table_name)
                    
                    schema["tables"][table_name] = {
                        "columns": [{"name": col["name"], "type": str(col["type"])} for col in columns],
                        "primary_keys": primary_keys.get("constrained_columns", []),
                        "foreign_keys": [{
                            "referred_table": fk["referred_table"],
                            "referred_columns": fk["referred_columns"],
                            "constrained_columns": fk["constrained_columns"]
                        } for fk in foreign_keys]
                    }
            
            elif db_type == "MongoDB":
                with pymongo.MongoClient(connection_string) as client:
                    db_name = connection_string.split("/")[-1]
                    db = client[db_name]

                    schema["collections"] = {}
                    for collection_name in db.list_collection_names():
                        # Get a sample document to infer schema
                        sample = db[collection_name].find_one()
                        if sample:
                            schema["collections"][collection_name] = {
                                "fields": [{"name": k, "type": type(v).__name__} for k, v in sample.items()]
                            }
                        else:
                            schema["collections"][collection_name] = {"fields": []}
            
            st.session_state.db_schema = schema
            return schema
            
        except Exception as e:
            st.error(f"Failed to retrieve database schema: {str(e)}")
            return None
    
    def _file_upload_ui(self, connection_name):
        """UI for uploading database files"""
        st.subheader("Upload Database File")
        
        # Validate connection name
        if not connection_name:
            st.warning("Please provide a connection name before uploading a file.")
            return
        
        # File upload
        uploaded_file = st.file_uploader("Upload Database File", type=["db", "sqlite", "sqlite3", "csv", "xlsx", "json"])
        
        if uploaded_file is not None:
            # Determine file type
            file_type = self._get_file_type(uploaded_file)
            st.write(f"Detected file type: {file_type}")
            
            # Configuration based on file type
            if file_type in ["db", "sqlite", "sqlite3"]:
                # For SQLite databases
                db_type = "SQLite"
                db_path = self._save_uploaded_file(uploaded_file)
                connection_string = f"sqlite:///{db_path}"
                
                # Show path where the file was saved
                st.info(f"Database saved at: {db_path}")
                
            elif file_type == "csv":
                # For CSV files
                db_type = "SQLite"
                table_name = st.text_input("Table Name for the CSV Data", value="imported_data")
                
                # Create an in-memory SQLite database from the CSV
                db_path = self._csv_to_sqlite(uploaded_file, table_name)
                connection_string = f"sqlite:///{db_path}"
                
                # Show success message
                st.info(f"CSV data imported into SQLite database as table '{table_name}'")
                
            elif file_type == "xlsx":
                # For Excel files
                db_type = "SQLite"
                sheet_name = st.text_input("Sheet Name (leave blank for first sheet)", value="")
                table_name = st.text_input("Table Name for the Excel Data", value="imported_data")
                
                # Create an in-memory SQLite database from the Excel file
                db_path = self._excel_to_sqlite(uploaded_file, table_name, sheet_name)
                connection_string = f"sqlite:///{db_path}"
                
                # Show success message
                sheet_info = f"sheet '{sheet_name}'" if sheet_name else "first sheet"
                st.info(f"Excel data ({sheet_info}) imported into SQLite database as table '{table_name}'")
                
            elif file_type == "json":
                # For JSON files
                db_type = "SQLite"
                table_name = st.text_input("Table Name for the JSON Data", value="imported_data")
                
                # Create an in-memory SQLite database from the JSON
                db_path = self._json_to_sqlite(uploaded_file, table_name)
                connection_string = f"sqlite:///{db_path}"
                
                # Show success message
                st.info(f"JSON data imported into SQLite database as table '{table_name}'")

            else:
                # Unsupported file type — stop here, don't show Save button
                st.error(
                    f"Unsupported file type: '.{uploaded_file.name.split('.')[-1]}'. "
                    "Please upload a .db, .sqlite, .sqlite3, .csv, .xlsx, or .json file."
                )
                return
            
            # Test and save connection
            if st.button("Save Connection"):
                if not connection_name:
                    st.error("Please provide a connection name.")
                    return
                
                try:
                    # Test the connection
                    engine = create_engine(connection_string)
                    conn = engine.connect()
                    conn.close()
                    
                    # Save connection details — SQLite file paths have no credentials
                    st.session_state.db_connections[connection_name] = {
                        "type": db_type,
                        "connection_string": connection_string,  # safe: SQLite path, no password
                        "display_string": connection_string,
                        "connection_params": {},  # no credentials for file-based connections
                        "details": {
                            "database": db_path,
                            "file_type": file_type,
                            "original_filename": uploaded_file.name
                        }
                    }
                    
                    # Save session state
                    save_session_state()
                    
                    # Activate the connection
                    st.session_state.connected_db = connection_name
                    st.session_state.current_connection = st.session_state.db_connections[connection_name]
                    
                    # Get database schema
                    self.get_database_schema()
                    
                    st.success(f"Database '{connection_name}' successfully configured and connected!")
                    
                    # Refresh the page to update all components
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
    
    def _get_file_type(self, uploaded_file):
        """Determine the type of the uploaded file"""
        # Get file extension
        filename = uploaded_file.name
        file_extension = filename.split(".")[-1].lower()
        
        if file_extension in ["db", "sqlite", "sqlite3"]:
            return "sqlite"
        elif file_extension == "csv":
            return "csv"
        elif file_extension in ["xlsx", "xls"]:
            return "xlsx"
        elif file_extension == "json":
            return "json"
        else:
            return "unknown"
    
    def _save_uploaded_file(self, uploaded_file):
        """Save the uploaded file to disk"""
        # Create a temporary directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), "temp_db")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate a unique filename
        file_extension = uploaded_file.name.split(".")[-1]
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"db_{timestamp}.{file_extension}"
        file_path = os.path.join(temp_dir, filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    
    def _csv_to_sqlite(self, csv_file, table_name):
        """Convert a CSV file to a SQLite database"""
        # Create a temporary directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), "temp_db")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate a unique filename for the SQLite database
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        db_filename = f"csv_import_{timestamp}.db"
        db_path = os.path.join(temp_dir, db_filename)
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Create a SQLite database and save the DataFrame to it
        engine = create_engine(f"sqlite:///{db_path}")
        df.to_sql(table_name, engine, index=False, if_exists="replace")
        
        return db_path
    
    def _excel_to_sqlite(self, excel_file, table_name, sheet_name=None):
        """Convert an Excel file to a SQLite database"""
        # Create a temporary directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), "temp_db")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate a unique filename for the SQLite database
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        db_filename = f"excel_import_{timestamp}.db"
        db_path = os.path.join(temp_dir, db_filename)
        
        # Read the Excel file
        if sheet_name:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
        else:
            df = pd.read_excel(excel_file)
        
        # Create a SQLite database and save the DataFrame to it
        engine = create_engine(f"sqlite:///{db_path}")
        df.to_sql(table_name, engine, index=False, if_exists="replace")
        
        return db_path
    
    def _json_to_sqlite(self, json_file, table_name):
        """Convert a JSON file to a SQLite database"""
        # Create a temporary directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), "temp_db")
        os.makedirs(temp_dir, exist_ok=True)
    
        # Generate a unique filename for the SQLite database
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        db_filename = f"json_import_{timestamp}.db"
        db_path = os.path.join(temp_dir, db_filename)
    
        # Read the JSON file
        try:
            # First, try the simpler method
            df = pd.read_json(json_file)
        except Exception:
            try:
                # Reset file pointer to beginning
                json_file.seek(0)
                content = json_file.read().decode('utf-8')
                if content.strip():  # Check if content is not empty
                    json_data = json.loads(content)
                    if isinstance(json_data, list):
                        df = pd.json_normalize(json_data)
                    else:
                        df = pd.json_normalize([json_data])
                else:
                    st.error("The JSON file appears to be empty.")
                    df = pd.DataFrame()
            except Exception as e:
                st.error(f"Failed to parse JSON file: {str(e)}")
                df = pd.DataFrame({
                    "column1": ["Sample data 1", "Sample data 2"],
                    "column2": [123, 456],
                    "column3": [True, False]
                })
                st.info("Created a sample table since the JSON couldn't be parsed.")
        
        # Serialize list/dict columns to JSON strings
        for col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)

        # Create a SQLite database and save the DataFrame to it
        engine = create_engine(f"sqlite:///{db_path}")
        df.to_sql(table_name, engine, index=False, if_exists="replace")
    
        return db_path

    
    def _split_sql_statements(self, query):
        """Split a SQL string into individual statements, respecting quotes and CTEs.
        
        A simple split on ';' would break strings containing semicolons.
        This parser tracks quote state so it only splits on top-level semicolons.
        """
        statements = []
        current = []
        in_single_quote = False
        in_double_quote = False
        i = 0
        while i < len(query):
            char = query[i]
            
            # Handle escape sequences inside quotes
            if char == '\\' and i + 1 < len(query) and (in_single_quote or in_double_quote):
                current.append(char)
                current.append(query[i + 1])
                i += 2
                continue
            
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            
            if char == ';' and not in_single_quote and not in_double_quote:
                stmt = ''.join(current).strip()
                if stmt:
                    statements.append(stmt)
                current = []
            else:
                current.append(char)
            i += 1
        
        # Don't forget the last statement (may not end with ;)
        last = ''.join(current).strip()
        if last:
            statements.append(last)
        
        return statements

    def execute_query(self, query):
        """Execute a SQL query against the currently connected database.
        
        Supports multiple SQL statements separated by semicolons.
        For multi-statement queries, each statement is executed in order.
        The result of the last SELECT-like statement is returned as a DataFrame.
        """
        if not st.session_state.connected_db or not st.session_state.current_connection:
            st.error("No active database connection.")
            return None
        
        db_type = st.session_state.current_connection["type"]
        connection_string = self._build_connection_string(st.session_state.current_connection)
        
        try:
            # Log query to history
            query_item = {
                "query": query,
                "database": st.session_state.connected_db,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.session_state.query_history.insert(0, query_item)
            
            # Limit history to 20 items
            if len(st.session_state.query_history) > 20:
                st.session_state.query_history = st.session_state.query_history[:20]
            
            save_session_state()
            
            # Execute query based on database type
            if db_type in ["PostgreSQL", "MySQL", "SQLite"]:
                engine = create_engine(connection_string)
                
                # Handle SQLite-specific queries
                if db_type == "SQLite" and "INFORMATION_SCHEMA.TABLES" in query:
                    if "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES" in query:
                        modified_query = "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                        with engine.connect() as conn:
                            result = pd.read_sql(modified_query, conn)
                            st.session_state.query_results = result
                            return result
                    else:
                        modified_query = query.replace("INFORMATION_SCHEMA.TABLES", "sqlite_master WHERE type='table'")
                        with engine.connect() as conn:
                            result = pd.read_sql(modified_query, conn)
                            st.session_state.query_results = result
                            return result
                
                # Split into individual statements for multi-statement support
                statements = self._split_sql_statements(query)
                
                if not statements:
                    st.warning("No SQL statements found to execute.")
                    return None
                
                # For a single statement, use the fast path
                if len(statements) == 1:
                    stmt = statements[0]
                    if stmt.strip().upper().startswith(("SELECT", "SHOW", "DESCRIBE", "EXPLAIN", "WITH", "PRAGMA")):
                        with engine.connect() as conn:
                            result = pd.read_sql(stmt, conn)
                        st.session_state.query_results = result
                        engine.dispose()
                        return result
                    else:
                        with engine.begin() as conn:
                            result = conn.execute(text(stmt))
                        affected = result.rowcount if result.rowcount != -1 else 0
                        msg = f"Query executed successfully. {affected} row(s) affected."
                        st.session_state.query_results = msg
                        engine.dispose()
                        return msg
                
                # Multi-statement execution
                last_select_result = None
                total_affected = 0
                executed_count = 0
                
                with engine.begin() as conn:
                    for i, stmt in enumerate(statements):
                        stmt_stripped = stmt.strip()
                        if not stmt_stripped:
                            continue
                        
                        stmt_upper = stmt_stripped.upper()
                        is_select = stmt_upper.startswith(("SELECT", "SHOW", "DESCRIBE", "EXPLAIN", "WITH", "PRAGMA"))
                        
                        try:
                            if is_select:
                                df = pd.read_sql(stmt_stripped, conn)
                                last_select_result = df
                            else:
                                result = conn.execute(text(stmt_stripped))
                                affected = result.rowcount if result.rowcount != -1 else 0
                                total_affected += affected
                            executed_count += 1
                        except Exception as stmt_error:
                            error_msg = f"Statement {i + 1} failed: {str(stmt_error)}"
                            st.session_state.last_query_error = error_msg
                            st.error(error_msg)
                            engine.dispose()
                            return None
                
                engine.dispose()
                
                # Return the last SELECT result if there was one
                if last_select_result is not None:
                    st.session_state.query_results = last_select_result
                    return last_select_result
                else:
                    msg = f"All {executed_count} statement(s) executed successfully. {total_affected} total row(s) affected."
                    st.session_state.query_results = msg
                    return msg
            
            elif db_type == "MongoDB":
                # Parse MongoDB query from SQL-like syntax
                try:
                    with pymongo.MongoClient(connection_string) as client:
                        db_name = connection_string.split("/")[-1]
                        db = client[db_name]
                        
                        # Parse the query to determine operation type
                        query_parts = query.strip().split(" ", 2)
                        operation = query_parts[0].lower()
                        
                        if operation == "find":
                            collection_name = query_parts[1]
                            filter_json = "{}" if len(query_parts) < 3 else query_parts[2]
                            filter_dict = json.loads(filter_json)
                            result = list(db[collection_name].find(filter_dict))
                            
                        elif operation == "aggregate":
                            collection_name = query_parts[1]
                            pipeline_json = query_parts[2]
                            pipeline = json.loads(pipeline_json)
                            result = list(db[collection_name].aggregate(pipeline))
                            
                        elif operation == "insert":
                            collection_name = query_parts[1]
                            document_json = query_parts[2]
                            document = json.loads(document_json)
                            result = db[collection_name].insert_one(document)
                            return f"Document inserted with ID: {result.inserted_id}"
                            
                        elif operation == "update":
                            collection_name = query_parts[1]
                            update_parts = query_parts[2].split(" ", 2)
                            filter_json = update_parts[0]
                            update_json = update_parts[1]
                            filter_dict = json.loads(filter_json)
                            update_dict = json.loads(update_json)
                            result = db[collection_name].update_many(filter_dict, update_dict)
                            return f"Updated {result.modified_count} documents"
                            
                        elif operation == "delete":
                            collection_name = query_parts[1]
                            filter_json = query_parts[2]
                            filter_dict = json.loads(filter_json)
                            result = db[collection_name].delete_many(filter_dict)
                            return f"Deleted {result.deleted_count} documents"
                            
                        else:
                            st.error(f"Unsupported MongoDB operation: {operation}")
                            return None
                            
                        # Convert MongoDB result to DataFrame for find and aggregate
                        if operation in ["find", "aggregate"]:
                            if result:
                                df = pd.DataFrame(result)
                                if "_id" in df.columns:
                                    df = df.drop("_id", axis=1)
                                st.session_state.query_results = df
                                return df
                            else:
                                st.session_state.query_results = pd.DataFrame()
                                return pd.DataFrame()
                except Exception as e:
                    st.error(f"MongoDB operation failed: {str(e)}")
                    return f"Error: {str(e)}"
        
        except Exception as e:
            error_message = f"Query execution failed: {str(e)}"
            # Store raw error so NLP layer can pass it to the LLM for auto-fix
            st.session_state.last_query_error = error_message
            st.error(error_message)
            return None
            
    def _mask_password_in_connection_string(self, connection_string):
        """Replace actual password with placeholder in connection string for display"""
        return re.sub(r':(.*?)@', ':********@', connection_string)

    def _build_connection_string(self, conn_details):
        """Reconstruct the connection string from stored params at runtime.
        Passwords are stored in connection_params and never in the top-level
        session state dict. For SQLite/file-based connections the string is
        credential-free and is returned as-is.
        """
        db_type = conn_details.get("type", "")
        params = conn_details.get("connection_params", {})

        if db_type == "SQLite" or not params:
            return conn_details.get("connection_string", "")

        host = params.get("host", "localhost")
        port = params.get("port", "")
        database = params.get("database", "")
        username = params.get("username", "")
        password = params.get("password", "")

        if db_type == "PostgreSQL":
            pw = password or os.getenv("PGPASSWORD", "")
            return f"postgresql://{username}:{pw}@{host}:{port}/{database}"
        elif db_type == "MySQL":
            return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == "MongoDB":
            if username and password:
                return f"mongodb://{username}:{password}@{host}:{port}/{database}"
            return f"mongodb://{host}:{port}/{database}"

        return conn_details.get("connection_string", "")