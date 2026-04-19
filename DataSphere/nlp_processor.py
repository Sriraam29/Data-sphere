import streamlit as st
import os
import requests
import json
import re
import pandas as pd
try:
    import sqlparse
except ImportError:
    sqlparse = None
try:
    import google.genai as genai
    GENAI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai
        GENAI_AVAILABLE = True
    except ImportError:
        GENAI_AVAILABLE = False
        genai = None
from groq import Groq

# Groq model — update here to change it across the entire module
GROQ_MODEL = "llama-3.1-8b-instant"

# Maximum number of messages kept in chat/Q&A histories
# Older messages beyond this limit are trimmed to control API token usage
MAX_CHAT_HISTORY = 20
MAX_QA_HISTORY = 20

class NLPProcessor:
    def __init__(self):
        # Initialize session state for API keys if not already set
        if "groq_api_key" not in st.session_state:
            st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")
        
        if "gemini_api_key" not in st.session_state:
            st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        
        # Initialize Groq client if API key is available
        self.groq_client = None
        self.gemini_model = None
        self._update_clients()
        
        # Initialize session state variables
        if "current_query" not in st.session_state:
            st.session_state.current_query = ""
        
        if "natural_language_query" not in st.session_state:
            st.session_state.natural_language_query = ""
            
        # Initialize chat history for follow-up questions
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        # Initialize database Q&A history
        if "db_qa_history" not in st.session_state:
            st.session_state.db_qa_history = []
            
        # Initialize active conversation flag
        if "active_conversation" not in st.session_state:
            st.session_state.active_conversation = False
    
    def _update_clients(self):
        """Update API clients based on current session state API keys"""
        # Update Groq client
        if st.session_state.groq_api_key:
            self.groq_client = Groq(api_key=st.session_state.groq_api_key)
        else:
            self.groq_client = None
            
        # Update Gemini client
        if st.session_state.gemini_api_key and GENAI_AVAILABLE:
            try:
                # Try new google.genai package first
                self.gemini_client = genai.Client(api_key=st.session_state.gemini_api_key)
                self.use_new_genai = True
            except AttributeError:
                # Fall back to old google.generativeai package
                genai.configure(api_key=st.session_state.gemini_api_key)
                self.gemini_client = genai
                self.use_new_genai = False
        else:
            self.gemini_client = None
            self.use_new_genai = False
    
    def text_to_sql_ui(self, db_manager):
        """UI for converting natural language to SQL with follow-up chat"""
        st.subheader("Convert Natural Language to SQL")
        
        # Create tabs for different query modes
        query_tab, chat_tab, db_qa_tab = st.tabs(["Query Generator", "Follow-up Chat", "Database Q&A"])
        
        with query_tab:
            # Select LLM model
            model_provider = st.selectbox(
                "Select AI Model Provider", 
                options=["Groq", "Gemini"],
                key="model_provider"
            )
            
            # API Key Configuration
            with st.expander("API Key Configuration", expanded=not (st.session_state.groq_api_key or st.session_state.gemini_api_key)):
                st.info("You need to provide API keys to use the text-to-SQL feature. These keys are stored only in your current session.")
                
                # Groq API Key
                groq_api_key = st.text_input(
                    "Groq API Key", 
                    value=st.session_state.groq_api_key,
                    type="password",
                    help="Get your Groq API key from https://console.groq.com/keys",
                    key="groq_api_input"
                )
                
                # Gemini API Key
                gemini_api_key = st.text_input(
                    "Gemini API Key", 
                    value=st.session_state.gemini_api_key,
                    type="password",
                    help="Get your Gemini API key from https://ai.google.dev/",
                    key="gemini_api_input"
                )
                
                # Update API keys in session state
                if groq_api_key != st.session_state.groq_api_key:
                    st.session_state.groq_api_key = groq_api_key
                    self._update_clients()
                    
                if gemini_api_key != st.session_state.gemini_api_key:
                    st.session_state.gemini_api_key = gemini_api_key
                    self._update_clients()
            
            # Show API status
            if model_provider == "Groq":
                if not st.session_state.groq_api_key:
                    st.warning("Groq API key is not configured. Please enter your API key in the configuration section.")
            elif model_provider == "Gemini":
                if not st.session_state.gemini_api_key:
                    st.warning("Gemini API key is not configured. Please enter your API key in the configuration section.")
            
            # Natural language input
            nl_query = st.text_area(
                "Enter your question in natural language",
                value=st.session_state.natural_language_query,
                height=100,
                key="nl_query_input"
            )
            
            # Context information
            st.info("Providing context about your database structure helps generate better queries.")
            
            # Display database schema information if available
            if st.session_state.db_schema:
                schema_info = self._format_schema_info(st.session_state.db_schema)
                with st.expander("Database Schema Information (Used for Context)", expanded=False):
                    st.text_area("Schema", value=schema_info, height=150, key="schema_display", disabled=True)
            else:
                schema_info = ""
                st.warning("No database schema information available. Connect to a database to see schema details.")
            
            # Generate SQL button
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                generate_button = st.button("Generate SQL", key="generate_sql_btn")
            
            with col2:
                # Add a button to start a conversation for follow-ups
                if st.button("Start Conversation", key="start_conversation_btn"):
                    if not nl_query:
                        st.warning("Please enter a question first.")
                    else:
                        st.session_state.active_conversation = True
                        st.session_state.chat_history = []
                        st.rerun()
            
            # Update session state for nl_query
            if nl_query != st.session_state.natural_language_query:
                st.session_state.natural_language_query = nl_query
            
            # Process when generate button is clicked
            if generate_button and nl_query:
                # Check if API key is configured for selected model
                if model_provider == "Groq" and not st.session_state.groq_api_key:
                    st.error("Groq API key is not configured. Please enter your API key in the configuration section.")
                    return
                
                if model_provider == "Gemini" and not st.session_state.gemini_api_key:
                    st.error("Gemini API key is not configured. Please enter your API key in the configuration section.")
                    return
                    
                with st.spinner("Generating SQL query..."):
                    try:
                        sql_query = None
                        if model_provider == "Groq":
                            sql_query = self._generate_sql_groq(nl_query, schema_info)
                        elif model_provider == "Gemini":
                            sql_query = self._generate_sql_gemini(nl_query, schema_info)
                        
                        if sql_query:
                            sql_query = self._format_sql_query(sql_query)
                            st.session_state.current_query = sql_query
                            
                            # Add to chat history if in conversation mode
                            if st.session_state.active_conversation:
                                st.session_state.chat_history.append({
                                    "role": "user",
                                    "content": nl_query
                                })
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": f"I've generated this SQL query based on your question:\n\n```sql\n{sql_query}\n```"
                                })
                                # Trim to keep token usage under control
                                if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
                                    st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]
                            
                            st.success("SQL query generated successfully!")
                        else:
                            st.error("Failed to generate SQL query. Please try rephrasing your question.")
                    except Exception as e:
                        st.error(f"Error generating SQL query: {str(e)}")
        
        with chat_tab:
            self._follow_up_chat_ui(db_manager)
            
        with db_qa_tab:
            self._database_qa_ui(db_manager)

        # Display the generated SQL query and results OUTSIDE the tabs
        # so they persist across tab switches without needing a rerun
        if st.session_state.current_query:
            st.divider()
            st.subheader("Generated SQL Query")
            st.code(st.session_state.current_query, language="sql")

            if st.button("Execute Query", key="execute_nl_query_btn"):
                with st.spinner("Executing query..."):
                    result = db_manager.execute_query(st.session_state.current_query)

                # Auto-retry once if execution failed: send the error back to the
                # LLM so it can fix the query before showing anything to the user
                if result is None:
                    last_error = st.session_state.get("last_query_error", "")
                    if last_error and st.session_state.get("natural_language_query"):
                        with st.spinner("Query failed — asking AI to fix it automatically..."):
                            schema_info = self._format_schema_info(st.session_state.db_schema) if st.session_state.db_schema else ""
                            fixed_sql = None
                            if model_provider == "Groq":
                                fixed_sql = self._generate_sql_groq_with_retry(
                                    st.session_state.natural_language_query,
                                    schema_info,
                                    st.session_state.current_query,
                                    last_error
                                )
                            elif model_provider == "Gemini":
                                fixed_sql = self._generate_sql_gemini_with_retry(
                                    st.session_state.natural_language_query,
                                    schema_info,
                                    st.session_state.current_query,
                                    last_error
                                )
                        if fixed_sql:
                            fixed_sql = self._format_sql_query(fixed_sql)
                            st.session_state.current_query = fixed_sql
                            st.info("⚠️ Original query had an error. AI generated a corrected version — please review and execute again.")
                        else:
                            st.error("Could not auto-fix the query. Please rephrase your question.")
                elif isinstance(result, pd.DataFrame):
                    st.success(f"Query executed successfully! {len(result)} rows returned.")
                    st.subheader("Query Results")
                    st.dataframe(result, width="stretch")
                    st.session_state.query_results = result
                else:
                    st.success(str(result))

            # If results already in session state from a previous run, show them
            elif st.session_state.get("query_results") is not None and isinstance(st.session_state.query_results, pd.DataFrame):
                st.subheader("Last Query Results")
                st.dataframe(st.session_state.query_results, width="stretch")
    
    def sql_editor_ui(self, db_manager):
        """UI for SQL editor"""
        st.subheader("SQL Query Editor")
        
        # Query name and description (for saving to workspace)
        col1, col2 = st.columns(2)
        with col1:
            query_name = st.text_input(
                "Query Name",
                value=st.session_state.get("query_name", ""),
                key="query_name_input",
                placeholder="Enter a name for this query"
            )
        
        with col2:
            query_description = st.text_input(
                "Description",
                value=st.session_state.get("query_description", ""),
                key="query_description_input",
                placeholder="Optional description"
            )
        
        # SQL query editor
        # Use a dedicated key so this widget does NOT overwrite current_query
        # just by being rendered — only an explicit Execute writes back.
        if "sql_editor_content" not in st.session_state or (
            st.session_state.current_query and
            st.session_state.sql_editor_content != st.session_state.current_query
        ):
            st.session_state.sql_editor_content = st.session_state.current_query

        sql_query = st.text_area(
            "Enter SQL Query",
            value=st.session_state.sql_editor_content,
            height=150,
            key="sql_editor"
        )
        st.session_state.sql_editor_content = sql_query
        
        # Store query name and description in session state
        st.session_state.query_name = query_name
        st.session_state.query_description = query_description
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Execute query button
            if st.button("Execute Query", key="execute_sql_btn"):
                if not sql_query:
                    st.warning("Please enter a SQL query.")
                    return

                # Lock the current_query to what's in the editor at click time
                st.session_state.current_query = sql_query

                with st.spinner("Executing query..."):
                    result = db_manager.execute_query(sql_query)

                if isinstance(result, pd.DataFrame):
                    st.success(f"Query executed successfully! {len(result)} rows returned.")
                    st.session_state.query_results = result
                    st.subheader("Query Results")
                    st.dataframe(result, width="stretch")
                elif result is not None:
                    st.success(str(result))
        
        with col2:
            # Save to workspace button
            if st.button("Save to Workspace", key="save_to_workspace_btn"):
                if not sql_query:
                    st.warning("Please enter a SQL query.")
                    return
                
                if not query_name:
                    st.warning("Please enter a name for this query.")
                    return
                
                # Check if user is logged in
                current_user = None
                if "user_management" in st.session_state:
                    current_user = st.session_state.user_management.get_current_user()
                
                if not current_user:
                    st.warning("Please log in to save queries to workspaces.")
                    return
                
                # Check if collaboration module is available
                if "collaboration" in st.session_state:
                    # Generate a unique ID for the query
                    import uuid
                    query_id = str(uuid.uuid4())
                    
                    # Create query data
                    query_data = {
                        "name": query_name,
                        "description": query_description,
                        "sql": sql_query,
                        "created_by": current_user,
                        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "modified_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "database": st.session_state.get("connected_db", "Unknown")
                    }
                    
                    # Add to workspace
                    if st.session_state.collaboration.add_to_workspace("query", query_id, query_data):
                        st.success(f"Query '{query_name}' saved to workspace '{st.session_state.current_workspace}'.")
                    else:
                        st.error("Failed to save query to workspace.")
                else:
                    st.error("Collaboration module not available.")
        
        with col3:
            # Clear button
            if st.button("Clear Editor", key="clear_sql_btn"):
                st.session_state.current_query = ""
                st.session_state.query_name = ""
                st.session_state.query_description = ""
                st.rerun()
    
    def _generate_sql_groq(self, nl_query, schema_info):
        """Generate SQL query using Groq API"""
        if not self.groq_client:
            st.error("Groq API key is not configured. Please enter your API key in the configuration section.")
            return None
        
        prompt = self._create_prompt(nl_query, schema_info)
        system_msg = self._get_sql_system_message()
        
        try:
            # Use Groq API to generate SQL
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2048
            )
            
            # Extract the SQL query from the response
            sql = self._extract_sql_from_response(response.choices[0].message.content)
            
            # Validate table names against schema
            if sql:
                validation_error = self._validate_sql_tables(sql)
                if validation_error:
                    # Auto-retry with the validation error
                    st.warning(f"⚠️ Generated query references invalid tables. Auto-correcting…")
                    retry_prompt = self._create_prompt(
                        nl_query, schema_info,
                        previous_sql=sql,
                        error_message=validation_error
                    )
                    retry_response = self.groq_client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": "You are an expert SQL debugger. The previous query used tables that DO NOT EXIST. Fix it using ONLY the tables listed in the schema. Return ONLY the corrected SQL."},
                            {"role": "user", "content": retry_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=2048
                    )
                    sql = self._extract_sql_from_response(retry_response.choices[0].message.content)
            
            return sql
        except Exception as e:
            st.error(f"Error calling Groq API: {str(e)}")
            return None
    
    def _generate_sql_gemini(self, nl_query, schema_info):
        """Generate SQL query using Gemini API"""
        if not self.gemini_client:
            st.error("Gemini API key is not configured. Please enter your API key in the configuration section.")
            return None
        
        system_msg = self._get_sql_system_message()
        full_prompt = f"{system_msg}\n\n{self._create_prompt(nl_query, schema_info)}"
        
        try:
            # Use Gemini API to generate SQL
            if self.use_new_genai:
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=full_prompt
                )
                sql = self._extract_sql_from_response(response.text)
            else:
                model = self.gemini_client.GenerativeModel('gemini-pro')
                response = model.generate_content(full_prompt)
                sql = self._extract_sql_from_response(response.text)
            
            # Validate table names against schema
            if sql:
                validation_error = self._validate_sql_tables(sql)
                if validation_error:
                    st.warning(f"⚠️ Generated query references invalid tables. Auto-correcting…")
                    retry_prompt = f"{system_msg}\n\n{self._create_prompt(nl_query, schema_info, previous_sql=sql, error_message=validation_error)}"
                    if self.use_new_genai:
                        retry_response = self.gemini_client.models.generate_content(
                            model="gemini-2.0-flash-exp",
                            contents=retry_prompt
                        )
                        sql = self._extract_sql_from_response(retry_response.text)
                    else:
                        retry_model = self.gemini_client.GenerativeModel('gemini-pro')
                        retry_response = retry_model.generate_content(retry_prompt)
                        sql = self._extract_sql_from_response(retry_response.text)
            
            return sql
        except Exception as e:
            st.error(f"Error calling Gemini API: {str(e)}")
            return None
    
    def _generate_sql_groq_with_retry(self, nl_query, schema_info, broken_sql, error_message):
        """Re-prompt Groq with the broken query + error so it can fix it."""
        if not self.groq_client:
            return None
        prompt = self._create_prompt(nl_query, schema_info, previous_sql=broken_sql, error_message=error_message)
        try:
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert SQL debugger. Fix the broken SQL query using the exact error message and schema provided. Return ONLY the corrected SQL, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2048
            )
            return self._extract_sql_from_response(response.choices[0].message.content)
        except Exception:
            return None

    def _generate_sql_gemini_with_retry(self, nl_query, schema_info, broken_sql, error_message):
        """Re-prompt Gemini with the broken query + error so it can fix it."""
        if not self.gemini_client:
            return None
        prompt = self._create_prompt(nl_query, schema_info, previous_sql=broken_sql, error_message=error_message)
        try:
            if self.use_new_genai:
                response = self.gemini_client.models.generate_content(model="gemini-2.0-flash-exp", contents=prompt)
                return self._extract_sql_from_response(response.text)
            else:
                model = self.gemini_client.GenerativeModel("gemini-pro")
                response = model.generate_content(prompt)
                return self._extract_sql_from_response(response.text)
        except Exception:
            return None

    def _get_valid_table_names(self):
        """Extract valid table names from the current database schema."""
        schema = st.session_state.get("db_schema")
        if not schema:
            return []
        
        if "tables" in schema:
            return list(schema["tables"].keys())
        elif "collections" in schema:
            return list(schema["collections"].keys())
        return []

    def _format_sql_query(self, sql):
        """Format the SQL query to be easily readable by humans."""
        if not sql:
            return sql
        try:
            if sqlparse is not None:
                return sqlparse.format(
                    sql,
                    reindent=True,
                    keyword_case='upper',
                    identifier_case='lower',
                    strip_comments=False,
                    use_space_around_operators=True,
                    comma_first=False
                )
            return sql
        except Exception:
            return sql

    def _validate_sql_tables(self, sql):
        """Validate that all table references in the SQL exist in the schema.
        
        Returns an error message string if invalid tables are found,
        or None if all tables are valid.
        """
        valid_tables = self._get_valid_table_names()
        if not valid_tables:
            return None  # Can't validate without schema
        
        valid_tables_lower = {t.lower() for t in valid_tables}
        
        # Extract table references from SQL using regex
        # Match: FROM table, JOIN table, INTO table, UPDATE table, TABLE table
        # Also handle schema-qualified names (schema.table) and quoted names
        table_patterns = [
            r'\bFROM\s+(?:ONLY\s+)?["\']?(\w+)["\']?',
            r'\bJOIN\s+["\']?(\w+)["\']?',
            r'\bINTO\s+["\']?(\w+)["\']?',
            r'\bUPDATE\s+["\']?(\w+)["\']?',
            r'\bTABLE\s+["\']?(\w+)["\']?',
        ]
        
        referenced_tables = set()
        sql_upper = sql.upper()
        
        for pattern in table_patterns:
            matches = re.finditer(pattern, sql_upper)
            for match in matches:
                table_name = match.group(1).strip('"\'')
                referenced_tables.add(table_name.lower())
        
        # Also extract CTE names (WITH name AS ...) — these are valid temporary tables
        cte_pattern = r'\bWITH\s+(\w+)\s+AS\s*\('
        cte_names = set()
        for match in re.finditer(cte_pattern, sql_upper):
            cte_names.add(match.group(1).lower())
        
        # Also handle comma-separated CTEs: , name AS (
        cte_comma_pattern = r',\s*(\w+)\s+AS\s*\('
        for match in re.finditer(cte_comma_pattern, sql_upper):
            cte_names.add(match.group(1).lower())
        
        # Remove known non-table keywords that regex might pick up
        non_table_keywords = {
            "select", "set", "values", "where", "group", "order", "having",
            "limit", "offset", "union", "intersect", "except", "all", "distinct",
            "exists", "not", "null", "true", "false", "case", "when", "then",
            "else", "end", "as", "on", "and", "or", "in", "between", "like",
            "is", "lateral", "each", "recursive",
        }
        
        # Filter: keep only actual table references (not CTEs, not keywords)
        invalid_tables = []
        for table in referenced_tables:
            if table in non_table_keywords:
                continue
            if table in cte_names:
                continue  # CTEs are valid temporary references
            if table not in valid_tables_lower:
                invalid_tables.append(table)
        
        if invalid_tables:
            return (
                f"INVALID TABLE NAMES DETECTED: {', '.join(invalid_tables)}. "
                f"These tables DO NOT EXIST in the database. "
                f"The ONLY valid tables are: {', '.join(valid_tables)}. "
                f"Rewrite the query using ONLY these valid table names."
            )
        
        return None

    def _get_sql_system_message(self):
        """Return a comprehensive system message for SQL generation LLM calls."""
        return (
            "You are an elite SQL expert who converts natural language questions into "
            "production-quality SQL queries. You are proficient with ALL advanced SQL features:\n\n"
            "ADVANCED SQL CAPABILITIES YOU MUST USE WHEN APPROPRIATE:\n"
            "- **Window Functions**: ROW_NUMBER(), RANK(), DENSE_RANK(), NTILE(), "
            "LAG(), LEAD(), FIRST_VALUE(), LAST_VALUE(), SUM() OVER(), AVG() OVER(), "
            "COUNT() OVER() with PARTITION BY and ORDER BY clauses.\n"
            "- **Common Table Expressions (CTEs)**: WITH clause for breaking complex queries "
            "into readable, logical steps. Use recursive CTEs for hierarchical data.\n"
            "- **Subqueries**: Correlated and non-correlated subqueries in SELECT, FROM, WHERE, "
            "and HAVING clauses. Use EXISTS/NOT EXISTS for existence checks.\n"
            "- **Set Operations**: UNION, UNION ALL, INTERSECT, EXCEPT for combining result sets.\n"
            "- **Advanced JOINs**: LEFT, RIGHT, FULL OUTER, CROSS, SELF joins. "
            "Use LATERAL joins where supported.\n"
            "- **Aggregation**: GROUP BY with ROLLUP, CUBE, GROUPING SETS. "
            "Use FILTER clause for conditional aggregation where supported.\n"
            "- **CASE expressions**: For conditional logic, pivoting, and computed columns.\n"
            "- **String & Date functions**: Appropriate to the SQL dialect.\n\n"
            "QUERY QUALITY RULES:\n"
            "- ALWAYS format SQL with proper indentation and line breaks for readability.\n"
            "- Use meaningful aliases for all tables (e.g., o for orders, c for customers).\n"
            "- Alias every derived/computed column with AS and a descriptive name.\n"
            "- Use CTEs instead of deeply nested subqueries when possible.\n"
            "- Prefer window functions over self-joins for ranking/running totals.\n"
            "- Return ONLY the raw SQL query — no markdown fences, no explanations, no comments.\n"
            "- End every query with a semicolon."
        )

    def _create_prompt(self, nl_query, schema_info, previous_sql=None, error_message=None):
        """Create a strongly-guided prompt for SQL generation.

        If previous_sql and error_message are supplied, the prompt asks the
        model to fix the broken query rather than generate from scratch.
        """
        db_type = "SQLite"
        if st.session_state.current_connection:
            db_type = st.session_state.current_connection.get("type", "SQL")

        dialect_notes = {
            "SQLite": (
                "- Use SQLite syntax only.\n"
                "- Do NOT use INFORMATION_SCHEMA; use sqlite_master instead.\n"
                "- String concatenation uses || not CONCAT().\n"
                "- No RIGHT JOIN support; rewrite as LEFT JOIN.\n"
                "- SQLite supports window functions (ROW_NUMBER, RANK, etc.) since version 3.25.\n"
                "- Use COALESCE() for null handling. No ISNULL() or IFNULL() variants."
            ),
            "PostgreSQL": (
                "- Use PostgreSQL syntax.\n"
                "- String functions: LENGTH(), LOWER(), UPPER(), TRIM(), SUBSTRING().\n"
                "- Use ILIKE for case-insensitive matching.\n"
                "- Cast with ::type syntax where appropriate.\n"
                "- Supports advanced window functions, CTEs, LATERAL joins, FILTER clause.\n"
                "- Use generate_series() for sequences. Use DISTINCT ON for deduplication.\n"
                "- Aggregate functions like AVG(), SUM() work ONLY on numeric columns — "
                "NEVER apply them to TEXT/VARCHAR columns."
            ),
            "MySQL": (
                "- Use MySQL syntax.\n"
                "- Use LIMIT for row limiting.\n"
                "- String functions: LENGTH(), LOWER(), UPPER(), TRIM(), SUBSTRING().\n"
                "- Use backticks for identifiers if they clash with reserved words.\n"
                "- MySQL 8.0+ supports window functions and CTEs.\n"
                "- Use IFNULL() or COALESCE() for null handling."
            ),
        }.get(db_type, "- Use standard ANSI SQL syntax.")

        # Extract valid table names and columns from schema for strict validation
        valid_tables_list = self._get_valid_table_names()
        schema = st.session_state.get("db_schema", {})
        tables_schema = schema.get("tables", {})
        
        if valid_tables_list:
            table_col_lines = []
            for table_name in valid_tables_list:
                table_info = tables_schema.get(table_name, {})
                cols = table_info.get("columns", [])
                col_names = [c["name"] for c in cols] if cols else []
                if col_names:
                    table_col_lines.append(f"  {table_name}: {', '.join(col_names)}")
                else:
                    table_col_lines.append(f"  {table_name}")
            
            valid_tables_block = (
                f"VALID TABLES AND THEIR COLUMNS (use ONLY these exact names):\n"
                + "\n".join(table_col_lines) + "\n\n"
                f"⚠️ CRITICAL: If a table or column name is NOT listed above, it DOES NOT EXIST.\n"
                f"Do NOT guess or invent table/column names like adding prefixes/suffixes.\n"
                f"If the schema doesn't have what's asked for, explain that — do NOT fabricate names.\n\n"
            )
        else:
            valid_tables_block = ""

        schema_block = f"""{valid_tables_block}DATABASE SCHEMA
===============
{schema_info}
""" if schema_info else "No schema provided — infer table/column names carefully from the question.\n"

        # Detect if the question likely needs advanced SQL features
        advanced_hints = self._detect_query_complexity(nl_query)

        if previous_sql and error_message:
            task_block = f"""TASK: FIX A BROKEN SQL QUERY
==============================
The following query was generated for the question below but failed with an error.
Rewrite it so it executes correctly.

ORIGINAL QUESTION:
{nl_query}

BROKEN QUERY:
{previous_sql}

DATABASE ERROR:
{error_message}

Fix the query. Address the exact error. Use ONLY the tables and columns listed in the schema above.
Return ONLY the corrected SQL — no markdown, no explanation."""
        else:
            task_block = f"""TASK: GENERATE A SQL QUERY
===========================
Convert the natural language question below into a correct, well-formatted SQL query.

QUESTION:
{nl_query}

{advanced_hints}"""

        rules = f"""STRICT RULES — follow every one:
1. Use ONLY tables and columns that appear in the schema above. Do NOT invent table or column names.
2. If the question involves a many-to-many relationship, use the join table from the schema.
3. For filtering aggregates use HAVING, not WHERE.
4. Prefer explicit JOIN ... ON over implicit comma joins.
5. Alias every derived column with a meaningful name (e.g. AS avg_duration).
6. Return ONLY the raw SQL — no markdown code fences, no explanation, no comments.
7. End the query with a semicolon.
8. Format the query with proper indentation and line breaks.
9. When ranking, top-N, or running totals are needed, USE window functions.
10. When the query is complex, break it into CTEs using WITH for readability.
11. When comparing subsets of data, consider using subqueries or CTEs.
12. {dialect_notes}"""

        return f"{schema_block}\n{task_block}\n\n{rules}"

    def _detect_query_complexity(self, nl_query):
        """Detect if the user's question needs advanced SQL features and return hints."""
        hints = []
        q = nl_query.lower()

        # Window function indicators
        window_keywords = ["rank", "top", "bottom", "nth", "running total", "cumulative",
                           "moving average", "row number", "lead", "lag", "previous",
                           "next", "percentile", "ntile", "quartile", "first", "last",
                           "dense rank", "partition"]
        if any(kw in q for kw in window_keywords):
            hints.append(
                "HINT: This question likely requires WINDOW FUNCTIONS. "
                "Use ROW_NUMBER(), RANK(), DENSE_RANK(), LAG(), LEAD(), "
                "or aggregate OVER (PARTITION BY ... ORDER BY ...) as appropriate."
            )

        # Subquery indicators
        subquery_keywords = ["more than average", "above average", "below average",
                             "higher than", "lower than", "compared to", "relative to",
                             "where the", "that have", "who have", "not in", "exists",
                             "does not exist", "in the list"]
        if any(kw in q for kw in subquery_keywords):
            hints.append(
                "HINT: This question likely requires a SUBQUERY. "
                "Use a subquery in WHERE, HAVING, or FROM clause as appropriate."
            )

        # CTE indicators
        cte_keywords = ["step by step", "first find", "then calculate", "multi-step",
                        "breakdown", "with the result", "use the result", "based on the",
                        "intermediate"]
        if any(kw in q for kw in cte_keywords):
            hints.append(
                "HINT: This question involves multiple logical steps. "
                "Use CTEs (WITH ... AS) to make the query readable and modular."
            )

        # Set operation indicators
        set_keywords = ["union", "combine", "both", "together", "from both",
                        "intersection", "in common", "except", "not in other",
                        "difference between"]
        if any(kw in q for kw in set_keywords):
            hints.append(
                "HINT: This question may require SET OPERATIONS like UNION, "
                "INTERSECT, or EXCEPT to combine result sets."
            )

        # Multiple aggregation indicators
        multi_agg_keywords = ["and also", "along with", "as well as", "together with",
                              "for each", "per", "group by", "breakdown by",
                              "summarize", "summary", "statistics", "stats"]
        if any(kw in q for kw in multi_agg_keywords):
            hints.append(
                "HINT: This question requires MULTIPLE AGGREGATIONS. "
                "Use GROUP BY with multiple aggregate functions and meaningful aliases."
            )

        if hints:
            return "ADVANCED SQL HINTS FOR THIS QUESTION:\n" + "\n".join(hints)
        return ""

    
    def _extract_sql_from_response(self, response_text):
        """Extract SQL query from model response with robust multi-pattern parsing."""
        if not response_text:
            return None

        text = response_text.strip()

        # 1. Try to extract from ```sql ... ``` code blocks (may be multiple)
        sql_blocks = re.findall(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if sql_blocks:
            # Join multiple SQL blocks with newline (handles multi-statement responses)
            combined = "\n\n".join(block.strip() for block in sql_blocks if block.strip())
            if combined:
                return combined

        # 2. Try generic code blocks ```...```
        generic_blocks = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
        for block in generic_blocks:
            block = block.strip()
            # Check if block looks like SQL
            if block and any(block.upper().lstrip().startswith(kw) for kw in
                           ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER",
                            "DROP", "WITH", "EXPLAIN"]):
                return block

        # 3. No code blocks — extract SQL by finding known starting keywords
        lines = text.split('\n')
        sql_lines = []
        sql_start_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE",
                              "ALTER", "DROP", "WITH", "EXPLAIN", "SHOW", "DESCRIBE"]

        in_sql = False
        for line in lines:
            stripped = line.strip()
            upper_stripped = stripped.upper()

            # Skip obvious non-SQL lines
            if any(upper_stripped.startswith(skip) for skip in
                   ["NOTE:", "EXPLANATION:", "HERE", "THIS", "THE ", "I ", "PLEASE",
                    "BELOW", "ABOVE", "**", "##", "#"]):
                if in_sql:  # End of SQL block
                    break
                continue

            # Start collecting when we see a SQL keyword
            if any(upper_stripped.startswith(keyword) for keyword in sql_start_keywords):
                in_sql = True
                sql_lines.append(stripped)
            elif in_sql and stripped:
                # Continue collecting SQL lines
                sql_lines.append(stripped)
            elif in_sql and not stripped:
                # Empty line inside SQL — might be formatting, keep going
                sql_lines.append("")

        if sql_lines:
            # Remove trailing empty lines
            while sql_lines and not sql_lines[-1].strip():
                sql_lines.pop()
            return '\n'.join(sql_lines)

        # 4. Last resort — clean and return the whole response
        cleaned = text.replace("```", "").strip()
        return cleaned if cleaned else None
    
    def _format_schema_info(self, schema):
        """Format schema as pseudo-CREATE TABLE statements.

        This representation is far more recognisable to LLMs than a plain
        indented list — they are trained on DDL and respond much more accurately.
        """
        lines = []

        if "tables" in schema:
            for table_name, table_info in schema["tables"].items():
                col_defs = []
                for col in table_info["columns"]:
                    name = col["name"]
                    ctype = col["type"]
                    suffix = " PRIMARY KEY" if name in table_info.get("primary_keys", []) else ""
                    col_defs.append(f"    {name} {ctype}{suffix}")

                for fk in table_info.get("foreign_keys", []):
                    src = ", ".join(fk["constrained_columns"])
                    ref_table = fk["referred_table"]
                    ref_cols = ", ".join(fk["referred_columns"])
                    col_defs.append(
                        f"    FOREIGN KEY ({src}) REFERENCES {ref_table}({ref_cols})"
                    )

                lines.append(f"CREATE TABLE {table_name} (")
                lines.append(",\n".join(col_defs))
                lines.append(");\n")

        elif "collections" in schema:
            for coll_name, coll_info in schema["collections"].items():
                lines.append(f"-- Collection: {coll_name}")
                for field in coll_info.get("fields", []):
                    lines.append(f"--   {field['name']} ({field['type']})")
                lines.append("")

        return "\n".join(lines)
        
    def _follow_up_chat_ui(self, db_manager):
        """UI for follow-up chat about SQL queries"""
        st.subheader("Follow-up Chat")
        
        # Check if API keys are configured
        if not st.session_state.groq_api_key and not st.session_state.gemini_api_key:
            st.warning("Please configure at least one API key in the Query Generator tab to use the follow-up chat feature.")
            return
            
        # Display chat history
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div style="background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #1e1e1e; color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <strong>Assistant:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            if st.session_state.active_conversation:
                st.info("Your conversation has started. Ask a follow-up question below.")
            else:
                st.info("Start a conversation in the Query Generator tab by entering a question and clicking 'Start Conversation'.")
                return
                
        # Input for follow-up questions
        follow_up = st.text_area(
            "Ask a follow-up question",
            key="follow_up_input",
            height=100,
            placeholder="Example: 'Can you modify that query to also include...?' or 'What if I want to filter by...?'"
        )
        
        # Get schema info
        if st.session_state.db_schema:
            schema_info = self._format_schema_info(st.session_state.db_schema)
        else:
            schema_info = ""
            
        # Send button
        if st.button("Send", key="send_follow_up_btn") and follow_up:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": follow_up
            })
            
            # Determine which model to use
            model_provider = st.session_state.get("model_provider", "Groq" if st.session_state.groq_api_key else "Gemini")
            
            with st.spinner("Generating response..."):
                try:
                    # Construct the conversation history for context
                    conversation_history = "\n\n".join([
                        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                        for msg in st.session_state.chat_history[:-1]  # Exclude the latest user message
                    ])
                    
                    # Generate response
                    if model_provider == "Groq" and st.session_state.groq_api_key:
                        response = self._generate_follow_up_response_groq(follow_up, conversation_history, schema_info)
                    elif model_provider == "Gemini" and st.session_state.gemini_api_key:
                        response = self._generate_follow_up_response_gemini(follow_up, conversation_history, schema_info)
                    else:
                        st.error("No API key configured for the selected model.")
                        return
                    
                    if response:
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response
                        })
                        # Trim to keep token usage under control
                        if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
                            st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]
                        
                        # Check if response contains SQL query and update current_query
                        sql_match = re.search(r'```sql\n(.*?)\n```', response, re.DOTALL)
                        if sql_match:
                            sql_query = sql_match.group(1).strip()
                            st.session_state.current_query = sql_query
                    else:
                        st.error("Failed to generate a response. Please try again.")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
            
            # Rerun to update the UI
            st.rerun()
            
        # End conversation button
        if st.session_state.active_conversation and st.button("End Conversation", key="end_conversation_btn"):
            st.session_state.active_conversation = False
            st.success("Conversation ended. You can start a new one from the Query Generator tab.")
            st.rerun()
            
    def _database_qa_ui(self, db_manager):
        """UI for natural language Q&A about the database"""
        st.subheader("Database Q&A")
        
        # Check if API keys are configured
        if not st.session_state.groq_api_key and not st.session_state.gemini_api_key:
            st.warning("Please configure at least one API key in the Query Generator tab to use the Database Q&A feature.")
            return
            
        # Check if connected to a database
        if not st.session_state.db_schema:
            st.warning("Please connect to a database first to use this feature.")
            return
            
        # Display previous Q&A
        if st.session_state.db_qa_history:
            for qa in st.session_state.db_qa_history:
                st.markdown(f"""
                <div style="background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong>Question:</strong> {qa["question"]}
                </div>
                <div style="background-color: #1e1e1e; color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                    <strong>Answer:</strong> {qa["answer"]}
                </div>
                """, unsafe_allow_html=True)
                
        # Input for database questions
        db_question = st.text_area(
            "Ask a question about your database",
            key="db_question_input",
            height=100,
            placeholder="Example: 'What tables are in my database?' or 'What columns are in the customers table?'"
        )
        
        # Get schema info
        schema_info = self._format_schema_info(st.session_state.db_schema)
        
        # Ask button
        if st.button("Ask", key="ask_db_question_btn") and db_question:
            # Determine which model to use
            model_provider = st.session_state.get("model_provider", "Groq" if st.session_state.groq_api_key else "Gemini")
            
            with st.spinner("Generating answer..."):
                try:
                    # Generate answer
                    if model_provider == "Groq" and st.session_state.groq_api_key:
                        answer = self._generate_db_answer_groq(db_question, schema_info)
                    elif model_provider == "Gemini" and st.session_state.gemini_api_key:
                        answer = self._generate_db_answer_gemini(db_question, schema_info)
                    else:
                        st.error("No API key configured for the selected model.")
                        return
                    
                    if answer:
                        # Add to Q&A history
                        st.session_state.db_qa_history.append({
                            "question": db_question,
                            "answer": answer
                        })
                        # Trim to keep token usage under control
                        if len(st.session_state.db_qa_history) > MAX_QA_HISTORY:
                            st.session_state.db_qa_history = st.session_state.db_qa_history[-MAX_QA_HISTORY:]
                    else:
                        st.error("Failed to generate an answer. Please try again.")
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
            
            # Rerun to update the UI
            st.rerun()
            
        # Clear history button
        if st.session_state.db_qa_history and st.button("Clear History", key="clear_db_qa_history_btn"):
            st.session_state.db_qa_history = []
            st.success("Q&A history cleared.")
            st.rerun()
            
    def _generate_follow_up_response_groq(self, follow_up, conversation_history, schema_info):
        """Generate a response to a follow-up question using Groq API"""
        if not self.groq_client:
            return None
            
        # Construct the prompt
        prompt = f"""
        You are an expert SQL assistant helping with follow-up questions about SQL queries. You have access to the conversation history and database schema.
        
        Database Schema Information:
        {schema_info}
        
        Conversation History:
        {conversation_history}
        
        User's Follow-up Question:
        {follow_up}
        
        Please respond to the follow-up question. If the user is asking for a modified SQL query, provide the complete SQL query in a code block using markdown formatting (```sql ... ```). Also provide a brief explanation of what you changed and why.
        
        If the user is asking for clarification about the existing query, explain the relevant parts clearly.
        """
        
        try:
            completion = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert SQL assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            
            response = completion.choices[0].message.content.strip()
            return response
        except Exception as e:
            st.error(f"Error with Groq API: {str(e)}")
            return None
            
    def _generate_follow_up_response_gemini(self, follow_up, conversation_history, schema_info):
        """Generate a response to a follow-up question using Gemini API"""
        if not self.gemini_client:
            return None
            
        # Construct the prompt
        prompt = f"""
        You are an expert SQL assistant helping with follow-up questions about SQL queries. You have access to the conversation history and database schema.
        
        Database Schema Information:
        {schema_info}
        
        Conversation History:
        {conversation_history}
        
        User's Follow-up Question:
        {follow_up}
        
        Please respond to the follow-up question. If the user is asking for a modified SQL query, provide the complete SQL query in a code block using markdown formatting (```sql ... ```). Also provide a brief explanation of what you changed and why.
        
        If the user is asking for clarification about the existing query, explain the relevant parts clearly.
        """
        
        try:
            if self.use_new_genai:
                # New google.genai package
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=prompt
                )
                return response.text.strip()
            else:
                # Old google.generativeai package  
                model = self.gemini_client.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                return response.text.strip()
        except Exception as e:
            st.error(f"Error with Gemini API: {str(e)}")
            return None
            
    def _generate_db_answer_groq(self, question, schema_info):
        """Generate an answer to a database question using Groq API"""
        if not self.groq_client:
            return None
            
        # Construct the prompt
        prompt = f"""
        You are an expert database assistant. Your task is to answer questions about a database based on its schema.
        
        Database Schema Information:
        {schema_info}
        
        User's Question:
        {question}
        
        Please provide a clear, concise answer to the question. Focus on explaining database concepts in simple terms.
        Do not generate SQL queries unless specifically asked. Instead, explain the database structure, relationships, and concepts in plain English.
        """
        
        try:
            completion = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert database assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            
            response = completion.choices[0].message.content.strip()
            return response
        except Exception as e:
            st.error(f"Error with Groq API: {str(e)}")
            return None
            
    def _generate_db_answer_gemini(self, question, schema_info):
        """Generate an answer to a database question using Gemini API"""
        if not self.gemini_client:
            return None
            
        # Construct the prompt
        prompt = f"""
        You are an expert database assistant. Your task is to answer questions about a database based on its schema.
        
        Database Schema Information:
        {schema_info}
        
        User's Question:
        {question}
        
        Please provide a clear, concise answer to the question. Focus on explaining database concepts in simple terms.
        Do not generate SQL queries unless specifically asked. Instead, explain the database structure, relationships, and concepts in plain English.
        """
        
        try:
            if self.use_new_genai:
                # New google.genai package
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            else:
                # Old google.generativeai package
                model = self.gemini_client.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            st.error(f"Error with Gemini API: {str(e)}")
            return None