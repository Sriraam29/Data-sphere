import streamlit as st
import pandas as pd
import json
import io
import re
import requests
import base64
import sqlalchemy
from sqlalchemy import create_engine, text
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

class EnterpriseIntegration:
    def __init__(self):
        """Initialize the Enterprise Integration module"""
        # Initialize session state for API endpoints
        if "api_endpoints" not in st.session_state:
            st.session_state.api_endpoints = {}
        
        if "data_lineage" not in st.session_state:
            st.session_state.data_lineage = {}
    
    def integration_ui(self, db_manager):
        """UI for enterprise integration features"""
        st.subheader("Enterprise Integration")
        
        # Check if connected to a database
        if not st.session_state.get("connected_db"):
            st.warning("Please connect to a database first.")
            return
        
        tab1, tab2, tab3 = st.tabs(["REST API Endpoints", "Data Lineage & Governance", "BI Tools Integration"])
        
        with tab1:
            self._api_endpoints_ui(db_manager)
        
        with tab2:
            self._data_lineage_ui(db_manager)
        
        with tab3:
            self._bi_integration_ui(db_manager)
    
    def _api_endpoints_ui(self, db_manager):
        """UI for managing REST API endpoints"""
        st.subheader("REST API Endpoints")
        
        # API endpoints management
        with st.expander("Create New API Endpoint", expanded=True):
            endpoint_name = st.text_input("Endpoint Name", placeholder="e.g., get_customers")
            
            # HTTP method
            http_method = st.selectbox("HTTP Method", options=["GET", "POST", "PUT", "DELETE"])
            
            # SQL query
            sql_query = st.text_area(
                "SQL Query",
                height=150,
                placeholder="SELECT * FROM customers WHERE region = :region",
                help="Use :param_name for parameters (e.g., :customer_id)"
            )
            
            # Parameters
            st.subheader("Parameters")
            param_name = st.text_input("Parameter Name", placeholder="e.g., region")
            param_type = st.selectbox("Parameter Type", options=["String", "Integer", "Float", "Boolean", "Date"])
            param_required = st.checkbox("Required Parameter", value=True)
            
            # List of parameters
            if "temp_params" not in st.session_state:
                st.session_state.temp_params = []
            
            if st.button("Add Parameter") and param_name:
                st.session_state.temp_params.append({
                    "name": param_name,
                    "type": param_type.lower(),
                    "required": param_required
                })
                st.success(f"Added parameter: {param_name}")
            
            # Display parameters
            if st.session_state.temp_params:
                st.write("Parameters:")
                for i, param in enumerate(st.session_state.temp_params):
                    st.write(f"- {param['name']} ({param['type']}, {'required' if param['required'] else 'optional'})")
                    
                    if st.button(f"Remove", key=f"remove_param_{i}"):
                        st.session_state.temp_params.pop(i)
                        st.rerun()
            
            # Authentication settings
            st.subheader("Authentication")
            auth_required = st.checkbox("Require Authentication")
            
            auth_type = "None"
            if auth_required:
                auth_type = st.selectbox("Authentication Type", options=["API Key", "Basic Auth", "JWT"])
            
            # Rate limiting
            st.subheader("Rate Limiting")
            enable_rate_limit = st.checkbox("Enable Rate Limiting")
            
            rate_limit = 100
            if enable_rate_limit:
                rate_limit = st.number_input("Requests per Minute", min_value=1, max_value=1000, value=100)
            
            # Create endpoint
            if st.button("Create API Endpoint") and endpoint_name and sql_query:
                # Extract parameters from SQL query
                sql_params = re.findall(r':(\w+)', sql_query)
                
                # Verify that all SQL parameters are defined
                missing_params = [p for p in sql_params if p not in [param["name"] for param in st.session_state.temp_params]]
                
                if missing_params:
                    st.error(f"Missing parameter definitions for: {', '.join(missing_params)}")
                else:
                    # Create the endpoint
                    endpoint = {
                        "name": endpoint_name,
                        "method": http_method,
                        "query": sql_query,
                        "parameters": st.session_state.temp_params.copy(),
                        "auth_required": auth_required,
                        "auth_type": auth_type if auth_required else "None",
                        "rate_limit": rate_limit if enable_rate_limit else 0,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "path": f"/api/{endpoint_name}"
                    }
                    
                    st.session_state.api_endpoints[endpoint_name] = endpoint
                    st.session_state.temp_params = []
                    
                    st.success(f"API endpoint created: /api/{endpoint_name}")
                    
                    # Record in data lineage
                    if "data_lineage" not in st.session_state:
                        st.session_state.data_lineage = {}
                    
                    # Extract tables from SQL query (simplified)
                    tables = re.findall(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
                    tables.extend(re.findall(r'JOIN\s+(\w+)', sql_query, re.IGNORECASE))
                    
                    st.session_state.data_lineage[f"api_{endpoint_name}"] = {
                        "type": "api_endpoint",
                        "sources": tables,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "last_modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "owner": "DataSphere",
                        "description": f"API endpoint for {endpoint_name}"
                    }
        
        # Display existing endpoints
        if st.session_state.api_endpoints:
            st.subheader("Existing API Endpoints")
            
            for name, endpoint in st.session_state.api_endpoints.items():
                with st.expander(f"{endpoint['method']} {endpoint['path']}"):
                    st.write(f"**Created:** {endpoint['created_at']}")
                    st.code(endpoint['query'], language="sql")
                    
                    # Parameters
                    if endpoint['parameters']:
                        st.write("**Parameters:**")
                        params_df = pd.DataFrame(endpoint['parameters'])
                        st.dataframe(params_df)
                    
                    # Authentication
                    st.write(f"**Authentication:** {endpoint['auth_type']}")
                    
                    # Rate limiting
                    if endpoint['rate_limit'] > 0:
                        st.write(f"**Rate Limit:** {endpoint['rate_limit']} requests per minute")
                    
                    # Endpoint usage example
                    st.write("**Example Usage:**")
                    
                    if endpoint['method'] == "GET":
                        param_str = "&".join([f"{p['name']}={{{p['name']}}}" for p in endpoint['parameters']])
                        example = f"GET {endpoint['path']}?{param_str}"
                    else:
                        params = {p['name']: f"{{{p['name']}}}" for p in endpoint['parameters']}
                        example = f"{endpoint['method']} {endpoint['path']}\nContent-Type: application/json\n\n{json.dumps(params, indent=2)}"
                    
                    st.code(example)
                    
                    # Test endpoint
                    st.subheader("Test Endpoint")
                    
                    # Input fields for parameters
                    test_params = {}
                    for param in endpoint['parameters']:
                        if param['type'] == 'integer':
                            test_params[param['name']] = st.number_input(
                                f"Parameter: {param['name']} ({param['type']})",
                                key=f"test_{name}_{param['name']}"
                            )
                        elif param['type'] == 'float':
                            test_params[param['name']] = st.number_input(
                                f"Parameter: {param['name']} ({param['type']})",
                                key=f"test_{name}_{param['name']}",
                                format="%.2f"
                            )
                        elif param['type'] == 'boolean':
                            test_params[param['name']] = st.checkbox(
                                f"Parameter: {param['name']} ({param['type']})",
                                key=f"test_{name}_{param['name']}"
                            )
                        elif param['type'] == 'date':
                            test_params[param['name']] = st.date_input(
                                f"Parameter: {param['name']} ({param['type']})",
                                key=f"test_{name}_{param['name']}"
                            ).isoformat()
                        else:  # string
                            test_params[param['name']] = st.text_input(
                                f"Parameter: {param['name']} ({param['type']})",
                                key=f"test_{name}_{param['name']}"
                            )
                    
                    if st.button("Test API Call", key=f"test_{name}"):
                        with st.spinner("Executing API call..."):
                            try:
                                # Use parameterized queries to prevent SQL injection
                                query = endpoint['query']
                                
                                # For SQL databases, we'll use SQLAlchemy's text() with parameters
                                if db_manager.current_connection["type"] in ["PostgreSQL", "MySQL", "SQLite"]:
                                    # Create a connection
                                    engine = create_engine(db_manager.current_connection["connection_string"])
                                    with engine.connect() as conn:
                                        # Convert parameters to the right types based on their definitions
                                        typed_params = {}
                                        for param_name, param_value in test_params.items():
                                            # Find parameter definition
                                            param_def = next((p for p in endpoint['parameters'] if p['name'] == param_name), None)
                                            if param_def:
                                                # Convert to appropriate type
                                                if param_def['type'] == 'integer':
                                                    typed_params[param_name] = int(param_value)
                                                elif param_def['type'] == 'float':
                                                    typed_params[param_name] = float(param_value)
                                                elif param_def['type'] == 'boolean':
                                                    typed_params[param_name] = bool(param_value)
                                                else:
                                                    typed_params[param_name] = str(param_value)
                                        
                                        # Execute with parameters
                                        if query.strip().lower().startswith(("select", "show", "describe", "explain")):
                                            result = pd.read_sql(sqlalchemy.text(query), conn, params=typed_params)
                                        else:
                                            conn.execute(sqlalchemy.text(query), typed_params)
                                            result = "Query executed successfully."
                                else:
                                    # For MongoDB, we still need to do string replacement but with extra validation
                                    # This is a simplified approach - a real solution would need more robust parsing
                                    for param_name, param_value in test_params.items():
                                        # Validate parameter to prevent injection
                                        if isinstance(param_value, str):
                                            # Escape quotes in strings
                                            param_value = param_value.replace('"', '\\"').replace("'", "\\'")
                                            # Wrap in quotes
                                            param_value = f'"{param_value}"'
                                        elif isinstance(param_value, bool):
                                            param_value = str(param_value).lower()
                                        else:
                                            param_value = str(param_value)
                                        
                                        query = query.replace(f":{param_name}", param_value)
                                    
                                    # Execute query
                                    result = db_manager.execute_query(query)
                                
                                if isinstance(result, pd.DataFrame):
                                    st.success(f"Query executed successfully! {len(result)} rows returned.")
                                    
                                    # Display results
                                    st.dataframe(result)
                                    
                                    # Show response in JSON format
                                    st.subheader("API Response (JSON)")
                                    
                                    # Convert DataFrame to JSON
                                    json_result = result.to_json(orient="records")
                                    json_obj = json.loads(json_result)
                                    
                                    # Pretty print JSON
                                    st.json(json_obj)
                                else:
                                    st.success("Query executed successfully!")
                            except Exception as e:
                                st.error(f"Error executing API call: {str(e)}")
                    
                    # Delete endpoint
                    if st.button("Delete Endpoint", key=f"delete_{name}"):
                        del st.session_state.api_endpoints[name]
                        if f"api_{name}" in st.session_state.data_lineage:
                            del st.session_state.data_lineage[f"api_{name}"]
                        st.success(f"Deleted endpoint: {name}")
                        st.rerun()
    
    def _data_lineage_ui(self, db_manager):
        """UI for data lineage and governance"""
        st.subheader("Data Lineage & Governance")
        
        # Data lineage visualization
        st.write("Data lineage tracks how data flows through your system")
        
        # Add manual lineage entry
        with st.expander("Add Manual Lineage Entry"):
            lineage_name = st.text_input("Entry Name", placeholder="e.g., monthly_sales_report")
            lineage_type = st.selectbox("Entry Type", options=["Report", "Dashboard", "ETL Job", "Data Extract", "Custom"])
            
            # Source tables
            source_options = []
            schema = st.session_state.get("db_schema", {})
            
            if "tables" in schema:
                source_options = list(schema["tables"].keys())
            elif "collections" in schema:
                source_options = list(schema["collections"].keys())
            
            source_tables = st.multiselect("Source Tables/Collections", options=source_options)
            
            lineage_description = st.text_area("Description", placeholder="Describe the purpose and content of this data artifact")
            
            lineage_owner = st.text_input("Owner", value="DataSphere")
            
            if st.button("Add Lineage Entry") and lineage_name and source_tables:
                # Add to data lineage
                st.session_state.data_lineage[lineage_name] = {
                    "type": lineage_type.lower(),
                    "sources": source_tables,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "last_modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "owner": lineage_owner,
                    "description": lineage_description
                }
                
                st.success(f"Added lineage entry: {lineage_name}")
                st.rerun()
        
        # Display lineage graph
        if st.session_state.data_lineage:
            st.subheader("Data Lineage Graph")
            
            # Create a proper graph visualization using matplotlib and networkx
            try:
                import networkx as nx
                import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors
                
                # Create a directed graph
                G = nx.DiGraph()
                
                # Get all sources and targets
                all_sources = set()
                for entry, details in st.session_state.data_lineage.items():
                    all_sources.update(details.get("sources", []))
                
                # Add nodes for source tables
                for source in sorted(all_sources):
                    G.add_node(source, type="table")
                
                # Add nodes for lineage entries and edges from sources
                for entry, details in st.session_state.data_lineage.items():
                    G.add_node(entry, type=details.get("type", "unknown"))
                    
                    # Add edges from sources to this entry
                    for source in details.get("sources", []):
                        G.add_edge(source, entry)
                
                # Create the plot
                plt.figure(figsize=(10, 8))
                
                # Define node colors based on type
                color_map = {
                    "table": "lightblue",
                    "report": "lightgreen",
                    "dashboard": "lightyellow",
                    "etl job": "lightcoral",
                    "data extract": "lightpink",
                    "api_endpoint": "lightgrey",
                    "custom": "white"
                }
                
                # Get node positions using a hierarchical layout
                pos = nx.spring_layout(G, k=0.5, iterations=50)
                
                # Draw nodes with different colors based on type
                for node_type in set(nx.get_node_attributes(G, "type").values()):
                    node_list = [node for node, data in G.nodes(data=True) if data.get("type") == node_type]
                    nx.draw_networkx_nodes(
                        G, pos, 
                        nodelist=node_list,
                        node_color=color_map.get(node_type.lower(), "white"),
                        node_size=1500,
                        alpha=0.8
                    )
                
                # Draw edges
                nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, width=1.5, alpha=0.7)
                
                # Draw labels
                nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
                
                # Add a legend
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=node_type.capitalize())
                    for node_type, color in color_map.items()
                ]
                plt.legend(handles=legend_elements, loc='upper right')
                
                # Remove axis
                plt.axis('off')
                
                # Save to buffer and display
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close()
                buf.seek(0)
                
                st.image(buf, caption="Data Lineage Graph", use_column_width=True)
                
            except Exception as e:
                st.error(f"Error generating graph visualization: {str(e)}")
                
                # Fallback to text-based representation
                st.write("Falling back to text-based representation:")
                lineage_text = "```\n"
                
                # Get all sources and targets
                all_sources = set()
                for entry, details in st.session_state.data_lineage.items():
                    all_sources.update(details.get("sources", []))
                
                # Draw source tables
                for source in sorted(all_sources):
                    lineage_text += f"[{source}] (Database Table)\n"
                    
                    # Find entries that use this source
                    for entry, details in st.session_state.data_lineage.items():
                        if source in details.get("sources", []):
                            lineage_text += f"  ↓\n[{entry}] ({details.get('type', 'unknown').capitalize()})\n"
                
                lineage_text += "```"
                
                st.markdown(lineage_text)
            
            # Display entries in table format
            st.subheader("Lineage Entries")
            
            lineage_records = []
            for name, details in st.session_state.data_lineage.items():
                lineage_records.append({
                    "Name": name,
                    "Type": details.get("type", "").capitalize(),
                    "Sources": ", ".join(details.get("sources", [])),
                    "Owner": details.get("owner", ""),
                    "Created": details.get("created_at", ""),
                    "Last Modified": details.get("last_modified", "")
                })
            
            if lineage_records:
                lineage_df = pd.DataFrame(lineage_records)
                st.dataframe(lineage_df)
            
            # Search lineage
            st.subheader("Search Lineage")
            search_term = st.text_input("Search by table or artifact name")
            
            if search_term:
                search_results = []
                
                # Search in entry names
                for name, details in st.session_state.data_lineage.items():
                    if search_term.lower() in name.lower():
                        search_results.append({
                            "Type": "Artifact",
                            "Name": name,
                            "Details": details.get("description", ""),
                            "Related": ", ".join(details.get("sources", []))
                        })
                
                # Search in sources
                for name, details in st.session_state.data_lineage.items():
                    for source in details.get("sources", []):
                        if search_term.lower() in source.lower():
                            search_results.append({
                                "Type": "Source Table",
                                "Name": source,
                                "Details": f"Used in {name}",
                                "Related": name
                            })
                
                if search_results:
                    search_df = pd.DataFrame(search_results)
                    st.dataframe(search_df)
                else:
                    st.info(f"No results found for '{search_term}'")
        else:
            st.info("No lineage data available yet. Add entries manually or create API endpoints.")
        
        # Data governance section
        st.subheader("Data Governance")
        
        with st.expander("Data Governance Policies"):
            # Sample policies
            policies = [
                {
                    "name": "Data Retention Policy",
                    "description": "Defines how long different types of data should be retained",
                    "status": "Draft"
                },
                {
                    "name": "Data Classification",
                    "description": "Guidelines for classifying data sensitivity",
                    "status": "Active"
                },
                {
                    "name": "Access Control",
                    "description": "Policies for controlling access to sensitive data",
                    "status": "Active"
                }
            ]
            
            for policy in policies:
                st.write(f"**{policy['name']}** ({policy['status']})")
                st.write(policy['description'])
                st.write("---")
    
    def _bi_integration_ui(self, db_manager):
        """UI for BI tools integration"""
        st.subheader("BI Tools Integration")
        
        # Select BI tool
        bi_tool = st.selectbox(
            "BI Tool",
            options=["Power BI", "Tableau", "Looker", "Other"]
        )
        
        # Connection information
        st.subheader(f"Connect to {bi_tool}")
        
        # Store the selected BI tool in session state
        if "selected_bi_tool" not in st.session_state:
            st.session_state.selected_bi_tool = bi_tool
        elif st.session_state.selected_bi_tool != bi_tool:
            # If the user changed the selection, update it
            st.session_state.selected_bi_tool = bi_tool
            # Clear any previous connection method selections
            if "powerbi_connection_method" in st.session_state:
                del st.session_state.powerbi_connection_method
            if "tableau_connection_method" in st.session_state:
                del st.session_state.tableau_connection_method
            if "looker_connection_method" in st.session_state:
                del st.session_state.looker_connection_method
        
        # Show only the selected BI tool integration UI
        if bi_tool == "Power BI":
            self._power_bi_integration_ui(db_manager)
        elif bi_tool == "Tableau":
            self._tableau_integration_ui(db_manager)
        elif bi_tool == "Looker":
            self._looker_integration_ui(db_manager)
        else:
            self._generic_bi_integration_ui(db_manager)
    
    def _power_bi_integration_ui(self, db_manager):
        """UI for Power BI integration"""
        st.write("Connect your Power BI Desktop or Power BI Service to DataSphere")
        
        # Connection methods
        connection_method = st.radio(
            "Connection Method",
            options=["Direct Database Connection", "OData Feed", "Export Data Model"],
            key="powerbi_connection_method"
        )
        
        if connection_method == "Direct Database Connection":
            # Database connection details
            st.subheader("Database Connection Details")
            
            server = st.text_input("Server", value="localhost", disabled=True)
            database = st.text_input("Database", value=st.session_state.get("db_name", ""), disabled=True)
            username = st.text_input("Username", value=st.session_state.get("db_user", ""), disabled=True)
            
            st.info("""
            In Power BI Desktop:
            1. Click 'Get Data' > 'Database' > Select your database type
            2. Enter the connection details shown above
            3. Select the tables you want to import
            """)
        
        elif connection_method == "OData Feed":
            # Generate OData endpoint
            st.subheader("OData Feed URL")
            
            odata_url = f"https://yourdomain.com/api/odata"
            st.code(odata_url)
            
            st.info("""
            In Power BI Desktop:
            1. Click 'Get Data' > 'OData feed'
            2. Enter the OData URL shown above
            3. Select the entities you want to import
            """)
            
            st.warning("Note: OData feed functionality requires server-side implementation.")
        
        elif connection_method == "Export Data Model":
            # Generate schema file for Power BI
            st.subheader("Export Semantic Model")
            
            # Get table schema
            schema = st.session_state.get("db_schema", {})
            
            if not schema:
                st.warning("No schema information available")
                return
            
            # Select tables to include
            table_options = []
            if "tables" in schema:
                table_options = list(schema["tables"].keys())
            elif "collections" in schema:
                table_options = list(schema["collections"].keys())
            
            selected_tables = st.multiselect("Select Tables/Collections to Export", options=table_options)
            
            if selected_tables and st.button("Generate Power BI Template"):
                # In a real implementation, would generate a proper PBIT template
                # Here we'll create a simplified JSON representation
                
                model = {
                    "name": "DataSphere Semantic Model",
                    "tables": []
                }
                
                # Add tables
                for table_name in selected_tables:
                    if "tables" in schema and table_name in schema["tables"]:
                        table_info = schema["tables"][table_name]
                        
                        table = {
                            "name": table_name,
                            "columns": []
                        }
                        
                        # Add columns
                        for column in table_info["columns"]:
                            table["columns"].append({
                                "name": column["name"],
                                "dataType": self._map_sql_type_to_powerbi(column["type"])
                            })
                        
                        model["tables"].append(table)
                
                # Generate JSON
                model_json = json.dumps(model, indent=2)
                
                # Offer download
                st.download_button(
                    label="Download Model JSON",
                    data=model_json,
                    file_name="datasphere_model.json",
                    mime="application/json"
                )
                
                st.info("""
                How to use:
                1. Save this JSON file
                2. In Power BI Desktop, you can use it as a reference for creating measures and relationships
                3. For a complete template, export would include a .pbit file (not implemented in this demo)
                """)
        
        # Power BI Direct Query support
        st.subheader("Power BI Direct Query")
        st.write("Direct Query allows Power BI to query data directly from the source without importing it")
        
        st.warning("""
        Prerequisites for Direct Query:
        1. Database must be accessible from Power BI (VPN or public access)
        2. Performance optimization is crucial for good user experience
        """)
        
        # Performance optimization tips
        with st.expander("Performance Optimization Tips"):
            st.write("""
            **Optimize for Direct Query:**
            - Create appropriate indexes on frequently queried columns
            - Use materialized views for complex aggregations
            - Limit the number of visuals per page in Power BI
            - Use query folding-friendly operations in Power Query
            """)
    
    def _tableau_integration_ui(self, db_manager):
        """UI for Tableau integration"""
        st.write("Connect your Tableau Desktop or Tableau Server to DataSphere")
        
        # Connection methods
        connection_method = st.radio(
            "Connection Method",
            options=["Direct Database Connection", "Web Data Connector", "Extract File"],
            key="tableau_connection_method"
        )
        
        if connection_method == "Direct Database Connection":
            # Database connection details
            st.subheader("Database Connection Details")
            
            server = st.text_input("Server", value="localhost", disabled=True)
            database = st.text_input("Database", value=st.session_state.get("db_name", ""), disabled=True)
            username = st.text_input("Username", value=st.session_state.get("db_user", ""), disabled=True)
            
            st.info("""
            In Tableau Desktop:
            1. Select 'Connect to Data' > Select your database type
            2. Enter the connection details shown above
            3. Select the tables you want to use
            """)
        
        elif connection_method == "Web Data Connector":
            # Web Data Connector details
            st.subheader("Web Data Connector URL")
            
            wdc_url = f"https://yourdomain.com/api/tableau-wdc"
            st.code(wdc_url)
            
            st.info("""
            In Tableau Desktop:
            1. Select 'Connect to Data' > 'Web Data Connector'
            2. Enter the WDC URL shown above
            3. Follow the authentication and configuration steps
            """)
            
            st.warning("Note: Web Data Connector functionality requires server-side implementation.")
        
        elif connection_method == "Extract File":
            # Generate Tableau extract
            st.subheader("Generate Tableau Extract")
            
            # Get table schema
            schema = st.session_state.get("db_schema", {})
            
            if not schema:
                st.warning("No schema information available")
                return
            
            # Select tables to include
            table_options = []
            if "tables" in schema:
                table_options = list(schema["tables"].keys())
            elif "collections" in schema:
                table_options = list(schema["collections"].keys())
            
            selected_tables = st.multiselect("Select Tables/Collections to Extract", options=table_options)
            
            # Select rows limit
            row_limit = st.number_input("Row Limit per Table", min_value=1, max_value=1000000, value=10000)
            
            if selected_tables and st.button("Generate Extract Data"):
                with st.spinner("Generating extract data..."):
                    try:
                        # Create CSV files for each table
                        for table_name in selected_tables:
                            # Query data
                            query = f"""
                            SELECT * FROM {table_name}
                            LIMIT {row_limit}
                            """
                            
                            result = db_manager.execute_query(query)
                            
                            if isinstance(result, pd.DataFrame) and not result.empty:
                                # Convert to CSV
                                csv = result.to_csv(index=False)
                                
                                # Offer download
                                st.download_button(
                                    label=f"Download {table_name} Data",
                                    data=csv,
                                    file_name=f"{table_name}.csv",
                                    mime="text/csv",
                                    key=f"download_{table_name}"
                                )
                    except Exception as e:
                        st.error(f"Error generating extract: {str(e)}")
            
            st.info("""
            How to use:
            1. Download the CSV files for your tables
            2. In Tableau Desktop, connect to 'Text File' data source
            3. Select the downloaded CSV files
            4. Create relationships between tables as needed
            """)
    
    def _looker_integration_ui(self, db_manager):
        """UI for Looker integration"""
        st.write("Connect Looker to DataSphere for advanced analytics")
        
        # Connection methods
        connection_method = st.radio(
            "Connection Method",
            options=["LookML Generation", "Database Connection", "Looker API"],
            key="looker_connection_method"
        )
        
        if connection_method == "LookML Generation":
            # LookML generation
            st.subheader("Generate LookML")
            
            # Get table schema
            schema = st.session_state.get("db_schema", {})
            
            if not schema:
                st.warning("No schema information available")
                return
            
            # Select tables to include
            table_options = []
            if "tables" in schema:
                table_options = list(schema["tables"].keys())
            elif "collections" in schema:
                table_options = list(schema["collections"].keys())
            
            selected_tables = st.multiselect("Select Tables for LookML", options=table_options)
            
            # Generate LookML
            if selected_tables and st.button("Generate LookML"):
                lookml = self._generate_lookml(schema, selected_tables)
                
                # Display LookML
                st.subheader("Generated LookML")
                st.code(lookml, language="yaml")
                
                # Offer download
                st.download_button(
                    label="Download LookML",
                    data=lookml,
                    file_name="datasphere_model.lookml",
                    mime="text/plain"
                )
        elif connection_method == "Database Connection":
            st.subheader("Database Connection Details")
            
            # Display connection details
            server = st.text_input("Server", value="localhost", disabled=True)
            database = st.text_input("Database", value=st.session_state.get("db_name", ""), disabled=True)
            username = st.text_input("Username", value=st.session_state.get("db_user", ""), disabled=True)
            
            st.info("""
            In Looker Admin:
            1. Go to Connections
            2. Create a new connection with your database type
            3. Enter the connection details shown above
            4. Test the connection
            """)
            
        elif connection_method == "Looker API":
            st.subheader("Looker API Integration")
            st.info("This feature allows you to programmatically interact with Looker.")
            
            # API credentials
            client_id = st.text_input("Client ID", type="password")
            client_secret = st.text_input("Client Secret", type="password")
            looker_url = st.text_input("Looker Instance URL", placeholder="https://your-instance.looker.com")
            
            st.warning("API integration requires additional setup on your Looker instance.")
        
        # Looker connection instructions
        with st.expander("Looker Connection Setup Instructions"):
            st.write("""
            **Steps to connect Looker to your database:**
            
            1. **Configure Database Connection in Looker:**
               - In Looker Admin, go to Connections
               - Create a new connection with your database type
               - Enter connection details (host, database, username, password)
               - Test the connection
            
            2. **Create LookML Project:**
               - Create a new LookML project
               - Add the generated LookML files
               - Commit and deploy
            
            3. **Create Explores:**
               - Define Explores to make your data queryable
               - Set up joins between tables
               - Add measures and dimensions as needed
            """)
    
    def _generic_bi_integration_ui(self, db_manager):
        """UI for generic BI tool integration"""
        st.write("Connect any BI tool to DataSphere")
        
        # Connection methods
        connection_method = st.radio(
            "Connection Method",
            options=["Direct Database Connection", "OData/REST API", "File Export"],
            key="generic_connection_method"
        )
        
        if connection_method == "Direct Database Connection":
            # Database connection details
            st.subheader("Database Connection Details")
            
            connection_string = f"Driver={{PostgreSQL}}; Server=localhost; Database={st.session_state.get('db_name', '')}; Uid={st.session_state.get('db_user', '')}; Pwd=********;"
            st.code(connection_string)
            
            st.info("""
            Generic steps for connecting your BI tool:
            1. Look for "Connect to Data" or "Add Data Source" in your BI tool
            2. Select your database type (PostgreSQL, MySQL, etc.)
            3. Enter the connection details or connection string
            4. Select the tables you want to use
            """)
        
        elif connection_method == "OData/REST API":
            # API details
            st.subheader("API Endpoints")
            
            if st.session_state.api_endpoints:
                for name, endpoint in st.session_state.api_endpoints.items():
                    st.write(f"**{endpoint['method']} {endpoint['path']}**")
                    st.write(endpoint.get('description', ''))
            else:
                st.info("No API endpoints defined yet. Create them in the REST API Endpoints tab.")
            
            st.info("""
            Generic steps for connecting to API:
            1. In your BI tool, look for "Web Service" or "REST API" connector
            2. Enter the API URL and authentication details
            3. Configure the data format (usually JSON)
            4. Map the returned data to tables in your BI tool
            """)
        
        elif connection_method == "File Export":
            # File export options
            st.subheader("Export Data")
            
            # Get table schema
            schema = st.session_state.get("db_schema", {})
            
            if not schema:
                st.warning("No schema information available")
                return
            
            # Select tables to include
            table_options = []
            if "tables" in schema:
                table_options = list(schema["tables"].keys())
            elif "collections" in schema:
                table_options = list(schema["collections"].keys())
            
            selected_tables = st.multiselect("Select Tables to Export", options=table_options)
            
            # Select export format
            export_format = st.selectbox(
                "Export Format",
                options=["CSV", "Excel", "JSON", "Parquet"]
            )
            
            if selected_tables and st.button("Export Data"):
                with st.spinner("Exporting data..."):
                    try:
                        for table_name in selected_tables:
                            # Query data
                            query = f"SELECT * FROM {table_name} LIMIT 10000"
                            result = db_manager.execute_query(query)
                            
                            if isinstance(result, pd.DataFrame) and not result.empty:
                                if export_format == "CSV":
                                    data = result.to_csv(index=False)
                                    mime = "text/csv"
                                    ext = "csv"
                                elif export_format == "Excel":
                                    excel_buffer = io.BytesIO()
                                    result.to_excel(excel_buffer, index=False)
                                    data = excel_buffer.getvalue()
                                    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    ext = "xlsx"
                                elif export_format == "JSON":
                                    data = result.to_json(orient="records")
                                    mime = "application/json"
                                    ext = "json"
                                elif export_format == "Parquet":
                                    # Parquet requires additional libraries, so we'll simulate it
                                    st.warning("Parquet export is simulated in this demo")
                                    data = result.to_csv(index=False)
                                    mime = "text/csv"
                                    ext = "csv"  # Would be parquet in a real implementation
                                
                                # Offer download
                                st.download_button(
                                    label=f"Download {table_name} data",
                                    data=data,
                                    file_name=f"{table_name}.{ext}",
                                    mime=mime,
                                    key=f"download_{table_name}"
                                )
                    except Exception as e:
                        st.error(f"Error exporting data: {str(e)}")
    
    def _map_sql_type_to_powerbi(self, sql_type: str) -> str:
        """Map SQL types to Power BI data types"""
        sql_type = sql_type.lower()
        
        if "int" in sql_type:
            return "Int64"
        elif "float" in sql_type or "double" in sql_type or "decimal" in sql_type or "numeric" in sql_type:
            return "Double"
        elif "date" in sql_type:
            return "DateTime"
        elif "time" in sql_type:
            return "DateTime"
        elif "bool" in sql_type:
            return "Boolean"
        else:
            return "String"
    
    def _generate_lookml(self, schema: Dict[str, Any], selected_tables: List[str]) -> str:
        """Generate LookML for selected tables"""
        lookml = "# LookML Model generated by DataSphere\n\n"
        
        for table_name in selected_tables:
            if "tables" in schema and table_name in schema["tables"]:
                table_info = schema["tables"][table_name]
                
                lookml += f"view: {table_name} {{\n"
                
                # Set SQL table name
                lookml += f"  sql_table_name: {table_name} ;;\n\n"
                
                # Add dimensions for each column
                for column in table_info["columns"]:
                    col_name = column["name"]
                    col_type = column["type"].lower()
                    
                    # Determine LookML type
                    lookml_type = "string"
                    if "int" in col_type:
                        lookml_type = "number"
                    elif "float" in col_type or "double" in col_type or "decimal" in col_type or "numeric" in col_type:
                        lookml_type = "number"
                    elif "date" in col_type:
                        lookml_type = "date"
                    elif "time" in col_type:
                        lookml_type = "date_time"
                    elif "bool" in col_type:
                        lookml_type = "yesno"
                    
                    # Add dimension
                    lookml += f"  dimension: {col_name} {{\n"
                    lookml += f"    type: {lookml_type}\n"
                    lookml += f"    sql: ${{TABLE}}.{col_name} ;;\n"
                    
                    # Add primary key label if applicable
                    if col_name in table_info["primary_keys"]:
                        lookml += "    primary_key: yes\n"
                    
                    lookml += "  }\n\n"
                
                # Add count measure
                lookml += "  measure: count {\n"
                lookml += "    type: count\n"
                lookml += "    drill_fields: [id]\n"
                lookml += "  }\n"
                
                # Add sum measures for numeric columns
                for column in table_info["columns"]:
                    col_name = column["name"]
                    col_type = column["type"].lower()
                    
                    # Add sum measure for numeric fields
                    if ("int" in col_type or "float" in col_type or "double" in col_type or 
                        "decimal" in col_type or "numeric" in col_type):
                        lookml += f"\n  measure: total_{col_name} {{\n"
                        lookml += "    type: sum\n"
                        lookml += f"    sql: ${{TABLE}}.{col_name} ;;\n"
                        lookml += "  }\n"
                
                lookml += "}\n\n"
        
        # Add model section
        lookml += "# Model definition\n"
        lookml += "model: datasphere_model {\n"
        
        # Add explores for each table
        for table_name in selected_tables:
            lookml += f"  explore: {table_name} {{\n"
            
            # Add joins based on foreign keys
            if "tables" in schema and table_name in schema["tables"]:
                table_info = schema["tables"][table_name]
                
                for fk in table_info.get("foreign_keys", []):
                    referred_table = fk["referred_table"]
                    if referred_table in selected_tables:
                        join_fields = []
                        
                        for i, col in enumerate(fk["constrained_columns"]):
                            ref_col = fk["referred_columns"][i] if i < len(fk["referred_columns"]) else fk["referred_columns"][0]
                            join_fields.append(f"${{{table_name}.{col}}} = ${{{referred_table}.{ref_col}}}")
                        
                        lookml += f"    join: {referred_table} {{\n"
                        lookml += f"      type: left_outer\n"
                        lookml += f"      relationship: many_to_one\n"
                        lookml += f"      sql_on: {' AND '.join(join_fields)} ;;\n"
                        lookml += "    }\n"
            
            lookml += "  }\n\n"
        
        lookml += "}\n"
        
        return lookml