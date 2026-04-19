import streamlit as st
import os
import pandas as pd

from database_manager import DatabaseManager
from nlp_processor import NLPProcessor
from schema_visualizer import SchemaVisualizer
from data_exporter import DataExporter
from utils import load_session_state, initialize_session_state
from query_optimizer import QueryOptimizer
from schema_advisor import SchemaAdvisor
from semantic_layer import SemanticLayer
from advanced_visualization import AdvancedVisualization
from enterprise_integration import EnterpriseIntegration
from cloud_storage import CloudStorage
from user_management import UserManagement
from semantic_templates import SemanticTemplates
from collaboration import Collaboration
from data_storytelling import DataStorytelling
from database_analyzer import DatabaseAnalyzer

# Check for full screen ER diagram route first
is_full_screen = False
if "view" in st.query_params and st.query_params["view"] == "full_screen_er_diagram":
    is_full_screen = True

try:
    if is_full_screen:
        # Set page config for full screen view
        st.set_page_config(
            page_title="Database Schema - Full Screen View",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
    else:
        # Regular page configuration
        st.set_page_config(
            page_title="DataSphere: Text-to-SQL Analytics Platform",
            page_icon="🔍",
            layout="wide"
        )
except Exception:
    pass

# Custom CSS for dark theme notifications
st.markdown("""
<style>
    /* Dark theme for info messages */
    .stAlert.st-info {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #2c2c2c;
    }
    
    /* Dark theme for success messages */
    .stAlert.st-success {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #2c2c2c;
    }
    
    /* Dark theme for warning messages */
    .stAlert.st-warning {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #2c2c2c;
    }
    
    /* Dark theme for error messages */
    .stAlert.st-error {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #2c2c2c;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

# Load any saved session state
load_session_state()

# Handle full screen ER diagram route
if is_full_screen:
    if st.session_state.get("db_schema"):
        # Add full screen specific styles
        st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            .block-container {padding-top: 0; padding-bottom: 0; max-width: 100% !important;}
        </style>
        """, unsafe_allow_html=True)
        
        # Add back button
        st.markdown("""
        <div style="margin: 20px;">
            <a href="/" style="text-decoration: none; color: #4D7A97; display: flex; align-items: center; width: fit-content;">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 5px;">
                    <path d="M19 12H5M12 19l-7-7 7-7"/>
                </svg>
                Back to Dashboard
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        schema_visualizer = SchemaVisualizer()
        schema_visualizer.display_full_screen_er_diagram_content(st.session_state.db_schema)
        st.stop()
    else:
        st.error("No database schema available. Please connect to a database first.")
        if st.button("Go to Dashboard"):
            st.query_params.clear()
            st.rerun()
        st.stop()

# Sidebar navigation
st.sidebar.title("DataSphere")
st.sidebar.caption("Text-to-SQL Analytics Platform")

# Navigation options - now with advanced features
nav_option = st.sidebar.radio(
    "Navigate to:",
    [
        "Home", 
        "Database Connections", 
        "Query Generator", 
        "Query Optimization",
        "Schema Visualization",
        "Schema Advisor",
        "Database Analysis",
        "Semantic Layer",
        "Semantic Templates",
        "Advanced Visualization",
        "Data Storytelling",
        "Enterprise Integration",
        "Cloud Storage",
        "Collaboration Hub",
        "User Management",
        "Query History"
    ]
)

# Database connection manager
db_manager = DatabaseManager()

# Natural language processor
nlp_processor = NLPProcessor()

# Schema visualizer
schema_visualizer = SchemaVisualizer()

# Data exporter
data_exporter = DataExporter()

# Advanced feature modules
query_optimizer = QueryOptimizer()
schema_advisor = SchemaAdvisor()
semantic_layer = SemanticLayer()
advanced_visualization = AdvancedVisualization()
enterprise_integration = EnterpriseIntegration()

# New feature modules
cloud_storage = CloudStorage()
user_management = UserManagement()
semantic_templates = SemanticTemplates()
collaboration = Collaboration()
data_storytelling = DataStorytelling()
database_analyzer = DatabaseAnalyzer()

# Make modules available in session state for cross-module access
st.session_state.collaboration = collaboration
st.session_state.user_management = user_management
st.session_state.data_storytelling = data_storytelling

# Home page
if nav_option == "Home":
    st.title("Welcome to DataSphere")
    st.subheader("Advanced Text-to-SQL Analytics Platform")
    
    st.markdown("""
    DataSphere helps you analyze your data by converting natural language to SQL queries
    and connecting to multiple database types.
    
    **Key Features:**
    * Connect to PostgreSQL, MySQL, SQLite, and MongoDB databases
    * Convert natural language to SQL queries using AI (Groq or Gemini)
    * Execute queries and view results
    * Visualize database schema with ER diagrams
    * Export data in various formats (CSV, Excel, JSON)
    
    **Advanced Enterprise Features:**
    * Query optimization engine with execution plan analysis
    * AI-powered schema recommendations and data model generation
    * Semantic layer for business-friendly data modeling
    * Advanced data visualization dashboard with forecasting
    * Enterprise integration with REST API endpoints and BI tools
    
    **New Features:**
    * Cloud storage integration with AWS S3, Google Cloud Storage, and Azure
    * User management with role-based access control
    * Semantic layer templates for common business domains
    * Collaboration Hub with shared workspaces and version control
    * Comments and annotations on queries, models, and dashboards
    * Real-time notifications for team collaboration
    * Advanced Data Storytelling with interactive presentations
    * AI-generated narrative insights from your data
    * Annotated visualizations with context and explanations
    * **Database Analysis & Intelligence Report** — full database profiling with data quality, duplicate detection, and AI insights
    
    Get started by navigating to the **Database Connections** section to set up your database connection.
    """)
    
    # Display current connection status
    if st.session_state.get("connected_db"):
        st.success(f"✅ Currently connected to: {st.session_state.get('connected_db')}")
    else:
        st.info("⚠️ No active database connection. Please set up a connection to continue.")

# Database Connections page
elif nav_option == "Database Connections":
    st.title("Database Connections")
    
    connection_tab1, connection_tab2 = st.tabs(["Create Connection", "Manage Connections"])
    
    with connection_tab1:
        db_manager.create_connection_ui()
    
    with connection_tab2:
        db_manager.manage_connections_ui()

# Query Generator page
elif nav_option == "Query Generator":
    st.title("Text-to-SQL Query Generator")
    
    # Check if connected to a database
    if not st.session_state.get("connected_db"):
        st.warning("Please connect to a database first.")
        st.button("Go to Database Connections", on_click=lambda: st.session_state.update({"nav_option": "Database Connections"}))
    else:
        query_tab1, query_tab2, query_tab3 = st.tabs(["Text to SQL", "SQL Editor", "Query Results"])
        
        with query_tab1:
            nlp_processor.text_to_sql_ui(db_manager)
        
        with query_tab2:
            nlp_processor.sql_editor_ui(db_manager)
        
        with query_tab3:
            results = st.session_state.get("query_results")
            if results is not None:
                st.subheader("Query Results")

                if isinstance(results, pd.DataFrame):
                    st.caption(f"{len(results)} row(s) returned")
                    st.dataframe(results, width="stretch")

                    # Export options
                    st.subheader("Export Results")
                    data_exporter.export_ui(results)
                else:
                    # Non-SELECT result (INSERT/UPDATE/DELETE success message)
                    st.success(str(results))

                # Button to clear results
                if st.button("Clear Results", key="clear_results_btn"):
                    st.session_state.query_results = None
                    st.rerun()
            else:
                st.info("No query results to display. Execute a query from the Text to SQL or SQL Editor tab first.")

# Query Optimization page
elif nav_option == "Query Optimization":
    st.title("Advanced Query Optimization")
    
    # Check if connected to a database
    if not st.session_state.get("connected_db"):
        st.warning("Please connect to a database first.")
        st.button("Go to Database Connections", on_click=lambda: st.session_state.update({"nav_option": "Database Connections"}))
    else:
        query_optimizer.optimize_query_ui(db_manager)

# Schema Visualization page
elif nav_option == "Schema Visualization":
    st.title("Database Schema Visualization")
    
    # Check if connected to a database
    if not st.session_state.get("connected_db"):
        st.warning("Please connect to a database first.")
        st.button("Go to Database Connections", on_click=lambda: st.session_state.update({"nav_option": "Database Connections"}))
    else:
        schema_visualizer.visualize_schema_ui(db_manager)

# Schema Advisor page
elif nav_option == "Schema Advisor":
    st.title("AI-Powered Schema Advisor")
    
    # Check if connected to a database
    if not st.session_state.get("connected_db"):
        st.warning("Please connect to a database first.")
        st.button("Go to Database Connections", on_click=lambda: st.session_state.update({"nav_option": "Database Connections"}))
    else:
        schema_advisor.schema_advisor_ui(db_manager)

# Database Analysis page
elif nav_option == "Database Analysis":
    st.title("🔍 Database Analysis & Intelligence Report")
    
    # Check if connected to a database
    if not st.session_state.get("connected_db"):
        st.warning("Please connect to a database first.")
        st.button("Go to Database Connections", on_click=lambda: st.session_state.update({"nav_option": "Database Connections"}))
    else:
        database_analyzer.database_analysis_ui(db_manager)

# Semantic Layer page
elif nav_option == "Semantic Layer":
    st.title("Business Semantic Layer")
    
    # Check if connected to a database
    if not st.session_state.get("connected_db"):
        st.warning("Please connect to a database first.")
        st.button("Go to Database Connections", on_click=lambda: st.session_state.update({"nav_option": "Database Connections"}))
    else:
        semantic_layer.semantic_layer_ui(db_manager)

# Semantic Templates page
elif nav_option == "Semantic Templates":
    st.title("Semantic Layer Templates")
    
    # Check if connected to a database
    if not st.session_state.get("connected_db"):
        st.warning("Please connect to a database first.")
        st.button("Go to Database Connections", on_click=lambda: st.session_state.update({"nav_option": "Database Connections"}))
    else:
        semantic_templates.semantic_templates_ui(semantic_layer)

# Advanced Visualization page
elif nav_option == "Advanced Visualization":
    st.title("Advanced Data Visualization")
    
    # Check if connected to a database
    if not st.session_state.get("connected_db"):
        st.warning("Please connect to a database first.")
        st.button("Go to Database Connections", on_click=lambda: st.session_state.update({"nav_option": "Database Connections"}))
    else:
        advanced_visualization.visualization_ui(db_manager)

# Data Storytelling page
elif nav_option == "Data Storytelling":
    st.title("Advanced Data Storytelling")
    
    # Check if connected to a database
    if not st.session_state.get("connected_db"):
        st.warning("Please connect to a database first.")
        st.button("Go to Database Connections", on_click=lambda: st.session_state.update({"nav_option": "Database Connections"}))
    else:
        data_storytelling.storytelling_ui(db_manager, advanced_visualization)

# Enterprise Integration page
elif nav_option == "Enterprise Integration":
    st.title("Enterprise Integration Features")
    
    # Check if connected to a database
    if not st.session_state.get("connected_db"):
        st.warning("Please connect to a database first.")
        st.button("Go to Database Connections", on_click=lambda: st.session_state.update({"nav_option": "Database Connections"}))
    else:
        enterprise_integration.integration_ui(db_manager)

# Cloud Storage page
elif nav_option == "Cloud Storage":
    st.title("Cloud Storage Integration")
    cloud_storage.cloud_storage_ui()

# User Management page
elif nav_option == "User Management":
    st.title("User Management & Permissions")
    user_management.user_management_ui()

# Collaboration Hub page
elif nav_option == "Collaboration Hub":
    st.title("Collaboration Hub")
    collaboration.collaboration_ui()

# Query History page
elif nav_option == "Query History":
    st.title("Query History")
    
    # Check if there are any queries in history
    if not st.session_state.get("query_history"):
        st.info("No query history available.")
    else:
        # Display query history
        for i, query_item in enumerate(st.session_state.query_history):
            with st.expander(f"Query {i+1}: {query_item['timestamp']}"):
                st.code(query_item["query"], language="sql")
                st.write("Database:", query_item["database"])
                
                # Option to re-run query
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button(f"Re-run Query", key=f"rerun_{i}"):
                        st.session_state.current_query = query_item["query"]
                        db_manager.execute_query(query_item["query"])
                        st.rerun()
                
                with col2:
                    if st.button(f"Optimize Query", key=f"optimize_{i}"):
                        st.session_state.current_query = query_item["query"]
                        st.session_state.nav_option = "Query Optimization"
                        st.rerun()

# User info in sidebar
if hasattr(user_management, "get_current_user") and user_management.get_current_user():
    st.sidebar.markdown("---")
    
    # User info
    current_user = user_management.get_current_user()
    st.sidebar.write(f"**Logged in as:** {current_user}")
    
    # Notifications
    if hasattr(st.session_state, "notifications"):
        # Count unread notifications for current user
        unread_count = sum(
            1 for n in st.session_state.notifications 
            if n.get("for_user") == current_user and not n.get("read", False)
        )
        
        # Display notification indicator
        if unread_count > 0:
            st.sidebar.markdown(
                f"""
                <div style="background-color: #1e1e1e; padding: 8px; border-radius: 5px; margin-top: 10px;">
                    <span style="color: white; font-weight: bold;">
                        {unread_count} unread notification{'s' if unread_count > 1 else ''}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if st.sidebar.button("View Notifications"):
                st.session_state.nav_option = "Collaboration Hub"
                st.rerun()
    
    # Logout button
    if st.sidebar.button("Logout"):
        user_management.logout()
        st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("DataSphere Analytics Platform")
st.sidebar.caption("Version 3.0 Enterprise Edition")