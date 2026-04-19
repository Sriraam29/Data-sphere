import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io
import plotly.graph_objects as go
import base64
import streamlit.components.v1 as components
import graphviz
import re
import numpy as np
from typing import Dict, Any, List

class SchemaVisualizer:
    def __init__(self):
        pass
    
    def visualize_schema_ui(self, db_manager):
        """UI for schema visualization"""
        if not st.session_state.db_schema:
            st.warning("No database schema available. Please connect to a database first.")
            return
        
        schema = st.session_state.db_schema
        
        # Schema visualization options
        st.subheader("Database Schema")
        visualization_type = st.radio(
            "Visualization Type",
            ["Table List", "Entity Relationship Diagram", "Graphviz ER Diagram"],
            key="schema_viz_type"
        )
        
        # Display schema based on selected visualization type
        if visualization_type == "Table List":
            self.display_table_list(schema)
        elif visualization_type == "Entity Relationship Diagram":
            self.display_er_diagram(schema)
        elif visualization_type == "Graphviz ER Diagram":
            self.display_graphviz_er_diagram(schema)
    
    def display_table_list(self, schema):
        """Display database schema as a list of tables"""
        if "tables" in schema:
            # SQL-based databases (PostgreSQL, MySQL, SQLite)
            for table_name, table_info in schema["tables"].items():
                with st.expander(f"Table: {table_name}", expanded=False):
                    # Prepare data for table display
                    columns_data = []
                    for column in table_info["columns"]:
                        is_pk = "✓" if column["name"] in table_info["primary_keys"] else ""
                        
                        # Check if column is a foreign key
                        is_fk = ""
                        fk_reference = ""
                        for fk in table_info["foreign_keys"]:
                            if column["name"] in fk["constrained_columns"]:
                                is_fk = "✓"
                                fk_reference = f"{fk['referred_table']}.{', '.join(fk['referred_columns'])}"
                                break
                        
                        columns_data.append({
                            "Column Name": column["name"],
                            "Data Type": column["type"],
                            "Primary Key": is_pk,
                            "Foreign Key": is_fk,
                            "References": fk_reference
                        })
                    
                    # Display as a table
                    if columns_data:
                        df = pd.DataFrame(columns_data)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No columns found for this table.")
        
        elif "collections" in schema:
            # MongoDB
            for collection_name, collection_info in schema["collections"].items():
                with st.expander(f"Collection: {collection_name}", expanded=False):
                    # Prepare data for field display
                    if collection_info["fields"]:
                        fields_data = [{"Field Name": field["name"], "Data Type": field["type"]} 
                                      for field in collection_info["fields"]]
                        df = pd.DataFrame(fields_data)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No fields found for this collection.")
    


    def display_er_diagram(self, schema):
        """Display database schema as an ER diagram using SVG with explicit connection lines"""
        if "tables" not in schema:
            st.info("ER diagram visualization is only available for SQL-based databases.")
            return
        
        # Check if there are tables to visualize
        if not schema["tables"]:
            st.info("No tables found to generate ER diagram.")
            return
        
        # Create settings column
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Diagram Settings")
            
            # Layout direction
            direction = st.radio(
                "Layout Direction",
                ["Horizontal", "Vertical"],
                index=0,
                help="Arrange tables horizontally or vertically"
            )
            
            # Color theme
            theme = st.selectbox(
                "Color Theme",
                ["Blue", "Green", "Purple", "Orange", "Red"],
                index=0,
                help="Color theme for the diagram"
            )
            
            # Show columns
            show_columns = st.checkbox("Show Columns", value=True,
                                     help="Show column details in tables")
            
            # Generate button
            generate_button = st.button("Generate Diagram", use_container_width=True)
        
        with col2:
            if generate_button or True:  # Always show for now
                # Set theme colors
                if theme == "Blue":
                    header_bg = "#4D7A97"
                    line_color = "#2B5F82"
                    badge_pk = "#FF5722"
                    badge_fk = "#2196F3"
                elif theme == "Green":
                    header_bg = "#4CAF50"
                    line_color = "#2E7D32"
                    badge_pk = "#FF9800"
                    badge_fk = "#2196F3"
                elif theme == "Purple":
                    header_bg = "#673AB7"
                    line_color = "#4527A0"
                    badge_pk = "#FF5722"
                    badge_fk = "#2196F3"
                elif theme == "Orange":
                    header_bg = "#FF9800"
                    line_color = "#E65100"
                    badge_pk = "#E91E63"
                    badge_fk = "#2196F3"
                else:  # Red
                    header_bg = "#F44336"
                    line_color = "#B71C1C"
                    badge_pk = "#FFC107"
                    badge_fk = "#2196F3"
                
                # Calculate table positions and sizes
                table_width = 220
                table_header_height = 40
                row_height = 30
                
                # Collect table info for positioning
                tables_data = {}
                max_rows = 0
                
                for table_name, table_info in schema["tables"].items():
                    num_rows = len(table_info["columns"]) if show_columns else 0
                    tables_data[table_name] = {
                        "num_rows": num_rows,
                        "height": table_header_height + (num_rows * row_height),
                        "foreign_keys": table_info["foreign_keys"]
                    }
                    max_rows = max(max_rows, num_rows)
                
                # Calculate layout
                num_tables = len(tables_data)
                
                if direction == "Horizontal":
                    # Horizontal layout (tables side by side)
                    svg_width = (table_width + 50) * num_tables
                    svg_height = table_header_height + (max_rows * row_height) + 200  # Extra space for arrows
                    
                    # Position tables
                    x_pos = 50
                    for table_name in tables_data:
                        tables_data[table_name]["x"] = x_pos
                        tables_data[table_name]["y"] = 50
                        x_pos += table_width + 50
                else:
                    # Vertical layout (tables stacked)
                    svg_width = table_width * 3 + 100  # Allow for 3 columns
                    total_height = 0
                    for table_data in tables_data.values():
                        total_height += table_data["height"] + 50
                    svg_height = max(total_height, 600)
                    
                    # Position tables in a grid-like layout
                    x_pos, y_pos = 50, 50
                    col_count = 0
                    for table_name in tables_data:
                        if col_count >= 3:  # Start a new row after 3 tables
                            col_count = 0
                            x_pos = 50
                            y_pos += 300  # Move down for next row
                        
                        tables_data[table_name]["x"] = x_pos
                        tables_data[table_name]["y"] = y_pos
                        x_pos += table_width + 50
                        col_count += 1
                
                # Start SVG
                svg = f"""
                <svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="{line_color}"/>
                        </marker>
                    </defs>
                    <rect width="100%" height="100%" fill="#f8f9fa"/>
                """
                
                # Draw relationship lines first (so they appear behind tables)
                for source_table, table_info in tables_data.items():
                    for fk in table_info["foreign_keys"]:
                        target_table = fk["referred_table"]
                        
                        if target_table in tables_data:
                            # Get positions
                            x1 = table_info["x"] + table_width / 2
                            y1 = table_info["y"] + table_info["height"]
                            x2 = tables_data[target_table]["x"] + table_width / 2
                            y2 = tables_data[target_table]["y"]
                            
                            # Adjust for self-referencing tables
                            if source_table == target_table:
                                svg += f"""
                                <path d="M {x1} {y1} 
                                         C {x1} {y1+50}, {x1+100} {y1+50}, {x1+100} {y1} 
                                         C {x1+100} {y1-50}, {x1+50} {y1-50}, {x1} {y1-table_header_height}" 
                                      stroke="{line_color}" fill="none" stroke-width="2" marker-end="url(#arrowhead)"/>
                                """
                            else:
                                # Calculate control points for a curved line
                                # For horizontal layout, curve more if tables are far apart
                                distance = abs(x2 - x1)
                                curve_factor = min(100, distance / 3)
                                
                                # Draw curved line with arrow
                                svg += f"""
                                <path d="M {x1} {y1} 
                                         C {x1} {y1+curve_factor}, {x2} {y2-curve_factor}, {x2} {y2}" 
                                      stroke="{line_color}" fill="none" stroke-width="2" marker-end="url(#arrowhead)"/>
                                """
                                
                                # Add relationship label
                                mid_x = (x1 + x2) / 2
                                mid_y = (y1 + y2) / 2 + 15
                                
                                source_cols = ", ".join(fk["constrained_columns"])
                                target_cols = ", ".join(fk["referred_columns"])
                                label = f"{source_cols} → {target_cols}"
                                
                                svg += f"""
                                <rect x="{mid_x-5}" y="{mid_y-15}" width="10" height="20" fill="white" opacity="0.8" rx="5" ry="5"/>
                                <text x="{mid_x}" y="{mid_y}" font-family="Arial" font-size="12" text-anchor="middle" fill="{line_color}">{label}</text>
                                """
                
                # Draw tables
                for table_name, table_info in schema["tables"].items():
                    table_data = tables_data[table_name]
                    x = table_data["x"]
                    y = table_data["y"]
                    
                    # Table header
                    svg += f"""
                    <rect x="{x}" y="{y}" width="{table_width}" height="{table_header_height}" fill="{header_bg}" rx="5" ry="5"/>
                    <text x="{x + table_width/2}" y="{y + table_header_height/2 + 5}" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="white">{table_name}</text>
                    """
                    
                    if show_columns:
                        # Table body background
                        body_height = table_data["num_rows"] * row_height
                        svg += f"""
                        <rect x="{x}" y="{y + table_header_height}" width="{table_width}" height="{body_height}" fill="white" stroke="#ccc" stroke-width="1"/>
                        """
                        
                        # Column headers
                        svg += f"""
                        <line x1="{x}" y1="{y + table_header_height + row_height}" x2="{x + table_width}" y2="{y + table_header_height + row_height}" stroke="#ddd" stroke-width="1"/>
                        <text x="{x + 10}" y="{y + table_header_height + row_height/2 + 5}" font-family="Arial" font-size="12" font-weight="bold" fill="#333">Column</text>
                        <text x="{x + 110}" y="{y + table_header_height + row_height/2 + 5}" font-family="Arial" font-size="12" font-weight="bold" fill="#333">Type</text>
                        <text x="{x + 180}" y="{y + table_header_height + row_height/2 + 5}" font-family="Arial" font-size="12" font-weight="bold" fill="#333">Key</text>
                        """
                        
                        # Columns
                        for i, column in enumerate(table_info["columns"]):
                            row_y = y + table_header_height + row_height + (i * row_height)
                            
                            # Alternating row background
                            if i % 2 == 1:
                                svg += f"""
                                <rect x="{x}" y="{row_y}" width="{table_width}" height="{row_height}" fill="#f9f9f9"/>
                                """
                            
                            # Column name
                            svg += f"""
                            <text x="{x + 10}" y="{row_y + row_height/2 + 5}" font-family="Arial" font-size="12" fill="#333">{column["name"]}</text>
                            """
                            
                            # Data type
                            data_type = column["type"]
                            if len(data_type) > 10:
                                data_type = data_type[:8] + "..."
                            svg += f"""
                            <text x="{x + 110}" y="{row_y + row_height/2 + 5}" font-family="Arial" font-size="12" fill="#666">{data_type}</text>
                            """
                            
                            # Key indicators
                            key_x = x + 180
                            
                            # Primary key
                            if column["name"] in table_info["primary_keys"]:
                                svg += f"""
                                <rect x="{key_x}" y="{row_y + row_height/2 - 8}" width="25" height="16" rx="3" ry="3" fill="{badge_pk}"/>
                                <text x="{key_x + 12.5}" y="{row_y + row_height/2 + 5}" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle" fill="white">PK</text>
                                """
                                key_x += 30
                            
                            # Foreign key
                            is_fk = False
                            for fk in table_info["foreign_keys"]:
                                if column["name"] in fk["constrained_columns"]:
                                    is_fk = True
                                    break
                            
                            if is_fk:
                                svg += f"""
                                <rect x="{key_x}" y="{row_y + row_height/2 - 8}" width="25" height="16" rx="3" ry="3" fill="{badge_fk}"/>
                                <text x="{key_x + 12.5}" y="{row_y + row_height/2 + 5}" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle" fill="white">FK</text>
                                """
                            
                            # Row separator
                            svg += f"""
                            <line x1="{x}" y1="{row_y + row_height}" x2="{x + table_width}" y2="{row_y + row_height}" stroke="#ddd" stroke-width="1"/>
                            """
                
                # Close SVG
                svg += """
                </svg>
                """
                
                # Display in Streamlit
                st.components.v1.html(f"""
                <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; overflow: auto;">
                    {svg}
                </div>
                """, height=min(svg_height + 50, 700), scrolling=True)
                
                # Create a standalone HTML file for full-screen view
                standalone_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Database Schema Diagram</title>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            margin: 0;
                            padding: 20px;
                            background-color: #f8f9fa;
                        }}
                        .container {{
                            max-width: 100%;
                            overflow: auto;
                            background-color: white;
                            border-radius: 5px;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                            padding: 20px;
                        }}
                        h1 {{
                            color: {header_bg};
                            margin-top: 0;
                        }}
                        .toolbar {{
                            margin-bottom: 20px;
                        }}
                        .btn {{
                            background-color: {header_bg};
                            color: white;
                            border: none;
                            padding: 10px 15px;
                            border-radius: 4px;
                            cursor: pointer;
                            margin-right: 10px;
                        }}
                        .btn:hover {{
                            opacity: 0.9;
                        }}
                        @media print {{
                            .toolbar {{
                                display: none;
                            }}
                        }}
                        
                        /* Add zoom controls */
                        .zoom-controls {{
                            position: fixed;
                            top: 20px;
                            right: 20px;
                            background-color: white;
                            border-radius: 5px;
                            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                            padding: 10px;
                            z-index: 1000;
                        }}
                        .zoom-btn {{
                            background-color: {header_bg};
                            color: white;
                            border: none;
                            width: 30px;
                            height: 30px;
                            border-radius: 4px;
                            cursor: pointer;
                            margin: 0 5px;
                            font-size: 16px;
                            font-weight: bold;
                        }}
                    </style>
                </head>
                <body>
                    <div class="toolbar">
                        <button class="btn" onclick="window.print()">Print Diagram</button>
                        <button class="btn" onclick="window.close()">Close</button>
                    </div>
                    
                    <div class="zoom-controls">
                        <button class="zoom-btn" onclick="zoomIn()">+</button>
                        <button class="zoom-btn" onclick="zoomOut()">-</button>
                        <button class="zoom-btn" onclick="resetZoom()">↺</button>
                    </div>
                    
                    <h1>Database Schema Diagram</h1>
                    
                    <div class="container" id="diagram-container">
                        {svg}
                    </div>
                    
                    <script>
                        // Zoom functionality
                        let scale = 1;
                        const container = document.getElementById('diagram-container');
                        const svg = container.querySelector('svg');
                        
                        function updateZoom() {{
                            svg.style.transform = `scale(${{scale}})`;
                            svg.style.transformOrigin = 'top left';
                        }}
                        
                        function zoomIn() {{
                            scale += 0.1;
                            updateZoom();
                        }}
                        
                        function zoomOut() {{
                            if (scale > 0.3) {{
                                scale -= 0.1;
                                updateZoom();
                            }}
                        }}
                        
                        function resetZoom() {{
                            scale = 1;
                            updateZoom();
                        }}
                    </script>
                </body>
                </html>
                """
                
                # Save to a file for full-screen view
                import tempfile
                import os
                
                temp_dir = tempfile.gettempdir()
                html_path = os.path.join(temp_dir, "database_schema.html")
                
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(standalone_html)
                
                # Add a button to open in full screen
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("Open in Full Screen", use_container_width=True):
                        st.query_params["view"] = "full_screen_er_diagram"
                        st.rerun()
                
                # Also add a direct link to the HTML file
                st.markdown(f"""
                <div style="text-align: right; margin-top: 10px;">
                    <a href="file://{html_path}" target="_blank" style="
                        display: inline-block;
                        background-color: {header_bg};
                        color: white;
                        padding: 8px 15px;
                        text-decoration: none;
                        border-radius: 4px;
                        font-size: 14px;
                    ">
                        Open HTML File Directly
                    </a>
                </div>
                """, unsafe_allow_html=True)
                
                # Also provide a download button for the HTML file
                st.download_button(
                    label="Download HTML File",
                    data=standalone_html,
                    file_name="database_schema.html",
                    mime="text/html"
                )

                
    def display_graphviz_er_diagram(self, schema: Dict[str, Any]):
        """Display database schema as an ER diagram using Graphviz"""
        if "tables" not in schema:
            st.info("ER diagram visualization is only available for SQL-based databases.")
            return
        
        # Create a new Graphviz graph
        graph = graphviz.Digraph(
            name="ER Diagram", 
            engine="dot",
            node_attr={
                'shape': 'plaintext',
                'fontname': 'Arial',
                'fontsize': '12'
            },
            edge_attr={
                'fontname': 'Arial',
                'fontsize': '10',
                'arrowhead': 'crow',
                'arrowtail': 'none',
                'dir': 'both'
            },
            graph_attr={
                'bgcolor': 'white',
                'rankdir': 'LR',
                'splines': 'polyline',
                'concentrate': 'true'
            }
        )
        
        # Add nodes (tables) with HTML-like labels to show table structure
        for table_name, table_info in schema["tables"].items():
            # Create HTML-like label for table
            label = f'''<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                <TR>
                    <TD COLSPAN="4" BGCOLOR="#4D7A97" COLOR="white"><B>{table_name}</B></TD>
                </TR>
                <TR>
                    <TD BGCOLOR="#E9E9E9"><B>Column</B></TD>
                    <TD BGCOLOR="#E9E9E9"><B>Type</B></TD>
                    <TD BGCOLOR="#E9E9E9"><B>PK</B></TD>
                    <TD BGCOLOR="#E9E9E9"><B>FK</B></TD>
                </TR>
            '''
            
            # Add rows for each column
            for column in table_info["columns"]:
                # Check if column is a primary key
                is_pk = "✓" if column["name"] in table_info["primary_keys"] else ""
                
                # Check if column is a foreign key
                is_fk = ""
                for fk in table_info["foreign_keys"]:
                    if column["name"] in fk["constrained_columns"]:
                        is_fk = "✓"
                        break
                
                # Add row for this column with different background for PK/FK columns
                bg_color = "#E8F2FE" if is_pk or is_fk else "white"
                
                # Add row for this column
                label += f'''
                <TR BGCOLOR="{bg_color}">
                    <TD ALIGN="LEFT" PORT="{column["name"]}">{column["name"]}</TD>
                    <TD ALIGN="LEFT">{column["type"]}</TD>
                    <TD ALIGN="CENTER">{is_pk}</TD>
                    <TD ALIGN="CENTER">{is_fk}</TD>
                </TR>
                '''
            
            # Close the HTML table
            label += '''
            </TABLE>
            >'''
            
            # Add node for this table
            graph.node(table_name, label=label)
        
        # Add edges (relationships)
        for table_name, table_info in schema["tables"].items():
            for fk in table_info["foreign_keys"]:
                # Create edge label
                edge_label = f"{', '.join(fk['constrained_columns'])} → {', '.join(fk['referred_columns'])}"
                
                # Add edge with more precise connection points using ports
                # Use the column name as the port for more precise connections
                source_port = fk["constrained_columns"][0]
                target_port = fk["referred_columns"][0]
                
                # Add edge
                graph.edge(
                    f"{table_name}:{source_port}:e", 
                    f"{fk['referred_table']}:{target_port}:w",
                    label=edge_label,
                    color="#4D7A97",
                    penwidth="1.5"
                )
        
        # Display options
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Diagram Settings")
            
            # Layout options
            layout_engine = st.selectbox(
                "Layout Engine",
                options=["dot", "neato", "fdp", "sfdp", "circo", "twopi"],
                index=0,
                key="graphviz_layout_engine",
                help="Different layout algorithms for arranging the diagram"
            )
            
            # Update graph engine
            graph.engine = layout_engine
            
            # Direction options
            direction = st.selectbox(
                "Direction",
                options=["LR", "TB", "RL", "BT"],
                index=0,
                key="graphviz_direction",
                help="LR=Left to Right, TB=Top to Bottom, RL=Right to Left, BT=Bottom to Top"
            )
            
            # Update graph direction
            graph.graph_attr["rankdir"] = direction
            
            # Node spacing
            node_spacing = st.slider(
                "Node Spacing",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="graphviz_node_spacing",
                help="Horizontal spacing between nodes"
            )
            
            # Update node spacing
            graph.graph_attr["nodesep"] = str(node_spacing)
            
            # Rank spacing
            rank_spacing = st.slider(
                "Rank Spacing",
                min_value=0.1,
                max_value=5.0,
                value=0.5,
                step=0.1,
                key="graphviz_rank_spacing",
                help="Vertical spacing between ranks"
            )
            
            # Update rank spacing
            graph.graph_attr["ranksep"] = str(rank_spacing)
            
            # Edge style
            edge_style = st.selectbox(
                "Edge Style",
                options=["polyline", "spline", "ortho", "curved"],
                index=0,
                key="graphviz_edge_style",
                help="Style of the relationship lines"
            )
            
            # Update edge style
            graph.graph_attr["splines"] = edge_style
            
            # Show table details
            show_details = st.checkbox(
                "Show Column Details",
                value=True,
                key="graphviz_show_details",
                help="Show column details in tables"
            )
            
            # Theme selector
            theme = st.selectbox(
                "Color Theme",
                options=["Blue", "Green", "Purple", "Orange", "Monochrome"],
                index=0,
                key="graphviz_theme",
                help="Color theme for the diagram"
            )
            
            # If not showing details, regenerate the graph with simplified nodes
            if not show_details:
                # Clear the graph
                graph = graphviz.Digraph(
                    name="ER Diagram", 
                    engine=layout_engine,
                    node_attr={
                        'shape': 'box',
                        'style': 'filled',
                        'fillcolor': '#4D7A97',
                        'fontcolor': 'white',
                        'fontname': 'Arial',
                        'fontsize': '12',
                        'margin': '0.2,0.1'
                    },
                    edge_attr={
                        'fontname': 'Arial',
                        'fontsize': '10',
                        'arrowhead': 'crow',
                        'arrowtail': 'none',
                        'dir': 'both',
                        'color': '#4D7A97',
                        'penwidth': '1.5'
                    },
                    graph_attr={
                        'bgcolor': 'white',
                        'rankdir': direction,
                        'splines': edge_style,
                        'concentrate': 'true',
                        'nodesep': str(node_spacing),
                        'ranksep': str(rank_spacing)
                    }
                )
                
                # Add simplified nodes
                for table_name in schema["tables"].keys():
                    graph.node(table_name)
                
                # Add edges
                for table_name, table_info in schema["tables"].items():
                    for fk in table_info["foreign_keys"]:
                        edge_label = f"{', '.join(fk['constrained_columns'])} → {', '.join(fk['referred_columns'])}"
                        graph.edge(
                            table_name, 
                            fk["referred_table"],
                            label=edge_label
                        )
        
        with col2:
            # Render the graph
            if len(schema["tables"]) > 0:
                try:
                    # Add a title
                    st.subheader("Entity Relationship Diagram")
                    
                    # Display the graph
                    st.graphviz_chart(graph, use_container_width=True)
                    
                    # Export section
                    st.subheader("Export Options")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Export format
                        export_format = st.selectbox(
                            "Format",
                            options=["PDF", "PNG", "SVG"],
                            index=0,
                            key="graphviz_export_format"
                        )
                    
                    with col2:
                        # Custom filename
                        db_name = st.session_state.get("connected_db", "database")
                        default_filename = f"{db_name}_er_diagram"
                        filename = st.text_input(
                            "Filename",
                            value=default_filename,
                            key="graphviz_export_filename"
                        )
                    
                    with col3:
                        # Generate the appropriate file extension
                        file_ext = export_format.lower()
                        
                        # Export button
                        if st.button("Export Diagram", key="export_graphviz_diagram"):
                            # Render to the selected format
                            try:
                                if export_format == "PDF":
                                    graph_data = graph.pipe(format="pdf")
                                    mime_type = "application/pdf"
                                elif export_format == "PNG":
                                    graph_data = graph.pipe(format="png")
                                    mime_type = "image/png"
                                else:  # SVG
                                    graph_data = graph.pipe(format="svg")
                                    mime_type = "image/svg+xml"
                                
                                # Provide download button
                                st.download_button(
                                    label=f"Download {export_format}",
                                    data=graph_data,
                                    file_name=f"{filename}.{file_ext}",
                                    mime=mime_type
                                )
                            except Exception as e:
                                st.error(f"Error exporting diagram: {str(e)}")
                    
                    # Add a note about the diagram
                    st.info("""
                    **About this diagram:**
                    - Tables are shown with their columns, primary keys (PK), and foreign keys (FK)
                    - Relationships are shown with arrows pointing from foreign keys to primary keys
                    - You can adjust the layout using the controls on the left
                    """)
                    
                except Exception as e:
                    st.error(f"Error generating Graphviz diagram: {str(e)}")
                    st.info("Try adjusting the layout parameters or using a different layout engine.")
            else:
                st.info("No tables found to generate ER diagram.")
                
    def generate_standalone_html(self, schema: Dict[str, Any]) -> str:
        """Generate standalone HTML for the ER diagram that can be saved to a file"""
        if "tables" not in schema or not schema["tables"]:
            return "<html><body><h1>No database schema available</h1></body></html>"
        
        # Create a complete HTML document with the ER diagram
        try:
            # Set default values for visualization
            direction = "LR"
            theme = "Blue"
            show_details = True
            table_spacing = 40
            font_size = 12
            layout_type = "Entity Relationship"
            
            # Set theme colors
            if theme == "Blue":
                header_bg = "#4D7A97"
                header_text = "#FFFFFF"
                row_header_bg = "#E9E9E9"
                row_alt_bg = "#E8F2FE"
                border_color = "#CCCCCC"
                arrow_color = "#4D7A97"
                bg_color = "#FFFFFF"
                relationship_bg = "#F0F8FF"
            else:  # Default to Blue if theme is not recognized
                header_bg = "#4D7A97"
                header_text = "#FFFFFF"
                row_header_bg = "#E9E9E9"
                row_alt_bg = "#E8F2FE"
                border_color = "#CCCCCC"
                arrow_color = "#4D7A97"
                bg_color = "#FFFFFF"
                relationship_bg = "#F0F8FF"
            
            # Start building the complete HTML document
            html = f"""<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Database Schema Diagram</title>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                        background-color: {bg_color};
                    }}
                    .er-container {{ 
                        padding: 20px;
                        max-width: 100%;
                    }}
                    .er-title {{
                        font-size: 24px;
                        font-weight: bold;
                        margin-bottom: 20px;
                        color: {header_bg};
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    }}
                    .er-table {{ 
                        border-collapse: collapse; 
                        box-shadow: 0 2px 3px rgba(0,0,0,0.1);
                        background-color: white;
                        margin: 10px;
                        border-radius: 4px;
                        overflow: hidden;
                    }}
                    .er-table th {{ 
                        background-color: {header_bg}; 
                        color: {header_text}; 
                        padding: 10px; 
                        text-align: center;
                        font-size: {font_size}px;
                    }}
                    .er-table td {{ 
                        padding: 8px; 
                        border: 1px solid {border_color}; 
                        font-size: {font_size}px;
                    }}
                    .er-table tr.header-row th {{ 
                        background-color: {row_header_bg}; 
                        color: #333; 
                        font-size: {font_size}px;
                    }}
                    .er-table tr.pk-row td {{ 
                        background-color: {row_alt_bg}; 
                    }}
                    .er-table tr.fk-row td {{ 
                        background-color: {row_alt_bg}; 
                    }}
                    .pk-badge {{
                        background-color: {header_bg};
                        color: white;
                        border-radius: 3px;
                        padding: 1px 4px;
                        font-size: 10px;
                        margin-left: 5px;
                    }}
                    .fk-badge {{
                        background-color: {arrow_color};
                        color: white;
                        border-radius: 3px;
                        padding: 1px 4px;
                        font-size: 10px;
                        margin-left: 5px;
                        opacity: 0.8;
                    }}
                    .relationship-container {{ 
                        margin-top: 30px;
                        background-color: {relationship_bg};
                        padding: 15px;
                        border-radius: 5px;
                        border: 1px solid {border_color};
                    }}
                    .relationship-title {{
                        font-weight: bold;
                        margin-bottom: 15px;
                        color: {header_bg};
                        font-size: 18px;
                    }}
                    .relationship {{ 
                        padding: 8px; 
                        margin: 8px 0; 
                        border-radius: 4px; 
                        background-color: white; 
                        border: 1px solid {border_color};
                        display: flex;
                        align-items: center;
                    }}
                    .relationship-arrow {{ 
                        color: {arrow_color}; 
                        font-weight: bold;
                        margin: 0 10px;
                        font-size: 18px;
                    }}
                    .table-name {{
                        font-weight: bold;
                        color: {header_bg};
                    }}
                    .column-name {{
                        color: #333;
                    }}
                    .er-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                        gap: {table_spacing}px;
                    }}
                    .er-flex {{
                        display: flex;
                        flex-direction: {direction.lower()};
                        flex-wrap: wrap;
                        gap: {table_spacing}px;
                    }}
                    .er-diagram {{
                        position: relative;
                        min-height: 800px;
                        padding: 30px;
                        margin-bottom: 30px;
                        border: 1px solid {border_color};
                        border-radius: 5px;
                        background-color: white;
                        width: 100%;
                        overflow: auto;
                    }}
                    .table-entity {{
                        position: relative;
                        z-index: 2;
                    }}
                    .svg-container {{
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        z-index: 1;
                    }}
                    svg {{
                        width: 100%;
                        height: 100%;
                    }}
                    .svg-arrow {{
                        stroke: {arrow_color};
                        stroke-width: 1.5;
                        fill: none;
                    }}
                    .svg-arrow-head {{
                        fill: {arrow_color};
                    }}
                    .download-btn {{
                        background-color: {header_bg};
                        color: white;
                        border: none;
                        border-radius: 4px;
                        padding: 8px 15px;
                        cursor: pointer;
                        font-size: 14px;
                        display: inline-flex;
                        align-items: center;
                        gap: 5px;
                        text-decoration: none;
                        margin-right: 10px;
                    }}
                    .download-btn:hover {{
                        opacity: 0.9;
                    }}
                    .toolbar {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 20px;
                        padding: 10px;
                        background-color: #f5f5f5;
                        border-radius: 5px;
                    }}
                    .relationships-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                        gap: 10px;
                    }}
                    @media print {{
                        .toolbar {{
                            display: none;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="er-container">
                    <div class="toolbar">
                        <div class="er-title">Database Schema Diagram</div>
                        <div>
                            <button class="download-btn" onclick="window.print()">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <polyline points="6 9 6 2 18 2 18 9"></polyline>
                                    <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path>
                                    <rect x="6" y="14" width="12" height="8"></rect>
                                </svg>
                                Print Diagram
                            </button>
                            <button class="download-btn" onclick="window.close()">
                                Close
                            </button>
                        </div>
                    </div>
            """
            
            # Create a container for the ER diagram with SVG for arrows
            html += '<div class="er-diagram">'
            
            # Add SVG container for arrows
            html += '<div class="svg-container"><svg id="relationship-arrows"></svg></div>'
            
            # Add tables with absolute positioning
            table_count = len(schema["tables"])
            
            # Calculate positions based on direction
            if direction == "LR":
                # Horizontal layout with better spacing
                tables_per_row = min(3, table_count)  # Limit to 3 tables per row for better visibility
                rows = max(1, (table_count + tables_per_row - 1) // tables_per_row)
                
                # Calculate row heights based on table content
                row_heights = [0] * rows
                table_positions = []
                
                # First pass: determine row heights
                temp_index = 0
                for table_name, table_info in schema["tables"].items():
                    row = temp_index // tables_per_row
                    col = temp_index % tables_per_row
                    
                    # Estimate height based on number of columns (rough estimate)
                    estimated_height = 100 + (len(table_info["columns"]) * 30)
                    row_heights[row] = max(row_heights[row], estimated_height)
                    
                    # Store position info for second pass
                    table_positions.append({
                        "name": table_name,
                        "info": table_info,
                        "row": row,
                        "col": col
                    })
                    
                    temp_index += 1
                
                # Calculate cumulative row positions
                row_positions = [0]
                for i in range(1, len(row_heights)):
                    row_positions.append(row_positions[i-1] + row_heights[i-1] + 50)  # 50px gap between rows
                
                # Second pass: position tables with proper spacing
                for pos in table_positions:
                    table_name = pos["name"]
                    table_info = pos["info"]
                    row = pos["row"]
                    col = pos["col"]
                    
                    # Calculate position with proper spacing
                    left = col * (100 / tables_per_row)
                    top = row_positions[row]
                    
                    # Create table HTML with absolute positioning
                    html += f"""
                    <div class="table-entity" id="{table_name}" style="position: absolute; left: {left}%; top: {top}px; width: {90/tables_per_row}%;">
                        <table class="er-table" style="width: 100%;">
                            <tr>
                                <th colspan="3">{table_name}</th>
                            </tr>
                    """
                    
                    if show_details:
                        html += """
                            <tr class="header-row">
                                <th style="width: 50%;">Column</th>
                                <th style="width: 40%;">Type</th>
                                <th style="width: 10%;">Key</th>
                            </tr>
                        """
                        
                        # Add rows for each column
                        for column in table_info["columns"]:
                            # Check if column is a primary key
                            is_pk = '<span class="pk-badge">PK</span>' if column["name"] in table_info["primary_keys"] else ""
                            
                            # Check if column is a foreign key
                            is_fk = ""
                            for fk in table_info["foreign_keys"]:
                                if column["name"] in fk["constrained_columns"]:
                                    is_fk = '<span class="fk-badge">FK</span>'
                                    break
                            
                            # Determine row class based on PK/FK status
                            row_class = ""
                            if column["name"] in table_info["primary_keys"] and is_fk:
                                row_class = "pk-row fk-row"
                            elif column["name"] in table_info["primary_keys"]:
                                row_class = "pk-row"
                            elif is_fk:
                                row_class = "fk-row"
                            
                            # Add row for this column
                            html += f"""
                            <tr class="{row_class}">
                                <td id="{table_name}_{column['name']}">{column['name']}</td>
                                <td>{column['type']}</td>
                                <td align="center">{is_pk} {is_fk}</td>
                            </tr>
                            """
                    
                    html += """
                        </table>
                    </div>
                    """
            
            # Add JavaScript to draw relationship arrows
            html += """
            <script>
            // Function to draw arrows between tables
            function drawRelationships() {
                const svg = document.getElementById('relationship-arrows');
                if (!svg) return;
                
                // Clear existing arrows
                svg.innerHTML = '';
                
                // Add arrowhead marker definition
                var defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
                var marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
                marker.setAttribute('id', 'arrowhead');
                marker.setAttribute('markerWidth', '10');
                marker.setAttribute('markerHeight', '7');
                marker.setAttribute('refX', '9');
                marker.setAttribute('refY', '3.5');
                marker.setAttribute('orient', 'auto');
                
                var polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                polygon.setAttribute('points', '0 0, 10 3.5, 0 7');
                polygon.setAttribute('class', 'svg-arrow-head');
                
                marker.appendChild(polygon);
                defs.appendChild(marker);
                svg.appendChild(defs);
            """
            
            # Add each relationship arrow
            for table_name, table_info in schema["tables"].items():
                for fk in table_info["foreign_keys"]:
                    source_table = table_name
                    source_col = fk["constrained_columns"][0]
                    target_table = fk["referred_table"]
                    target_col = fk["referred_columns"][0]
                    
                    html += f"""
                    // Draw arrow from {source_table}.{source_col} to {target_table}.{target_col}
                    drawArrow('{source_table}', '{target_table}');
                    """
            
            # Add JavaScript functions to calculate positions and draw arrows
            html += """
                // Function to draw an arrow between two tables
                function drawArrow(sourceTableId, targetTableId) {
                    const sourceElem = document.getElementById(sourceTableId);
                    const targetElem = document.getElementById(targetTableId);
                    
                    if (!sourceElem || !targetElem) return;
                    
                    // Get positions
                    const sourceRect = sourceElem.getBoundingClientRect();
                    const targetRect = targetElem.getBoundingClientRect();
                    const svgRect = svg.getBoundingClientRect();
                    
                    // Calculate center points relative to SVG
                    const sourceX = sourceRect.left + sourceRect.width/2 - svgRect.left;
                    const sourceY = sourceRect.top + sourceRect.height/2 - svgRect.top;
                    const targetX = targetRect.left + targetRect.width/2 - svgRect.left;
                    const targetY = targetRect.top + targetRect.height/2 - svgRect.top;
                    
                    // Calculate start and end points on the edges of the tables
                    let startX, startY, endX, endY;
                    
                    // Determine which edge to use based on relative positions
                    if (Math.abs(sourceX - targetX) > Math.abs(sourceY - targetY)) {
                        // Horizontal connection
                        if (sourceX < targetX) {
                            // Source is to the left of target
                            startX = sourceRect.right - svgRect.left;
                            startY = sourceY;
                            endX = targetRect.left - svgRect.left;
                            endY = targetY;
                        } else {
                            // Source is to the right of target
                            startX = sourceRect.left - svgRect.left;
                            startY = sourceY;
                            endX = targetRect.right - svgRect.left;
                            endY = targetY;
                        }
                    } else {
                        // Vertical connection
                        if (sourceY < targetY) {
                            // Source is above target
                            startX = sourceX;
                            startY = sourceRect.bottom - svgRect.top;
                            endX = targetX;
                            endY = targetRect.top - svgRect.top;
                        } else {
                            // Source is below target
                            startX = sourceX;
                            startY = sourceRect.top - svgRect.top;
                            endX = targetX;
                            endY = targetRect.bottom - svgRect.top;
                        }
                    }
                    
                    // Create path for arrow
                    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                    
                    // Calculate control points for curve
                    const dx = Math.abs(endX - startX) * 0.5;
                    const dy = Math.abs(endY - startY) * 0.5;
                    
                    // Create curved path
                    const pathData = `M ${startX},${startY} C ${startX + dx},${startY} ${endX - dx},${endY} ${endX},${endY}`;
                    
                    path.setAttribute('d', pathData);
                    path.setAttribute('class', 'svg-arrow');
                    path.setAttribute('marker-end', 'url(#arrowhead)');
                    
                    // Add arrow to SVG
                    svg.appendChild(path);
                }
            }
            
            // Run after page loads
            window.addEventListener('load', drawRelationships);
            // Also run after a short delay to ensure elements are positioned
            setTimeout(drawRelationships, 500);
            </script>
            """
            
            # Close the ER diagram container
            html += "</div>"
            
            # Add relationships section
            html += """
            <div class="relationship-container">
                <div class="relationship-title">Database Relationships</div>
                <div class="relationships-grid">
            """
            
            # Add each relationship
            for table_name, table_info in schema["tables"].items():
                for fk in table_info["foreign_keys"]:
                    source_col = fk["constrained_columns"][0]
                    target_table = fk["referred_table"]
                    target_col = fk["referred_columns"][0]
                    
                    html += f"""
                    <div class="relationship">
                        <span><span class="table-name">{table_name}</span>.<span class="column-name">{source_col}</span></span>
                        <span class="relationship-arrow">➔</span>
                        <span><span class="table-name">{target_table}</span>.<span class="column-name">{target_col}</span></span>
                    </div>
                    """
            
            # Close relationships container
            html += """
                </div>
            </div>
            """
            
            # Close main container and HTML document
            html += """
                </div>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            return f"<html><body><h1>Error generating ER diagram</h1><p>{str(e)}</p></body></html>"
    
    def display_full_screen_er_diagram(self, schema: Dict[str, Any]):
        """Wrapper method for full-screen ER diagram that handles page config"""
        # This method is kept for backward compatibility
        # The actual implementation is in display_full_screen_er_diagram_content
        if "tables" not in schema:
            st.info("ER diagram visualization is only available for SQL-based databases.")
            return
        
        # Check if there are tables to visualize
        if not schema["tables"]:
            st.info("No tables found to generate ER diagram.")
            return
            
        self.display_full_screen_er_diagram_content(schema)
    
    def display_full_screen_er_diagram_content(self, schema: Dict[str, Any]):
        """Display a full-screen ER diagram using SVG with explicit connection lines"""
        if "tables" not in schema:
            st.info("ER diagram visualization is only available for SQL-based databases.")
            return
        
        # Check if there are tables to visualize
        if not schema["tables"]:
            st.info("No tables found to generate ER diagram.")
            return
        
        # Set theme colors
        header_bg = "#4D7A97"
        line_color = "#2B5F82"
        badge_pk = "#FF5722"
        badge_fk = "#2196F3"
        
        # Calculate table positions and sizes
        table_width = 220
        table_header_height = 40
        row_height = 30
        
        # Collect table info for positioning
        tables_data = {}
        max_rows = 0
        
        for table_name, table_info in schema["tables"].items():
            num_rows = len(table_info["columns"])
            tables_data[table_name] = {
                "num_rows": num_rows,
                "height": table_header_height + (num_rows * row_height),
                "foreign_keys": table_info["foreign_keys"]
            }
            max_rows = max(max_rows, num_rows)
        
        # Calculate layout
        num_tables = len(tables_data)
        
        # Horizontal layout (tables side by side)
        svg_width = max(1200, (table_width + 50) * min(num_tables, 5))
        svg_height = max(800, table_header_height + (max_rows * row_height) + 400)  # Extra space for arrows
        
        # Position tables in a grid layout
        x_pos, y_pos = 50, 50
        col_count = 0
        for table_name in tables_data:
            if col_count >= 5:  # Start a new row after 5 tables
                col_count = 0
                x_pos = 50
                y_pos += 300  # Move down for next row
            
            tables_data[table_name]["x"] = x_pos
            tables_data[table_name]["y"] = y_pos
            x_pos += table_width + 50
            col_count += 1
        
        # Start SVG
        svg = f"""
        <svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="{line_color}"/>
                </marker>
            </defs>
            <rect width="100%" height="100%" fill="#f8f9fa"/>
        """
        
        # Draw relationship lines first (so they appear behind tables)
        for source_table, table_info in tables_data.items():
            for fk in table_info["foreign_keys"]:
                target_table = fk["referred_table"]
                
                if target_table in tables_data:
                    # Get positions
                    x1 = table_info["x"] + table_width / 2
                    y1 = table_info["y"] + table_info["height"]
                    x2 = tables_data[target_table]["x"] + table_width / 2
                    y2 = tables_data[target_table]["y"]
                    
                    # Adjust for self-referencing tables
                    if source_table == target_table:
                        svg += f"""
                        <path d="M {x1} {y1} 
                                 C {x1} {y1+50}, {x1+100} {y1+50}, {x1+100} {y1} 
                                 C {x1+100} {y1-50}, {x1+50} {y1-50}, {x1} {y1-table_header_height}" 
                              stroke="{line_color}" fill="none" stroke-width="2" marker-end="url(#arrowhead)"/>
                        """
                    else:
                        # Calculate control points for a curved line
                        # For horizontal layout, curve more if tables are far apart
                        distance = abs(x2 - x1)
                        curve_factor = min(100, distance / 3)
                        
                        # Draw curved line with arrow
                        svg += f"""
                        <path d="M {x1} {y1} 
                                 C {x1} {y1+curve_factor}, {x2} {y2-curve_factor}, {x2} {y2}" 
                              stroke="{line_color}" fill="none" stroke-width="2" marker-end="url(#arrowhead)"/>
                        """
                        
                        # Add relationship label
                        mid_x = (x1 + x2) / 2
                        mid_y = (y1 + y2) / 2 + 15
                        
                        source_cols = ", ".join(fk["constrained_columns"])
                        target_cols = ", ".join(fk["referred_columns"])
                        label = f"{source_cols} → {target_cols}"
                        
                        svg += f"""
                        <rect x="{mid_x-5}" y="{mid_y-15}" width="10" height="20" fill="white" opacity="0.8" rx="5" ry="5"/>
                        <text x="{mid_x}" y="{mid_y}" font-family="Arial" font-size="12" text-anchor="middle" fill="{line_color}">{label}</text>
                        """
        
        # Draw tables
        for table_name, table_info in schema["tables"].items():
            table_data = tables_data[table_name]
            x = table_data["x"]
            y = table_data["y"]
            
            # Table header
            svg += f"""
            <rect x="{x}" y="{y}" width="{table_width}" height="{table_header_height}" fill="{header_bg}" rx="5" ry="5"/>
            <text x="{x + table_width/2}" y="{y + table_header_height/2 + 5}" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="white">{table_name}</text>
            """
            
            # Table body background
            body_height = table_data["num_rows"] * row_height
            svg += f"""
            <rect x="{x}" y="{y + table_header_height}" width="{table_width}" height="{body_height}" fill="white" stroke="#ccc" stroke-width="1"/>
            """
            
            # Column headers
            svg += f"""
            <line x1="{x}" y1="{y + table_header_height + row_height}" x2="{x + table_width}" y2="{y + table_header_height + row_height}" stroke="#ddd" stroke-width="1"/>
            <text x="{x + 10}" y="{y + table_header_height + row_height/2 + 5}" font-family="Arial" font-size="12" font-weight="bold" fill="#333">Column</text>
            <text x="{x + 110}" y="{y + table_header_height + row_height/2 + 5}" font-family="Arial" font-size="12" font-weight="bold" fill="#333">Type</text>
            <text x="{x + 180}" y="{y + table_header_height + row_height/2 + 5}" font-family="Arial" font-size="12" font-weight="bold" fill="#333">Key</text>
            """
            
            # Columns
            for i, column in enumerate(table_info["columns"]):
                row_y = y + table_header_height + row_height + (i * row_height)
                
                # Alternating row background
                if i % 2 == 1:
                    svg += f"""
                    <rect x="{x}" y="{row_y}" width="{table_width}" height="{row_height}" fill="#f9f9f9"/>
                    """
                
                # Column name
                svg += f"""
                <text x="{x + 10}" y="{row_y + row_height/2 + 5}" font-family="Arial" font-size="12" fill="#333">{column["name"]}</text>
                """
                
                # Data type
                data_type = column["type"]
                if len(data_type) > 10:
                    data_type = data_type[:8] + "..."
                svg += f"""
                <text x="{x + 110}" y="{row_y + row_height/2 + 5}" font-family="Arial" font-size="12" fill="#666">{data_type}</text>
                """
                
                # Key indicators
                key_x = x + 180
                
                # Primary key
                if column["name"] in table_info["primary_keys"]:
                    svg += f"""
                    <rect x="{key_x}" y="{row_y + row_height/2 - 8}" width="25" height="16" rx="3" ry="3" fill="{badge_pk}"/>
                    <text x="{key_x + 12.5}" y="{row_y + row_height/2 + 5}" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle" fill="white">PK</text>
                    """
                    key_x += 30
                
                # Foreign key
                is_fk = False
                for fk in table_info["foreign_keys"]:
                    if column["name"] in fk["constrained_columns"]:
                        is_fk = True
                        break
                
                if is_fk:
                    svg += f"""
                    <rect x="{key_x}" y="{row_y + row_height/2 - 8}" width="25" height="16" rx="3" ry="3" fill="{badge_fk}"/>
                    <text x="{key_x + 12.5}" y="{row_y + row_height/2 + 5}" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle" fill="white">FK</text>
                    """
                
                # Row separator
                svg += f"""
                <line x1="{x}" y1="{row_y + row_height}" x2="{x + table_width}" y2="{row_y + row_height}" stroke="#ddd" stroke-width="1"/>
                """
        
        # Close SVG
        svg += """
        </svg>
        """
        
        # Create a standalone HTML file with zoom controls
        standalone_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Database Schema Diagram</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    max-width: 100%;
                    overflow: auto;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 20px;
                }}
                h1 {{
                    color: {header_bg};
                    margin-top: 0;
                }}
                .toolbar {{
                    margin-bottom: 20px;
                }}
                .btn {{
                    background-color: {header_bg};
                    color: white;
                    border: none;
                    padding: 10px 15px;
                    border-radius: 4px;
                    cursor: pointer;
                    margin-right: 10px;
                }}
                .btn:hover {{
                    opacity: 0.9;
                }}
                @media print {{
                    .toolbar, .zoom-controls {{
                        display: none;
                    }}
                }}
                
                /* Add zoom controls */
                .zoom-controls {{
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                    padding: 10px;
                    z-index: 1000;
                }}
                .zoom-btn {{
                    background-color: {header_bg};
                    color: white;
                    border: none;
                    width: 30px;
                    height: 30px;
                    border-radius: 4px;
                    cursor: pointer;
                    margin: 0 5px;
                    font-size: 16px;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="toolbar">
                <button class="btn" onclick="window.print()">Print Diagram</button>
                <button class="btn" onclick="window.location.href='/'">Back to Dashboard</button>
            </div>
            
            <div class="zoom-controls">
                <button class="zoom-btn" onclick="zoomIn()">+</button>
                <button class="zoom-btn" onclick="zoomOut()">-</button>
                <button class="zoom-btn" onclick="resetZoom()">↺</button>
            </div>
            
            <h1>Database Schema Diagram</h1>
            
            <div class="container" id="diagram-container">
                {svg}
            </div>
            
            <script>
                // Zoom functionality
                let scale = 1;
                const container = document.getElementById('diagram-container');
                const svg = container.querySelector('svg');
                
                function updateZoom() {{
                    svg.style.transform = `scale(${{scale}})`;
                    svg.style.transformOrigin = 'top left';
                }}
                
                function zoomIn() {{
                    scale += 0.1;
                    updateZoom();
                }}
                
                function zoomOut() {{
                    if (scale > 0.3) {{
                        scale -= 0.1;
                        updateZoom();
                    }}
                }}
                
                function resetZoom() {{
                    scale = 1;
                    updateZoom();
                }}
            </script>
        </body>
        </html>
        """
        
        # Display in Streamlit
        st.components.v1.html(f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; overflow: auto;">
            {svg}
        </div>
        """, height=min(svg_height + 50, 800), scrolling=True)
        
        # Provide a download button for the HTML file
        st.download_button(
            label="Download HTML File",
            data=standalone_html,
            file_name="database_schema.html",
            mime="text/html"
        )
        
        # Create a standalone file for direct opening
        import tempfile
        import os
        
        temp_dir = tempfile.gettempdir()
        html_path = os.path.join(temp_dir, "database_schema.html")
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(standalone_html)
        
        st.markdown(f"""
        <div style="margin-top: 20px; text-align: center;">
            <a href="file://{html_path}" target="_blank" style="
                display: inline-block;
                background-color: #4D7A97;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 4px;
                font-size: 16px;
            ">
                Open in New Tab
            </a>
        </div>
        """, unsafe_allow_html=True)
        
    def display_interactive_er_diagram(self, schema: Dict[str, Any], full_screen=False):
        """Display database schema as an interactive network diagram using PyVis"""
        from pyvis.network import Network
        import tempfile
        import os
        import json
        
        if "tables" not in schema:
            st.info("ER diagram visualization is only available for SQL-based databases.")
            return
        
        # Check if there are tables to visualize
        if not schema["tables"]:
            st.info("No tables found to generate ER diagram.")
            return
        
        # Create settings column
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Diagram Settings")
            
            # Physics settings
            physics_enabled = st.checkbox("Enable Physics", value=True, 
                                         help="Enable physics simulation for automatic layout")
            
            # Layout options
            layout_type = st.selectbox(
                "Layout Algorithm",
                ["hierarchical", "force-directed"],
                index=0,
                help="Algorithm to arrange the tables"
            )
            
            # Direction (for hierarchical layout)
            if layout_type == "hierarchical":
                direction = st.selectbox(
                    "Direction",
                    ["LR", "RL", "UD", "DU"],
                    index=0,
                    help="LR=Left to Right, RL=Right to Left, UD=Up to Down, DU=Down to Up"
                )
            
            # Node spacing
            node_spacing = st.slider(
                "Node Spacing",
                min_value=100,
                max_value=300,
                value=150,
                step=10,
                help="Spacing between nodes"
            )
            
            # Color theme
            theme = st.selectbox(
                "Color Theme",
                ["Blue", "Green", "Purple", "Orange", "Red"],
                index=0,
                help="Color theme for the diagram"
            )
            
            # Show table details
            show_columns = st.checkbox("Show Columns", value=True,
                                     help="Show column details in tables")
            
            # Generate button
            generate_button = st.button("Generate Diagram", use_container_width=True)
        
        with col2:
            if generate_button or full_screen:
                # Set theme colors
                if theme == "Blue":
                    node_color = "#4D7A97"
                    edge_color = "#2B5F82"
                elif theme == "Green":
                    node_color = "#4CAF50"
                    edge_color = "#2E7D32"
                elif theme == "Purple":
                    node_color = "#673AB7"
                    edge_color = "#4527A0"
                elif theme == "Orange":
                    node_color = "#FF9800"
                    edge_color = "#E65100"
                else:  # Red
                    node_color = "#F44336"
                    edge_color = "#B71C1C"
                
                # Create a PyVis network
                height = "800px" if full_screen else "600px"
                net = Network(height=height, width="100%", directed=True, notebook=False)
                
                # Configure physics
                if physics_enabled:
                    physics = {
                        "enabled": True,
                        "solver": "forceAtlas2Based",
                        "forceAtlas2Based": {
                            "gravitationalConstant": -50,
                            "centralGravity": 0.01,
                            "springLength": node_spacing,
                            "springConstant": 0.08,
                            "damping": 0.4,
                            "avoidOverlap": 0.8
                        },
                        "stabilization": {
                            "enabled": True,
                            "iterations": 1000,
                            "updateInterval": 100
                        }
                    }
                else:
                    physics = {"enabled": False}
                
                # Configure hierarchical layout if selected
                if layout_type == "hierarchical":
                    layout = {
                        "hierarchical": {
                            "enabled": True,
                            "direction": direction,
                            "sortMethod": "directed",
                            "nodeSpacing": node_spacing,
                            "treeSpacing": 200,
                            "blockShifting": True,
                            "edgeMinimization": True,
                            "levelSeparation": node_spacing * 1.5
                        }
                    }
                else:
                    layout = {"hierarchical": {"enabled": False}}
                
                # Set options
                options = {
                    "layout": layout,
                    "physics": physics,
                    "interaction": {
                        "navigationButtons": True,
                        "keyboard": True,
                        "hover": True
                    },
                    "edges": {
                        "arrows": {
                            "to": {
                                "enabled": True,
                                "scaleFactor": 1
                            }
                        },
                        "color": edge_color,
                        "smooth": {
                            "enabled": True,
                            "type": "dynamic",
                            "roundness": 0.5
                        },
                        "font": {
                            "size": 12,
                            "align": "middle"
                        }
                    },
                    "nodes": {
                        "shape": "box",
                        "font": {
                            "size": 14,
                            "face": "arial",
                            "align": "center"
                        },
                        "margin": 10
                    }
                }
                
                # Add tables as nodes
                for table_name, table_info in schema["tables"].items():
                    # Create label with HTML formatting
                    label = f"<b>{table_name}</b>"
                    
                    if show_columns:
                        label += "<hr style='margin: 5px 0;'>"
                        
                        # Add primary keys first
                        for column in table_info["columns"]:
                            if column["name"] in table_info["primary_keys"]:
                                label += f"<b style='color: #FF5722;'>🔑 {column['name']}</b> ({column['type']})<br>"
                        
                        # Add foreign keys next
                        fk_columns = set()
                        for fk in table_info["foreign_keys"]:
                            for col in fk["constrained_columns"]:
                                fk_columns.add(col)
                        
                        for column in table_info["columns"]:
                            if column["name"] in fk_columns and column["name"] not in table_info["primary_keys"]:
                                label += f"<b style='color: #2196F3;'>🔗 {column['name']}</b> ({column['type']})<br>"
                        
                        # Add regular columns
                        for column in table_info["columns"]:
                            if column["name"] not in table_info["primary_keys"] and column["name"] not in fk_columns:
                                label += f"{column['name']} ({column['type']})<br>"
                    
                    # Add node
                    net.add_node(table_name, label=label, title=table_name, color=node_color, shape="box")
                
                # Add relationships as edges
                for table_name, table_info in schema["tables"].items():
                    for fk in table_info["foreign_keys"]:
                        source_table = table_name
                        target_table = fk["referred_table"]
                        
                        # Create edge label
                        edge_label = f"{', '.join(fk['constrained_columns'])} → {', '.join(fk['referred_columns'])}"
                        
                        # Add edge
                        net.add_edge(source_table, target_table, label=edge_label, title=edge_label)
                
                # Set options
                net.set_options(json.dumps(options))
                
                # Generate HTML file
                temp_dir = tempfile.gettempdir()
                html_path = os.path.join(temp_dir, "er_diagram.html")
                net.save_graph(html_path)
                
                # Customize the HTML to add download button and improve styling
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                
                # Add custom CSS and buttons
                custom_html = html_content.replace("</head>", """
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                    }
                    .toolbar {
                        position: absolute;
                        top: 10px;
                        right: 10px;
                        z-index: 999;
                        background-color: rgba(255, 255, 255, 0.8);
                        padding: 10px;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
                    }
                    .btn {
                        background-color: """ + node_color + """;
                        color: white;
                        border: none;
                        padding: 8px 15px;
                        margin: 0 5px;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 14px;
                    }
                    .btn:hover {
                        opacity: 0.9;
                    }
                    #mynetwork {
                        width: 100% !important;
                        height: 100vh !important;
                        background-color: #f9f9f9;
                    }
                </style>
                </head>""")
                
                # Add toolbar with buttons
                custom_html = custom_html.replace("<body>", """<body>
                <div class="toolbar">
                    <button class="btn" onclick="window.print()">Print Diagram</button>
                    <button class="btn" onclick="saveAsPNG()">Download as PNG</button>
                    <button class="btn" onclick="window.close()">Close</button>
                </div>
                <script>
                function saveAsPNG() {
                    // Create a canvas from the network
                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    const network = document.getElementById('mynetwork');
                    
                    // Set canvas dimensions
                    canvas.width = network.offsetWidth;
                    canvas.height = network.offsetHeight;
                    
                    // Draw network on canvas
                    html2canvas(network).then(function(canvas) {
                        // Create download link
                        const link = document.createElement('a');
                        link.download = 'database_schema.png';
                        link.href = canvas.toDataURL('image/png');
                        link.click();
                    });
                }
                </script>
                <script src="https://html2canvas.hertzen.com/dist/html2canvas.min.js"></script>
                """)
                
                # Save the customized HTML
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(custom_html)
                
                # Display in Streamlit
                if full_screen:
                    # For full screen, provide a link to open the HTML file directly
                    st.markdown(f"""
                    <div style="text-align: center; margin: 50px 0;">
                        <h1>Interactive ER Diagram</h1>
                        <p>Click the button below to open the interactive diagram in a new tab:</p>
                        <a href="file://{html_path}" target="_blank" style="
                            display: inline-block;
                            background-color: {node_color};
                            color: white;
                            padding: 12px 24px;
                            text-decoration: none;
                            border-radius: 4px;
                            font-size: 16px;
                            margin-top: 20px;
                        ">
                            Open Interactive ER Diagram
                        </a>
                        <p style="margin-top: 20px; color: #666;">
                            The diagram has been saved as an HTML file that you can also share with others.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Also provide a download button for the HTML file
                    with open(html_path, "r", encoding="utf-8") as f:
                        html_data = f.read()
                    
                    st.download_button(
                        label="Download HTML File",
                        data=html_data,
                        file_name="database_schema.html",
                        mime="text/html"
                    )
                else:
                    # For regular view, embed in an iframe
                    st.components.v1.html(f"""
                    <iframe src="file://{html_path}" width="100%" height="600px" frameborder="0"></iframe>
                    <div style="text-align: right; margin-top: 10px;">
                        <a href="?view=full_screen_er_diagram" target="_blank" style="
                            display: inline-block;
                            background-color: {node_color};
                            color: white;
                            padding: 8px 15px;
                            text-decoration: none;
                            border-radius: 4px;
                            font-size: 14px;
                        ">
                            Open in Full Screen
                        </a>
                    </div>
                    """, height=650)
                    
                    # Also provide a download button for the HTML file
                    with open(html_path, "r", encoding="utf-8") as f:
                        html_data = f.read()
                    
                    st.download_button(
                        label="Download HTML File",
                        data=html_data,
                        file_name="database_schema.html",
                        mime="text/html"
                    )
        if "tables" not in schema:
            st.info("ER diagram visualization is only available for SQL-based databases.")
            return
        
        # Check if there are tables to visualize
        if not schema["tables"]:
            st.info("No tables found to generate ER diagram.")
            return
            
        try:
            # Create settings column - different layout for full screen
            if full_screen:
                # For full screen, use a wider layout
                col1, col2 = st.columns([1, 5])
                st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>Database Schema Diagram</h1>", unsafe_allow_html=True)
            else:
                # Normal layout
                col1, col2 = st.columns([1, 3])
            
            with col1:
                st.subheader("Diagram Settings")
                
                # Layout type
                layout_type = st.selectbox(
                    "Layout Type",
                    options=["Table Grid", "Entity Relationship"],
                    index=1,
                    key="html_layout_type",
                    help="Choose between different layout styles"
                )
                
                # Direction options
                direction = st.selectbox(
                    "Direction",
                    options=["LR", "TB"],
                    index=0,
                    key="html_direction",
                    help="LR=Left to Right, TB=Top to Bottom"
                )
                
                # Theme selector
                theme = st.selectbox(
                    "Color Theme",
                    options=["Blue", "Green", "Purple", "Orange", "Light"],
                    index=0,
                    key="html_theme",
                    help="Color theme for the diagram"
                )
                
                # Show table details
                show_details = st.checkbox(
                    "Show Column Details",
                    value=True,
                    key="html_show_details",
                    help="Show column details in tables"
                )
                
                # Table spacing
                table_spacing = st.slider(
                    "Table Spacing",
                    min_value=20,
                    max_value=100,
                    value=40,
                    step=5,
                    key="html_table_spacing",
                    help="Spacing between tables"
                )
                
                # Font size
                font_size = st.slider(
                    "Font Size",
                    min_value=10,
                    max_value=16,
                    value=12,
                    step=1,
                    key="html_font_size",
                    help="Font size for table text"
                )
            
            # Set theme colors
            if theme == "Blue":
                header_bg = "#4D7A97"
                header_text = "#FFFFFF"
                row_header_bg = "#E9E9E9"
                row_alt_bg = "#E8F2FE"
                border_color = "#CCCCCC"
                arrow_color = "#4D7A97"
                bg_color = "#FFFFFF"
                relationship_bg = "#F0F8FF"
            elif theme == "Green":
                header_bg = "#2E8B57"
                header_text = "#FFFFFF"
                row_header_bg = "#E9E9E9"
                row_alt_bg = "#E8F8E9"
                border_color = "#CCCCCC"
                arrow_color = "#2E8B57"
                bg_color = "#FFFFFF"
                relationship_bg = "#F0FFF0"
            elif theme == "Purple":
                header_bg = "#673AB7"
                header_text = "#FFFFFF"
                row_header_bg = "#E9E9E9"
                row_alt_bg = "#F3E5F5"
                border_color = "#CCCCCC"
                arrow_color = "#673AB7"
                bg_color = "#FFFFFF"
                relationship_bg = "#F5F0FF"
            elif theme == "Orange":
                header_bg = "#FF5722"
                header_text = "#FFFFFF"
                row_header_bg = "#E9E9E9"
                row_alt_bg = "#FBE9E7"
                border_color = "#CCCCCC"
                arrow_color = "#FF5722"
                bg_color = "#FFFFFF"
                relationship_bg = "#FFF3E0"
            else:  # Light
                header_bg = "#F5F5F5"
                header_text = "#333333"
                row_header_bg = "#EEEEEE"
                row_alt_bg = "#F9F9F9"
                border_color = "#DDDDDD"
                arrow_color = "#666666"
                bg_color = "#FFFFFF"
                relationship_bg = "#F9F9F9"
            
            with col2:
                # Generate HTML for the ER diagram
                
                # Start building the HTML with better styling
                html = f"""
                <style>
                    body {{ background-color: {bg_color}; }}
                    .er-container {{ 
                        font-family: Arial, sans-serif; 
                        padding: 20px;
                        background-color: {bg_color};
                    }}
                    .er-title {{
                        font-size: 20px;
                        font-weight: bold;
                        margin-bottom: 15px;
                        color: {header_bg};
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    }}
                    .er-table {{ 
                        border-collapse: collapse; 
                        box-shadow: 0 2px 3px rgba(0,0,0,0.1);
                        background-color: white;
                        margin: 10px;
                        border-radius: 4px;
                        overflow: hidden;
                    }}
                    .er-table th {{ 
                        background-color: {header_bg}; 
                        color: {header_text}; 
                        padding: 10px; 
                        text-align: center;
                        font-size: {font_size}px;
                    }}
                    .er-table td {{ 
                        padding: 8px; 
                        border: 1px solid {border_color}; 
                        font-size: {font_size}px;
                    }}
                    .er-table tr.header-row th {{ 
                        background-color: {row_header_bg}; 
                        color: #333; 
                        font-size: {font_size}px;
                    }}
                    .er-table tr.pk-row td {{ 
                        background-color: {row_alt_bg}; 
                    }}
                    .er-table tr.fk-row td {{ 
                        background-color: {row_alt_bg}; 
                    }}
                    .pk-badge {{
                        background-color: {header_bg};
                        color: white;
                        border-radius: 3px;
                        padding: 1px 4px;
                        font-size: 10px;
                        margin-left: 5px;
                    }}
                    .fk-badge {{
                        background-color: {arrow_color};
                        color: white;
                        border-radius: 3px;
                        padding: 1px 4px;
                        font-size: 10px;
                        margin-left: 5px;
                        opacity: 0.8;
                    }}
                    .relationship-container {{ 
                        margin-top: 30px;
                        background-color: {relationship_bg};
                        padding: 15px;
                        border-radius: 5px;
                        border: 1px solid {border_color};
                    }}
                    .relationship-title {{
                        font-weight: bold;
                        margin-bottom: 10px;
                        color: {header_bg};
                        font-size: 16px;
                    }}
                    .relationship {{ 
                        padding: 8px; 
                        margin: 8px 0; 
                        border-radius: 4px; 
                        background-color: white; 
                        border: 1px solid {border_color};
                        display: flex;
                        align-items: center;
                    }}
                    .relationship-arrow {{ 
                        color: {arrow_color}; 
                        font-weight: bold;
                        margin: 0 10px;
                        font-size: 18px;
                    }}
                    .table-name {{
                        font-weight: bold;
                        color: {header_bg};
                    }}
                    .column-name {{
                        color: #333;
                    }}
                    .er-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                        gap: {table_spacing}px;
                    }}
                    .er-flex {{
                        display: flex;
                        flex-direction: {direction.lower()};
                        flex-wrap: wrap;
                        gap: {table_spacing}px;
                    }}
                    .er-diagram {{
                        position: relative;
                        min-height: {800 if full_screen else 700}px;
                        padding: {30 if full_screen else 20}px;
                        margin-bottom: 30px;
                        border: 1px solid {border_color};
                        border-radius: 5px;
                        background-color: white;
                        width: 100%;
                        overflow: auto;
                    }}
                    .table-entity {{
                        position: relative;
                        z-index: 2;
                    }}
                    .svg-container {{
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        z-index: 1;
                    }}
                    svg {{
                        width: 100%;
                        height: 100%;
                    }}
                    .svg-arrow {{
                        stroke: {arrow_color};
                        stroke-width: 1.5;
                        fill: none;
                    }}
                    .svg-arrow-head {{
                        fill: {arrow_color};
                    }}
                    .fullscreen-btn {{
                        background-color: {header_bg};
                        color: white;
                        border: none;
                        border-radius: 4px;
                        padding: 5px 10px;
                        cursor: pointer;
                        font-size: 12px;
                        display: flex;
                        align-items: center;
                        gap: 5px;
                    }}
                    .fullscreen-btn:hover {{
                        opacity: 0.9;
                    }}
                    .modal-overlay {{
                        display: none;
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background-color: rgba(0, 0, 0, 0.7);
                        z-index: 9998;
                        justify-content: center;
                        align-items: center;
                    }}
                    .modal-content {{
                        background-color: white;
                        width: 95%;
                        height: 95%;
                        border-radius: 8px;
                        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
                        overflow: auto;
                        position: relative;
                        padding: 20px;
                        z-index: 9999;
                    }}
                    .modal-header {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding-bottom: 15px;
                        border-bottom: 1px solid #eee;
                        margin-bottom: 20px;
                    }}
                    .modal-title {{
                        font-size: 20px;
                        font-weight: bold;
                        color: {header_bg};
                    }}
                    .modal-close {{
                        background: none;
                        border: none;
                        font-size: 24px;
                        cursor: pointer;
                        color: #666;
                    }}
                    .modal-close:hover {{
                        color: #333;
                    }}
                    .download-options {{
                        display: flex;
                        gap: 10px;
                        margin-top: 15px;
                    }}
                    .download-btn {{
                        background-color: {header_bg};
                        color: white;
                        border: none;
                        border-radius: 4px;
                        padding: 8px 15px;
                        cursor: pointer;
                        font-size: 14px;
                        display: flex;
                        align-items: center;
                        gap: 5px;
                    }}
                    .download-btn:hover {{
                        opacity: 0.9;
                    }}
                </style>
                
                <div class="er-container" id="er-container">
                    <div class="er-title">
                        <span>Database Schema Visualization</span>
                        <button onclick="openFullScreen()" class="fullscreen-btn">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
                            </svg>
                            Open in Full Screen
                        </button>
                    </div>
                    
                    <script>
                    function openFullScreen() {{
                        // Navigate to the full screen view
                        window.location.href = "?view=full_screen_er_diagram";
                    }}
                    </script>
                """
                
                # Choose layout based on selection
                if layout_type == "Table Grid":
                    # Grid layout for tables
                    html += f'<div class="er-grid">'
                    
                    # Add tables
                    for table_name, table_info in schema["tables"].items():
                        # Create table HTML
                        html += f"""
                        <div class="table-entity" id="{table_name}">
                            <table class="er-table" style="width: 100%;">
                                <tr>
                                    <th colspan="4">{table_name}</th>
                                </tr>
                        """
                        
                        if show_details:
                            html += """
                                <tr class="header-row">
                                    <th>Column</th>
                                    <th>Type</th>
                                    <th>PK</th>
                                    <th>FK</th>
                                </tr>
                            """
                            
                            # Add rows for each column
                            for column in table_info["columns"]:
                                # Check if column is a primary key
                                is_pk = "✓" if column["name"] in table_info["primary_keys"] else ""
                                
                                # Check if column is a foreign key
                                is_fk = ""
                                for fk in table_info["foreign_keys"]:
                                    if column["name"] in fk["constrained_columns"]:
                                        is_fk = "✓"
                                        break
                                
                                # Determine row class based on PK/FK status
                                row_class = ""
                                if is_pk and is_fk:
                                    row_class = "pk-row fk-row"
                                elif is_pk:
                                    row_class = "pk-row"
                                elif is_fk:
                                    row_class = "fk-row"
                                
                                # Add row for this column
                                html += f"""
                                <tr class="{row_class}">
                                    <td id="{table_name}_{column['name']}">{column['name']}</td>
                                    <td>{column['type']}</td>
                                    <td align="center">{is_pk}</td>
                                    <td align="center">{is_fk}</td>
                                </tr>
                                """
                        
                        html += """
                            </table>
                        </div>
                        """
                    
                    # Close the grid container
                    html += "</div>"
                    
                else:  # Entity Relationship layout
                    # Create a container for the ER diagram with SVG for arrows
                    html += '<div class="er-diagram">'
                    
                    # Add SVG container for arrows
                    html += '<div class="svg-container"><svg id="relationship-arrows"></svg></div>'
                    
                    # Add tables with absolute positioning
                    table_count = len(schema["tables"])
                    table_index = 0
                    
                    # Calculate positions based on direction
                    if direction == "LR":
                        # Horizontal layout with better spacing
                        tables_per_row = min(3, table_count)  # Limit to 3 tables per row for better visibility
                        rows = max(1, (table_count + tables_per_row - 1) // tables_per_row)
                        
                        # Calculate row heights based on table content
                        row_heights = [0] * rows
                        table_positions = []
                        
                        # First pass: determine row heights
                        temp_index = 0
                        for table_name, table_info in schema["tables"].items():
                            row = temp_index // tables_per_row
                            col = temp_index % tables_per_row
                            
                            # Estimate height based on number of columns (rough estimate)
                            estimated_height = 100 + (len(table_info["columns"]) * 30)
                            row_heights[row] = max(row_heights[row], estimated_height)
                            
                            # Store position info for second pass
                            table_positions.append({
                                "name": table_name,
                                "info": table_info,
                                "row": row,
                                "col": col
                            })
                            
                            temp_index += 1
                        
                        # Calculate cumulative row positions
                        row_positions = [0]
                        for i in range(1, len(row_heights)):
                            row_positions.append(row_positions[i-1] + row_heights[i-1] + 50)  # 50px gap between rows
                        
                        # Second pass: position tables with proper spacing
                        for pos in table_positions:
                            table_name = pos["name"]
                            table_info = pos["info"]
                            row = pos["row"]
                            col = pos["col"]
                            
                            # Calculate position with proper spacing
                            left = col * (100 / tables_per_row)
                            top = row_positions[row]
                            
                            # Create table HTML with absolute positioning
                            html += f"""
                            <div class="table-entity" id="{table_name}" style="position: absolute; left: {left}%; top: {top}px; width: {90/tables_per_row}%;">
                                <table class="er-table" style="width: 100%;">
                                    <tr>
                                        <th colspan="3">{table_name}</th>
                                    </tr>
                            """
                            
                            if show_details:
                                html += """
                                    <tr class="header-row">
                                        <th style="width: 50%;">Column</th>
                                        <th style="width: 40%;">Type</th>
                                        <th style="width: 10%;">Key</th>
                                    </tr>
                                """
                                
                                # Add rows for each column
                                for column in table_info["columns"]:
                                    # Check if column is a primary key
                                    is_pk = '<span class="pk-badge">PK</span>' if column["name"] in table_info["primary_keys"] else ""
                                    
                                    # Check if column is a foreign key
                                    is_fk = ""
                                    for fk in table_info["foreign_keys"]:
                                        if column["name"] in fk["constrained_columns"]:
                                            is_fk = '<span class="fk-badge">FK</span>'
                                            break
                                    
                                    # Determine row class based on PK/FK status
                                    row_class = ""
                                    if column["name"] in table_info["primary_keys"] and is_fk:
                                        row_class = "pk-row fk-row"
                                    elif column["name"] in table_info["primary_keys"]:
                                        row_class = "pk-row"
                                    elif is_fk:
                                        row_class = "fk-row"
                                    
                                    # Add row for this column
                                    html += f"""
                                    <tr class="{row_class}">
                                        <td id="{table_name}_{column['name']}">{column['name']}</td>
                                        <td>{column['type']}</td>
                                        <td align="center">{is_pk} {is_fk}</td>
                                    </tr>
                                    """
                            
                            html += """
                                </table>
                            </div>
                            """
                            
                            table_index += 1
                    
                    else:  # TB layout
                        # Vertical layout with better spacing
                        table_heights = []
                        table_positions = []
                        
                        # First pass: determine table heights
                        for table_name, table_info in schema["tables"].items():
                            # Estimate height based on number of columns
                            estimated_height = 100 + (len(table_info["columns"]) * 30)
                            table_heights.append(estimated_height)
                            
                            # Store position info for second pass
                            table_positions.append({
                                "name": table_name,
                                "info": table_info
                            })
                        
                        # Second pass: position tables with proper spacing
                        current_top = 20  # Start with some padding
                        for i, pos in enumerate(table_positions):
                            table_name = pos["name"]
                            table_info = pos["info"]
                            
                            # Create table HTML with absolute positioning
                            html += f"""
                            <div class="table-entity" id="{table_name}" style="position: absolute; left: 10%; top: {current_top}px; width: 80%;">
                                <table class="er-table" style="width: 100%;">
                                    <tr>
                                        <th colspan="3">{table_name}</th>
                                    </tr>
                            """
                            
                            # Update top position for next table
                            current_top += table_heights[i] + 50  # 50px gap between tables
                            
                            if show_details:
                                html += """
                                    <tr class="header-row">
                                        <th style="width: 50%;">Column</th>
                                        <th style="width: 40%;">Type</th>
                                        <th style="width: 10%;">Key</th>
                                    </tr>
                                """
                                
                                # Add rows for each column
                                for column in table_info["columns"]:
                                    # Check if column is a primary key
                                    is_pk = '<span class="pk-badge">PK</span>' if column["name"] in table_info["primary_keys"] else ""
                                    
                                    # Check if column is a foreign key
                                    is_fk = ""
                                    for fk in table_info["foreign_keys"]:
                                        if column["name"] in fk["constrained_columns"]:
                                            is_fk = '<span class="fk-badge">FK</span>'
                                            break
                                    
                                    # Determine row class based on PK/FK status
                                    row_class = ""
                                    if column["name"] in table_info["primary_keys"] and is_fk:
                                        row_class = "pk-row fk-row"
                                    elif column["name"] in table_info["primary_keys"]:
                                        row_class = "pk-row"
                                    elif is_fk:
                                        row_class = "fk-row"
                                    
                                    # Add row for this column
                                    html += f"""
                                    <tr class="{row_class}">
                                        <td id="{table_name}_{column['name']}">{column['name']}</td>
                                        <td>{column['type']}</td>
                                        <td align="center">{is_pk} {is_fk}</td>
                                    </tr>
                                    """
                            
                            html += """
                                </table>
                            </div>
                            """
                            
                            table_index += 1
                    
                    # Add JavaScript to draw relationship arrows
                    html += """
                    <script>
                    // Function to draw arrows between tables
                    function drawRelationships() {
                        const svg = document.getElementById('relationship-arrows');
                        if (!svg) return;
                        
                        // Clear existing arrows
                        svg.innerHTML = '';
                    """
                    
                    # Add each relationship arrow
                    for table_name, table_info in schema["tables"].items():
                        for fk in table_info["foreign_keys"]:
                            source_table = table_name
                            source_col = fk["constrained_columns"][0]
                            target_table = fk["referred_table"]
                            target_col = fk["referred_columns"][0]
                            
                            html += f"""
                            // Draw arrow from {source_table}.{source_col} to {target_table}.{target_col}
                            drawArrow('{source_table}', '{target_table}', '{source_col}', '{target_col}');
                            """
                    
                    # Add JavaScript functions to calculate positions and draw arrows
                    html += """
                        // Function to draw an arrow between two tables
                        function drawArrow(sourceTable, targetTable, sourceCol, targetCol) {
                            const sourceElem = document.getElementById(sourceTable);
                            const targetElem = document.getElementById(targetTable);
                            
                            if (!sourceElem || !targetElem) return;
                            
                            // Get positions
                            const sourceRect = sourceElem.getBoundingClientRect();
                            const targetRect = targetElem.getBoundingClientRect();
                            const svgRect = svg.getBoundingClientRect();
                            
                            // Calculate center points relative to SVG
                            const sourceX = sourceRect.left + sourceRect.width/2 - svgRect.left;
                            const sourceY = sourceRect.top + sourceRect.height/2 - svgRect.top;
                            const targetX = targetRect.left + targetRect.width/2 - svgRect.left;
                            const targetY = targetRect.top + targetRect.height/2 - svgRect.top;
                            
                            // Calculate start and end points on the edges of the tables
                            let startX, startY, endX, endY;
                            
                            // Determine which edge to use based on relative positions
                            if (Math.abs(sourceX - targetX) > Math.abs(sourceY - targetY)) {
                                // Horizontal connection
                                if (sourceX < targetX) {
                                    // Source is to the left of target
                                    startX = sourceRect.right - svgRect.left;
                                    startY = sourceY;
                                    endX = targetRect.left - svgRect.left;
                                    endY = targetY;
                                } else {
                                    // Source is to the right of target
                                    startX = sourceRect.left - svgRect.left;
                                    startY = sourceY;
                                    endX = targetRect.right - svgRect.left;
                                    endY = targetY;
                                }
                            } else {
                                // Vertical connection
                                if (sourceY < targetY) {
                                    // Source is above target
                                    startX = sourceX;
                                    startY = sourceRect.bottom - svgRect.top;
                                    endX = targetX;
                                    endY = targetRect.top - svgRect.top;
                                } else {
                                    // Source is below target
                                    startX = sourceX;
                                    startY = sourceRect.top - svgRect.top;
                                    endX = targetX;
                                    endY = targetRect.bottom - svgRect.top;
                                }
                            }
                            
                            // Create path for arrow
                            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                            
                            // Calculate control points for curve
                            const dx = Math.abs(endX - startX) * 0.5;
                            const dy = Math.abs(endY - startY) * 0.5;
                            
                            // Create curved path
                            const pathData = `M ${startX},${startY} C ${startX + dx},${startY} ${endX - dx},${endY} ${endX},${endY}`;
                            
                            path.setAttribute('d', pathData);
                            path.setAttribute('class', 'svg-arrow');
                            path.setAttribute('marker-end', 'url(#arrowhead)');
                            
                            // Add arrow to SVG
                            svg.appendChild(path);
                            
                            // Add arrowhead marker if it doesn't exist
                            if (!document.getElementById('arrowhead')) {
                                const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
                                const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
                                marker.setAttribute('id', 'arrowhead');
                                marker.setAttribute('markerWidth', '10');
                                marker.setAttribute('markerHeight', '7');
                                marker.setAttribute('refX', '9');
                                marker.setAttribute('refY', '3.5');
                                marker.setAttribute('orient', 'auto');
                                
                                const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                                polygon.setAttribute('points', '0 0, 10 3.5, 0 7');
                                polygon.setAttribute('class', 'svg-arrow-head');
                                
                                marker.appendChild(polygon);
                                defs.appendChild(marker);
                                svg.appendChild(defs);
                            }
                        }
                    }
                    
                    // Run after page loads
                    window.addEventListener('load', drawRelationships);
                    // Also run after a short delay to ensure elements are positioned
                    setTimeout(drawRelationships, 500);
                    </script>
                    """
                    
                    # Close the ER diagram container
                    html += "</div>"
                
                # Close main container
                html += """
                </div>
                """
                
                # Display the HTML - adjust height for full screen
                if full_screen:
                    st.components.v1.html(html, height=1500, scrolling=True)
                else:
                    st.components.v1.html(html, height=1000, scrolling=True)
                
                # Add a note about the diagram
                st.info("""
                **About this ER diagram:**
                - Tables are shown with their columns, primary keys (PK), and foreign keys (FK)
                - Relationships are shown with arrows between tables
                - You can adjust the appearance using the controls on the left
                - Use the full-screen button in the top-right corner to view the diagram in full screen
                """)
                
                # Create HTML for relationships section
                relationships_html = f"""
                <style>
                    body {{ background-color: {bg_color}; }}
                    .relationship-container {{ 
                        font-family: Arial, sans-serif;
                        margin-top: 20px;
                        background-color: {relationship_bg};
                        padding: 15px;
                        border-radius: 5px;
                        border: 1px solid {border_color};
                    }}
                    .relationship-title {{
                        font-weight: bold;
                        margin-bottom: 15px;
                        color: {header_bg};
                        font-size: 18px;
                    }}
                    .relationship {{ 
                        padding: 8px; 
                        margin: 8px 0; 
                        border-radius: 4px; 
                        background-color: white; 
                        border: 1px solid {border_color};
                        display: flex;
                        align-items: center;
                    }}
                    .relationship-arrow {{ 
                        color: {arrow_color}; 
                        font-weight: bold;
                        margin: 0 10px;
                        font-size: 18px;
                    }}
                    .table-name {{
                        font-weight: bold;
                        color: {header_bg};
                    }}
                    .column-name {{
                        color: #333;
                    }}
                    .relationships-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                        gap: 10px;
                    }}
                </style>
                
                <div class="relationship-container">
                    <div class="relationship-title">Database Relationships</div>
                    <div class="relationships-grid">
                """
                
                # Add each relationship
                for table_name, table_info in schema["tables"].items():
                    for fk in table_info["foreign_keys"]:
                        source_col = fk["constrained_columns"][0]
                        target_table = fk["referred_table"]
                        target_col = fk["referred_columns"][0]
                        
                        relationships_html += f"""
                        <div class="relationship">
                            <span><span class="table-name">{table_name}</span>.<span class="column-name">{source_col}</span></span>
                            <span class="relationship-arrow">➔</span>
                            <span><span class="table-name">{target_table}</span>.<span class="column-name">{target_col}</span></span>
                        </div>
                        """
                
                # Close relationships container
                relationships_html += """
                    </div>
                </div>
                """
                
                # Display relationships section - adjust height for full screen
                if full_screen:
                    st.components.v1.html(relationships_html, height=500, scrolling=True)
                else:
                    st.components.v1.html(relationships_html, height=300, scrolling=True)
                
        except Exception as e:
            st.error(f"Error generating ER diagram: {str(e)}")
            st.info("Try adjusting the settings or check the database schema.")
