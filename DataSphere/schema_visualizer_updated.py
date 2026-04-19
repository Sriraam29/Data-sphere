import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import graphviz
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
        else:
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
        """Display database schema as an ER diagram"""
        if "tables" not in schema:
            st.info("ER diagram visualization is only available for SQL-based databases.")
            return
        
        # Create graph for ER diagram
        G = nx.DiGraph()
        
        # Add nodes (tables)
        for table_name in schema["tables"].keys():
            G.add_node(table_name)
        
        # Add edges (relationships)
        for table_name, table_info in schema["tables"].items():
            for fk in table_info["foreign_keys"]:
                G.add_edge(
                    table_name, 
                    fk["referred_table"],
                    label=f"{', '.join(fk['constrained_columns'])} -> {', '.join(fk['referred_columns'])}"
                )
        
        # Generate the ER diagram
        if len(G.nodes) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="lightblue", ax=ax)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=1.5, arrowsize=20, ax=ax)
            
            # Draw node labels
            nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
            
            # Draw edge labels
            edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
            
            plt.title("Entity Relationship Diagram")
            plt.axis("off")
            
            # Display the ER diagram in Streamlit
            st.pyplot(fig)
            
            # Add download button for the diagram
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            
            # Provide download button
            btn = st.download_button(
                label="Download ER Diagram",
                data=buf,
                file_name="er_diagram.png",
                mime="image/png"
            )
        else:
            st.info("No tables found to generate ER diagram.")
            
    def display_graphviz_er_diagram(self, schema: Dict[str, Any]):
        """Display database schema as an ER diagram using Graphviz"""
        if "tables" not in schema:
            st.info("ER diagram visualization is only available for SQL-based databases.")
            return
        
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
            
            # Direction options
            direction = st.selectbox(
                "Direction",
                options=["LR", "TB", "RL", "BT"],
                index=0,
                key="graphviz_direction",
                help="LR=Left to Right, TB=Top to Bottom, RL=Right to Left, BT=Bottom to Top"
            )
            
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
            
            # Edge style
            edge_style = st.selectbox(
                "Edge Style",
                options=["polyline", "spline", "ortho", "curved"],
                index=0,
                key="graphviz_edge_style",
                help="Style of the relationship lines"
            )
            
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
            
            # Get theme colors
            theme_colors = {
                "Blue": {
                    "header": "#4D7A97",
                    "highlight": "#E8F2FE",
                    "edge": "#4D7A97"
                },
                "Green": {
                    "header": "#5B9C64",
                    "highlight": "#E8F5E9",
                    "edge": "#5B9C64"
                },
                "Purple": {
                    "header": "#7E57C2",
                    "highlight": "#F3E5F5",
                    "edge": "#7E57C2"
                },
                "Orange": {
                    "header": "#F57C00",
                    "highlight": "#FFF3E0",
                    "edge": "#F57C00"
                },
                "Monochrome": {
                    "header": "#424242",
                    "highlight": "#F5F5F5",
                    "edge": "#424242"
                }
            }
            
            # Get selected theme colors
            selected_theme = theme_colors[theme]
        
        # Create a new Graphviz graph
        if show_details:
            # Detailed view with table columns
            graph = graphviz.Digraph(
                name="ER Diagram", 
                engine=layout_engine,
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
                    'dir': 'both',
                    'color': selected_theme["edge"],
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
            
            # Add nodes (tables) with HTML-like labels to show table structure
            for table_name, table_info in schema["tables"].items():
                # Create HTML-like label for table
                label = f'''<
                <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR>
                        <TD COLSPAN="4" BGCOLOR="{selected_theme["header"]}" COLOR="white"><B>{table_name}</B></TD>
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
                    bg_color = selected_theme["highlight"] if is_pk or is_fk else "white"
                    
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
                        color=selected_theme["edge"],
                        penwidth="1.5"
                    )
        else:
            # Simplified view with just table names
            graph = graphviz.Digraph(
                name="ER Diagram", 
                engine=layout_engine,
                node_attr={
                    'shape': 'box',
                    'style': 'filled',
                    'fillcolor': selected_theme["header"],
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
                    'color': selected_theme["edge"],
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
                        label=edge_label,
                        color=selected_theme["edge"]
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