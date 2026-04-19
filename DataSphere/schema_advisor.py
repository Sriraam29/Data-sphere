import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io
import re
from typing import Dict, List, Any, Tuple, Optional

class SchemaAdvisor:
    def __init__(self):
        """Initialize the Schema Advisor"""
        pass
    
    def schema_advisor_ui(self, db_manager):
        """UI for AI-powered schema recommendations"""
        st.subheader("Schema Advisor")
        
        # Check if we have schema information
        if not st.session_state.get("db_schema"):
            st.warning("No database schema information available. Connect to a database first.")
            return
        
        tab1, tab2, tab3 = st.tabs(["Schema Analysis", "Normalization Advisor", "Natural Language Schema Design"])
        
        with tab1:
            self._schema_analysis_ui(db_manager)
        
        with tab2:
            self._normalization_advisor_ui(db_manager)
        
        with tab3:
            self._nl_schema_designer_ui()
    
    def _schema_analysis_ui(self, db_manager):
        """Analyze database schema and provide recommendations"""
        st.subheader("Schema Analysis")
        
        if st.button("Analyze Schema"):
            with st.spinner("Analyzing database schema..."):
                schema = st.session_state.get("db_schema", {})
                
                if not schema:
                    st.warning("No schema information available")
                    return
                
                # Calculate schema metrics
                metrics = self._calculate_schema_metrics(schema)
                
                # Display schema metrics
                st.subheader("Schema Metrics")
                metrics_df = pd.DataFrame([metrics])
                st.dataframe(metrics_df)
                
                # Generate schema health score
                health_score, health_factors = self._calculate_schema_health(schema)
                
                # Display schema health
                st.subheader("Schema Health Score")
                st.progress(health_score / 100)
                st.write(f"Overall Health: {health_score}/100")
                
                # Display health factors
                for category, factors in health_factors.items():
                    with st.expander(f"{category} Factors"):
                        for factor in factors:
                            if factor["type"] == "success":
                                st.success(factor["message"])
                            elif factor["type"] == "warning":
                                st.warning(factor["message"])
                            elif factor["type"] == "error":
                                st.error(factor["message"])
                
                # Generate schema improvement recommendations
                recommendations = self._generate_schema_recommendations(schema)
                
                # Display recommendations
                st.subheader("Schema Improvement Recommendations")
                for category, recs in recommendations.items():
                    with st.expander(category):
                        for rec in recs:
                            st.info(rec["title"])
                            st.write(rec["description"])
                            if "sql" in rec:
                                st.code(rec["sql"], language="sql")
    
    def _normalization_advisor_ui(self, db_manager):
        """Analyze database for normalization issues and provide recommendations"""
        st.subheader("Normalization Advisor")
        
        if st.button("Check Normalization"):
            with st.spinner("Analyzing database normalization..."):
                schema = st.session_state.get("db_schema", {})
                
                if not schema:
                    st.warning("No schema information available")
                    return
                
                # Check for normalization issues
                normalization_issues = self._check_normalization(schema)
                
                # Display normalization status
                if not any(issues for category, issues in normalization_issues.items()):
                    st.success("Your database schema appears to be well normalized!")
                else:
                    st.warning("Potential normalization issues detected")
                
                # Display normalization issues by form
                for form, issues in normalization_issues.items():
                    if issues:
                        with st.expander(f"{form} Issues"):
                            for issue in issues:
                                st.warning(issue["description"])
                                if "suggestion" in issue:
                                    st.info(f"Suggestion: {issue['suggestion']}")
                                if "sql" in issue:
                                    st.code(issue["sql"], language="sql")
    
    def _nl_schema_designer_ui(self):
        """Generate database schema from natural language description"""
        st.subheader("Natural Language Schema Designer")
        
        # Input for business requirements
        business_req = st.text_area(
            "Describe your data model in natural language",
            height=150,
            help="Describe the entities, their attributes, and relationships in plain English",
            placeholder="Example: A User has many Orders. An Order belongs to a User and contains many Products. A Product has a Category."
        )
        
        # Select database type
        db_type = st.selectbox(
            "Target Database Type",
            options=["PostgreSQL", "MySQL", "SQLite", "MongoDB"],
            index=0
        )
        
        # Generate schema button
        if st.button("Generate Schema") and business_req:
            with st.spinner("Generating schema from description..."):
                # This would use NLP/LLM to convert description to schema
                # Here we'll use a simplified example generator
                generated_schema = self._generate_schema_from_nl(business_req, db_type.lower())
                
                if generated_schema:
                    # Display the generated schema
                    st.success(f"✓ Successfully generated schema with {len(generated_schema)} table(s)")
                    
                    # Display visually using Graphviz for better visualization
                    if db_type.lower() != "mongodb":
                        # Convert generated schema to full schema format for visualization
                        full_schema = self._convert_to_full_schema_format(generated_schema)
                        
                        # Visualization type selection
                        viz_type = st.radio(
                            "Visualization Type",
                            ["Graphviz ER Diagram", "Simple Network Diagram"],
                            horizontal=True
                        )
                        
                        if viz_type == "Graphviz ER Diagram":
                            self._display_graphviz_diagram(full_schema)
                        else:
                            # Create simple network ER diagram for SQL databases
                            fig, ax = plt.subplots(figsize=(12, 8))
                            G = nx.DiGraph()
                            
                            # Add nodes for each table
                            for table, details in generated_schema.items():
                                table_label = f"{table}\n"
                                for col in details["columns"]:
                                    pk_marker = "🔑 " if col in details.get("primary_keys", []) else ""
                                    # Check if FK
                                    is_fk = any(rel.get("fk_column") == col for rel in details.get("relations", []) if rel["type"] == "many-to-one")
                                    fk_marker = "🔗 " if is_fk else ""
                                    table_label += f"{pk_marker}{fk_marker}{col}\n"
                                G.add_node(table, label=table_label)
                            
                            # Add edges for relationships
                            for table, details in generated_schema.items():
                                for relation in details.get("relations", []):
                                    if relation["type"] == "many-to-one":
                                        G.add_edge(table, relation["table"], 
                                                 label=f"{relation.get('fk_column', '')}")
                            
                            # Draw the graph
                            pos = nx.spring_layout(G, k=2, iterations=50)
                            nx.draw(G, pos, with_labels=True, node_color="lightblue", 
                                    node_size=4000, arrows=True, font_size=8, 
                                    arrowsize=20, edge_color='gray', arrowstyle='->')
                            
                            # Draw edge labels
                            edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True) if d.get("label")}
                            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
                            
                            plt.title("Entity Relationship Diagram", fontsize=14, fontweight='bold')
                            plt.axis('off')
                            
                            # Display the figure
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    # Display SQL schema creation script
                    if db_type.lower() != "mongodb":
                        create_statements = self._generate_sql_from_schema(generated_schema, db_type.lower())
                        st.subheader("SQL Schema Creation Script")
                        st.code(create_statements, language="sql")
                        
                        # Add download button
                        st.download_button(
                            label="📥 Download SQL Script",
                            data=create_statements,
                            file_name=f"schema_{db_type.lower()}.sql",
                            mime="text/plain"
                        )
                    else:
                        # For MongoDB, show JSON schema
                        st.subheader("MongoDB Collection Schema")
                        for collection, schema in generated_schema.items():
                            with st.expander(collection):
                                st.json(schema)
    
    def _calculate_schema_metrics(self, schema: Dict) -> Dict[str, Any]:
        """Calculate various metrics about the database schema"""
        metrics = {}
        
        if "tables" in schema:  # SQL database
            tables = schema["tables"]
            metrics["Table Count"] = len(tables)
            
            column_counts = [len(table_info["columns"]) for table_info in tables.values()]
            metrics["Total Columns"] = sum(column_counts)
            metrics["Avg Columns Per Table"] = round(sum(column_counts) / len(tables) if tables else 0, 2)
            
            pk_count = sum(len(table_info["primary_keys"]) for table_info in tables.values())
            metrics["Primary Keys"] = pk_count
            
            fk_count = sum(len(table_info["foreign_keys"]) for table_info in tables.values())
            metrics["Foreign Keys"] = fk_count
            
            metrics["Relationship Ratio"] = round(fk_count / len(tables) if tables else 0, 2)
            
            # Count indexes
            index_count = sum(len(table_info.get("indexes", [])) for table_info in tables.values())
            metrics["Indexes"] = index_count
            
        elif "collections" in schema:  # MongoDB
            collections = schema["collections"]
            metrics["Collection Count"] = len(collections)
            
            field_counts = [len(collection_info["fields"]) for collection_info in collections.values()]
            metrics["Total Fields"] = sum(field_counts)
            metrics["Avg Fields Per Collection"] = round(sum(field_counts) / len(collections) if collections else 0, 2)
            
            # Count embedded documents and arrays
            embedded_docs = sum(
                sum(1 for field in collection_info["fields"] if field["type"] == "object") 
                for collection_info in collections.values()
            )
            metrics["Embedded Documents"] = embedded_docs
            
            arrays = sum(
                sum(1 for field in collection_info["fields"] if field["type"] == "array") 
                for collection_info in collections.values()
            )
            metrics["Array Fields"] = arrays
            
            metrics["Nesting Ratio"] = round((embedded_docs + arrays) / sum(field_counts) if sum(field_counts) else 0, 2)
            
            # Count indexes
            index_count = sum(len(collection_info.get("indexes", [])) for collection_info in collections.values())
            metrics["Indexes"] = index_count
        
        return metrics
    
    def _calculate_schema_health(self, schema: Dict) -> Tuple[int, Dict[str, List[Dict[str, str]]]]:
        """Calculate schema health score and factors"""
        health_score = 100  # Start with perfect score
        health_factors = {
            "Good Practices": [],
            "Warnings": [],
            "Critical Issues": []
        }
        
        if "tables" in schema:  # SQL database
            tables = schema["tables"]
            
            # Check primary keys
            tables_without_pk = [name for name, info in tables.items() if not info["primary_keys"]]
            if not tables_without_pk:
                health_factors["Good Practices"].append({
                    "type": "success",
                    "message": "All tables have primary keys"
                })
            else:
                health_score -= 10 * len(tables_without_pk)
                health_factors["Critical Issues"].append({
                    "type": "error",
                    "message": f"Tables without primary keys: {', '.join(tables_without_pk)}"
                })
            
            # Check for very wide tables (many columns)
            wide_tables = [name for name, info in tables.items() if len(info["columns"]) > 20]
            if wide_tables:
                health_score -= 5 * len(wide_tables)
                health_factors["Warnings"].append({
                    "type": "warning",
                    "message": f"Tables with many columns (>20): {', '.join(wide_tables)}"
                })
            
            # Check foreign key coverage
            fk_counts = {name: len(info["foreign_keys"]) for name, info in tables.items()}
            if sum(fk_counts.values()) > 0:
                health_factors["Good Practices"].append({
                    "type": "success",
                    "message": f"Database uses foreign keys to maintain referential integrity"
                })
            else:
                health_score -= 15
                health_factors["Warnings"].append({
                    "type": "warning",
                    "message": "No foreign keys found - consider adding them to maintain referential integrity"
                })
            
            # Check for naming conventions
            snake_case_tables = all(re.match(r'^[a-z][a-z0-9_]*$', name) for name in tables.keys())
            if snake_case_tables:
                health_factors["Good Practices"].append({
                    "type": "success",
                    "message": "Table names follow snake_case convention"
                })
            else:
                health_score -= 5
                health_factors["Warnings"].append({
                    "type": "warning",
                    "message": "Some table names don't follow snake_case convention"
                })
            
        elif "collections" in schema:  # MongoDB
            collections = schema["collections"]
            
            # Check for very nested documents
            nested_collections = []
            for name, info in collections.items():
                max_nesting = 0
                for field in info["fields"]:
                    if field["type"] == "object" or field["type"] == "array":
                        # In a real implementation, we'd check actual nesting depth
                        max_nesting += 1
                
                if max_nesting > 3:
                    nested_collections.append(name)
            
            if nested_collections:
                health_score -= 5 * len(nested_collections)
                health_factors["Warnings"].append({
                    "type": "warning",
                    "message": f"Collections with deep nesting: {', '.join(nested_collections)}"
                })
            
            # Check for indexing
            collections_with_indexes = sum(1 for info in collections.values() if info.get("indexes", []))
            if collections_with_indexes == len(collections):
                health_factors["Good Practices"].append({
                    "type": "success",
                    "message": "All collections have indexes"
                })
            elif collections_with_indexes > 0:
                health_factors["Warnings"].append({
                    "type": "warning",
                    "message": f"{len(collections) - collections_with_indexes} collections without indexes"
                })
            else:
                health_score -= 15
                health_factors["Critical Issues"].append({
                    "type": "error",
                    "message": "No indexes found in any collections"
                })
        
        # Ensure health score is between 0 and 100
        health_score = max(0, min(100, health_score))
        
        return health_score, health_factors
    
    def _generate_schema_recommendations(self, schema: Dict) -> Dict[str, List[Dict[str, str]]]:
        """Generate recommendations for schema improvements"""
        recommendations = {
            "Performance Optimizations": [],
            "Data Integrity": [],
            "Naming Conventions": [],
            "Schema Simplification": []
        }
        
        if "tables" in schema:  # SQL database
            tables = schema["tables"]
            
            # Performance optimization recommendations
            for table_name, table_info in tables.items():
                # Check for missing indexes on foreign keys
                for fk in table_info["foreign_keys"]:
                    fk_columns = fk["constrained_columns"]
                    # In a real implementation, we would check if indexes exist for these columns
                    recommendations["Performance Optimizations"].append({
                        "title": f"Consider indexing foreign key in {table_name}",
                        "description": f"Adding an index on {', '.join(fk_columns)} can improve JOIN performance",
                        "sql": f"CREATE INDEX idx_{table_name}_{'_'.join(fk_columns)} ON {table_name} ({', '.join(fk_columns)});"
                    })
            
            # Data integrity recommendations
            tables_without_pk = [name for name, info in tables.items() if not info["primary_keys"]]
            for table in tables_without_pk:
                recommendations["Data Integrity"].append({
                    "title": f"Add primary key to {table}",
                    "description": "Every table should have a primary key for data integrity and performance",
                    "sql": f"ALTER TABLE {table} ADD COLUMN id SERIAL PRIMARY KEY;"
                })
            
            # Naming convention recommendations
            non_snake_tables = [name for name in tables.keys() if not re.match(r'^[a-z][a-z0-9_]*$', name)]
            for table in non_snake_tables:
                snake_name = re.sub(r'([A-Z])', r'_\1', table).lower().lstrip('_')
                recommendations["Naming Conventions"].append({
                    "title": f"Rename table {table} to {snake_name}",
                    "description": "Following consistent naming conventions improves maintainability",
                    "sql": f"ALTER TABLE {table} RENAME TO {snake_name};"
                })
        
        elif "collections" in schema:  # MongoDB
            collections = schema["collections"]
            
            # Performance optimization recommendations
            for coll_name, coll_info in collections.items():
                # Recommend indexes for frequently queried fields
                # In a real implementation, would analyze query patterns
                if coll_info["fields"]:
                    main_field = coll_info["fields"][0]["name"]
                    recommendations["Performance Optimizations"].append({
                        "title": f"Add index on {main_field} in {coll_name} collection",
                        "description": f"Indexing frequently queried fields improves performance",
                        "sql": f"db.{coll_name}.createIndex({{ {main_field}: 1 }});"
                    })
            
            # Schema simplification recommendations
            for coll_name, coll_info in collections.items():
                array_fields = [field["name"] for field in coll_info["fields"] if field["type"] == "array"]
                if len(array_fields) > 1:
                    recommendations["Schema Simplification"].append({
                        "title": f"Consider normalizing arrays in {coll_name}",
                        "description": f"Multiple array fields ({', '.join(array_fields)}) may indicate need for separate collections"
                    })
        
        return recommendations
    
    def _check_normalization(self, schema: Dict) -> Dict[str, List[Dict[str, str]]]:
        """Check for normalization issues in database schema"""
        normalization_issues = {
            "First Normal Form (1NF)": [],
            "Second Normal Form (2NF)": [],
            "Third Normal Form (3NF)": [],
            "Boyce-Codd Normal Form (BCNF)": []
        }
        
        if "tables" in schema:  # SQL database
            tables = schema["tables"]
            
            # 1NF issues - check for potential array/multi-value fields
            for table_name, table_info in tables.items():
                for column in table_info["columns"]:
                    col_name = column["name"]
                    col_type = column["type"].lower()
                    
                    # Check for comma-separated values or array-like fields
                    if any(substr in col_name.lower() for substr in ["list", "array", "values", "tags", "categories"]):
                        normalization_issues["First Normal Form (1NF)"].append({
                            "description": f"Column {col_name} in {table_name} may contain multiple values",
                            "suggestion": f"Create a separate table to store these values",
                            "sql": f"""
CREATE TABLE {table_name}_{col_name} (
    id SERIAL PRIMARY KEY,
    {table_name}_id INTEGER REFERENCES {table_name}(id),
    {col_name}_value VARCHAR(255) NOT NULL
);
"""
                        })
                    
                    # Check for columns that might store JSON or serialized data
                    if any(type_name in col_type for type_name in ["json", "text", "blob", "clob"]):
                        if any(substr in col_name.lower() for substr in ["data", "json", "properties", "attributes", "config", "settings"]):
                            normalization_issues["First Normal Form (1NF)"].append({
                                "description": f"Column {col_name} in {table_name} may store structured data in a single field",
                                "suggestion": f"Extract structured data into separate columns or tables"
                            })
            
            # 2NF issues - check for potential partial dependencies
            for table_name, table_info in tables.items():
                # Skip tables without composite keys
                if len(table_info["primary_keys"]) <= 1:
                    continue
                
                # Check if there are non-key attributes that might depend on part of the key
                non_key_columns = [col["name"] for col in table_info["columns"] 
                                  if col["name"] not in table_info["primary_keys"]]
                
                if non_key_columns:
                    # Look for columns that might depend on just part of the composite key
                    for pk_part in table_info["primary_keys"]:
                        # If a primary key part is a foreign key, columns might depend on just that part
                        is_fk = any(fk["constrained_columns"] == [pk_part] for fk in table_info["foreign_keys"])
                        
                        if is_fk:
                            # Extract the base name from the foreign key (e.g., "user" from "user_id")
                            base_name = pk_part[:-3] if pk_part.endswith("_id") else pk_part
                            
                            # Look for columns that might be related to this entity
                            related_columns = [col for col in non_key_columns 
                                              if base_name in col.lower() or col.startswith(base_name)]
                            
                            if related_columns:
                                normalization_issues["Second Normal Form (2NF)"].append({
                                    "description": f"Table {table_name} may have partial dependencies: {', '.join(related_columns)} might depend only on {pk_part}",
                                    "suggestion": f"Create a separate table with {pk_part} as the primary key and move dependent attributes there",
                                    "sql": f"""
CREATE TABLE {base_name}_details (
    {pk_part} INTEGER PRIMARY KEY,
    {', '.join(f'{col} VARCHAR(255)' for col in related_columns)}
);

-- Then modify {table_name} to remove these columns
ALTER TABLE {table_name} DROP COLUMN {', '.join(related_columns)};
"""
                                })
            
            # 3NF issues - check for potential transitive dependencies
            for table_name, table_info in tables.items():
                # Simplified heuristic - columns with names that suggest lookup values
                lookup_indicators = ["name", "description", "label", "title", "status", "type"]
                
                for col1 in table_info["columns"]:
                    if col1["name"] in table_info["primary_keys"]:
                        continue
                    
                    for col2 in table_info["columns"]:
                        if col2["name"] in table_info["primary_keys"] or col2["name"] == col1["name"]:
                            continue
                        
                        # If col1 is an ID and col2 is a lookup value associated with that ID
                        if (col1["name"].endswith("_id") and 
                            any(col2["name"].endswith(f"_{indicator}") for indicator in lookup_indicators)):
                            
                            base_name = col1["name"][:-3]  # Remove "_id"
                            normalization_issues["Third Normal Form (3NF)"].append({
                                "description": f"Possible transitive dependency in {table_name}: {col2['name']} may depend on {col1['name']} rather than the primary key",
                                "suggestion": f"Consider creating a separate {base_name} table",
                                "sql": f"""
CREATE TABLE {base_name} (
    id SERIAL PRIMARY KEY,
    {col2['name']} VARCHAR(255) NOT NULL
);

ALTER TABLE {table_name} ADD CONSTRAINT fk_{table_name}_{base_name} 
    FOREIGN KEY ({col1['name']}) REFERENCES {base_name}(id);
"""
                            })
                
                # Check for calculated or derived columns
                derived_indicators = ["total", "sum", "average", "count", "calculated", "derived"]
                for column in table_info["columns"]:
                    if any(indicator in column["name"].lower() for indicator in derived_indicators):
                        normalization_issues["Third Normal Form (3NF)"].append({
                            "description": f"Column {column['name']} in {table_name} may be a calculated or derived value",
                            "suggestion": "Consider calculating this value at query time instead of storing it"
                        })
            
            # BCNF issues - check for potential non-trivial functional dependencies
            for table_name, table_info in tables.items():
                # Look for potential candidate keys that aren't the primary key
                unique_columns = []
                
                # In a real implementation, we would analyze the data to find functional dependencies
                # Here we use naming conventions as a heuristic
                for column in table_info["columns"]:
                    col_name = column["name"].lower()
                    # Columns that might be unique
                    if any(unique_indicator in col_name for unique_indicator in 
                          ["email", "username", "code", "number", "uuid", "guid", "sku", "isbn"]):
                        unique_columns.append(column["name"])
                
                if unique_columns and table_info["primary_keys"]:
                    for unique_col in unique_columns:
                        if unique_col not in table_info["primary_keys"]:
                            normalization_issues["Boyce-Codd Normal Form (BCNF)"].append({
                                "description": f"Column {unique_col} in {table_name} may be a candidate key not used as primary key",
                                "suggestion": "Ensure all determinants are candidate keys"
                            })
        
        return normalization_issues
    
    def _generate_schema_from_nl(self, description: str, db_type: str) -> Dict[str, Any]:
        """Generate database schema from natural language description
        
        In a real implementation, this would use a Language Model to extract
        entities, attributes and relationships from the text description.
        Here we implement a simplified example generator for demonstration.
        """
        # This is a simplified example generator
        # In a real implementation, we would use a more sophisticated NLP approach
        
        # Extract potential entity names (capitalized words)
        entities = re.findall(r'\b[A-Z][a-z]+s?\b', description)
        
        # Create a simple schema based on extracted entities
        schema = {}
        
        # For SQL databases
        if db_type in ["postgresql", "mysql", "sqlite"]:
            # First pass: create basic table structures
            for entity in entities:
                # Convert to snake_case and singular
                table_name = re.sub(r'([A-Z])', r'_\1', entity).lower().lstrip('_')
                if table_name.endswith('s'):
                    table_name = table_name[:-1]
                
                # Generate some typical columns based on the entity name
                columns = [
                    "id",
                    f"{table_name}_name",
                    "description",
                    "created_at",
                    "updated_at"
                ]
                
                # Add entity-specific columns based on keywords in description
                keywords = {
                    "user": ["email", "password", "username", "role"],
                    "product": ["price", "sku", "stock_level", "category_id"],
                    "order": ["order_date", "total_amount", "status", "user_id"],
                    "customer": ["first_name", "last_name", "email", "phone"],
                    "category": ["category_name", "parent_id"],
                    "post": ["title", "content", "author_id", "published_at"],
                    "comment": ["content", "post_id", "user_id", "is_approved"]
                }
                
                # Add any matching keyword columns
                for key, key_columns in keywords.items():
                    if key in table_name or key in description.lower():
                        for col in key_columns:
                            if col not in columns:
                                columns.append(col)
                
                # Identify potential relations based on description
                relations = []
                for other_entity in entities:
                    other_table = re.sub(r'([A-Z])', r'_\1', other_entity).lower().lstrip('_')
                    if other_table.endswith('s'):
                        other_table = other_table[:-1]
                    
                    if other_table != table_name:
                        # Look for relationship indicators in description
                        rel_indicators = [
                            (f"{entity} has many {other_entity}", "one-to-many"),
                            (f"{entity} belongs to {other_entity}", "many-to-one"),
                            (f"{entity} contains {other_entity}", "one-to-many"),
                            (f"{other_entity} has many {entity}", "many-to-one"),
                            (f"{entity} has a {other_entity}", "many-to-one"),
                            (f"{entity} has {other_entity}", "many-to-one"),
                        ]
                        
                        for pattern, rel_type in rel_indicators:
                            if pattern.lower() in description.lower():
                                relations.append({
                                    "table": other_table,
                                    "type": rel_type,
                                    "fk_column": f"{other_table}_id"
                                })
                                break
                
                schema[table_name] = {
                    "columns": columns,
                    "primary_keys": ["id"],
                    "relations": relations
                }
            
            # Second pass: add foreign key columns for relations
            for table_name, table_info in schema.items():
                for relation in table_info["relations"]:
                    if relation["type"] == "many-to-one":
                        # This table needs a foreign key to the related table
                        fk_column = relation["fk_column"]
                        if fk_column not in table_info["columns"]:
                            table_info["columns"].append(fk_column)
                    elif relation["type"] == "one-to-many":
                        # The related table needs a foreign key to this table
                        related_table = relation["table"]
                        if related_table in schema:
                            fk_column = f"{table_name}_id"
                            if fk_column not in schema[related_table]["columns"]:
                                schema[related_table]["columns"].append(fk_column)
                            # Add reverse relation to the related table
                            reverse_relation = {
                                "table": table_name,
                                "type": "many-to-one",
                                "fk_column": fk_column
                            }
                            if reverse_relation not in schema[related_table]["relations"]:
                                schema[related_table]["relations"].append(reverse_relation)
        
        # For MongoDB
        elif db_type == "mongodb":
            for entity in entities:
                # Convert to snake_case and plural
                collection_name = re.sub(r'([A-Z])', r'_\1', entity).lower().lstrip('_')
                if not collection_name.endswith('s'):
                    collection_name += 's'
                
                # Generate fields for this collection
                fields = [
                    {"name": "_id", "type": "ObjectId"},
                    {"name": "name", "type": "string"},
                    {"name": "description", "type": "string"},
                    {"name": "created_at", "type": "date"},
                    {"name": "updated_at", "type": "date"}
                ]
                
                # Add entity-specific fields based on keywords in description
                keywords = {
                    "user": [
                        {"name": "email", "type": "string"},
                        {"name": "password", "type": "string"},
                        {"name": "username", "type": "string"},
                        {"name": "role", "type": "string"}
                    ],
                    "product": [
                        {"name": "price", "type": "number"},
                        {"name": "sku", "type": "string"},
                        {"name": "stock_level", "type": "number"},
                        {"name": "category", "type": "object"}
                    ],
                    "order": [
                        {"name": "order_date", "type": "date"},
                        {"name": "total_amount", "type": "number"},
                        {"name": "status", "type": "string"},
                        {"name": "user", "type": "object"},
                        {"name": "items", "type": "array"}
                    ]
                }
                
                # Add any matching keyword fields
                for key, key_fields in keywords.items():
                    if key in collection_name or key in description.lower():
                        for field in key_fields:
                            if not any(f["name"] == field["name"] for f in fields):
                                fields.append(field)
                
                schema[collection_name] = {
                    "fields": fields,
                    "indexes": [{"keys": {"_id": 1}, "unique": True}],
                    "validators": {
                        "validator": {
                            "$jsonSchema": {
                                "bsonType": "object",
                                "required": ["name", "created_at"],
                                "properties": {
                                    "name": {"bsonType": "string"},
                                    "created_at": {"bsonType": "date"}
                                }
                            }
                        }
                    }
                }
        
        return schema
    
    def _generate_sql_from_schema(self, schema: Dict[str, Any], db_type: str) -> str:
        """Generate SQL creation script from schema definition"""
        sql_statements = []
        foreign_key_statements = []
        
        # Define column types based on DB type
        type_mapping = {}
        if db_type == "postgresql":
            type_mapping = {
                "id": "SERIAL PRIMARY KEY",
                "created_at": "TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"
            }
            default_type = "VARCHAR(255)"
        elif db_type == "mysql":
            type_mapping = {
                "id": "INT AUTO_INCREMENT PRIMARY KEY",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
            }
            default_type = "VARCHAR(255)"
        elif db_type == "sqlite":
            type_mapping = {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            }
            default_type = "TEXT"
        
        # Special type mappings for common column patterns
        pattern_mapping = {
            "_id$": "INTEGER",
            "price|amount|total": "DECIMAL(10, 2)",
            "count|quantity|level": "INTEGER",
            "is_|has_": "BOOLEAN",
            "date$|_at$": "TIMESTAMP",
            "email": "VARCHAR(255)"
        }
        
        # Generate CREATE TABLE statements
        for table_name, table_info in schema.items():
            columns = []
            fk_constraints = []
            
            # Generate column definitions
            for col in table_info["columns"]:
                if col in type_mapping:
                    columns.append(f"{col} {type_mapping[col]}")
                else:
                    # Try to determine type based on patterns
                    col_type = default_type
                    for pattern, type_val in pattern_mapping.items():
                        if re.search(pattern, col):
                            col_type = type_val
                            break
                    
                    # Add NOT NULL for foreign keys
                    not_null = ""
                    if col.endswith("_id") and col != "id":
                        # Check if this is a foreign key
                        for relation in table_info.get("relations", []):
                            if relation["type"] == "many-to-one" and relation.get("fk_column") == col:
                                # Optional: make FK nullable or not
                                # not_null = " NOT NULL"
                                break
                    
                    columns.append(f"{col} {col_type}{not_null}")
            
            # Collect foreign key constraints
            for relation in table_info.get("relations", []):
                if relation["type"] == "many-to-one":
                    related_table = relation["table"]
                    fk_column = relation.get("fk_column", f"{related_table}_id")
                    
                    # Ensure the FK column exists in this table
                    if fk_column in table_info["columns"]:
                        fk_constraints.append(
                            f"CONSTRAINT fk_{table_name}_{related_table} "
                            f"FOREIGN KEY ({fk_column}) REFERENCES {related_table}(id)"
                        )
            
            # Create the CREATE TABLE statement
            sql = f"CREATE TABLE {table_name} (\n  "
            sql += ",\n  ".join(columns)
            
            # Add foreign key constraints inline
            if fk_constraints:
                sql += ",\n  " + ",\n  ".join(fk_constraints)
            
            sql += "\n);"
            sql_statements.append(sql)
        
        return "\n\n".join(sql_statements)
    
    def _convert_to_full_schema_format(self, generated_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert simplified generated schema to full database schema format for visualization"""
        full_schema = {"tables": {}}
        
        for table_name, table_info in generated_schema.items():
            # Convert column list to full column format
            columns = []
            for col_name in table_info["columns"]:
                columns.append({
                    "name": col_name,
                    "type": "VARCHAR(255)"  # Default type
                })
            
            # Convert relations to foreign keys
            foreign_keys = []
            for relation in table_info.get("relations", []):
                if relation["type"] == "many-to-one":
                    fk_column = relation.get("fk_column", f"{relation['table']}_id")
                    foreign_keys.append({
                        "constrained_columns": [fk_column],
                        "referred_table": relation["table"],
                        "referred_columns": ["id"]
                    })
            
            full_schema["tables"][table_name] = {
                "columns": columns,
                "primary_keys": table_info.get("primary_keys", ["id"]),
                "foreign_keys": foreign_keys
            }
        
        return full_schema
    
    def _display_graphviz_diagram(self, schema: Dict[str, Any]):
        """Display Graphviz ER diagram for generated schema"""
        import graphviz
        
        if "tables" not in schema or not schema["tables"]:
            st.info("No tables to visualize")
            return
        
        # Create a new Graphviz graph
        graph = graphviz.Digraph(
            name="Generated Schema ER Diagram",
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
                'dir': 'both',
                'color': '#4D7A97',
                'penwidth': '1.5'
            },
            graph_attr={
                'bgcolor': 'white',
                'rankdir': 'LR',
                'splines': 'polyline',
                'concentrate': 'true'
            }
        )
        
        # Add nodes (tables) with HTML-like labels
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
                
                # Truncate long type names
                col_type = column["type"]
                if len(col_type) > 15:
                    col_type = col_type[:13] + "..."
                
                # Add row for this column
                label += f'''
                <TR BGCOLOR="{bg_color}">
                    <TD ALIGN="LEFT" PORT="{column["name"]}">{column["name"]}</TD>
                    <TD ALIGN="LEFT">{col_type}</TD>
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
        
        # Display the graph in a scrollable container
        st.markdown("""<style>
        .graphviz-container { 
            max-height: 600px; 
            overflow: auto; 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            padding: 10px;
            background-color: white;
        }
        </style>""", unsafe_allow_html=True)
        
        with st.container():
            st.graphviz_chart(graph, use_container_width=False)
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            export_format = st.selectbox(
                "Export Format",
                options=["PNG", "PDF", "SVG"],
                index=0,
                key="nl_export_format"
            )
        
        with col2:
            if st.button("📥 Export Diagram", key="nl_export_btn"):
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
                    
                    st.download_button(
                        label=f"Download {export_format}",
                        data=graph_data,
                        file_name=f"schema_diagram.{export_format.lower()}",
                        mime=mime_type,
                        key="nl_download_btn"
                    )
                except Exception as e:
                    st.error(f"Error exporting diagram: {str(e)}")
