import streamlit as st
import pandas as pd
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set

class SemanticLayer:
    def __init__(self):
        """Initialize the Semantic Layer"""
        # Initialize session state for semantic models
        if "semantic_models" not in st.session_state:
            st.session_state.semantic_models = {}
        
        if "semantic_metrics" not in st.session_state:
            st.session_state.semantic_metrics = {}
    
    def semantic_layer_ui(self, db_manager):
        """UI for semantic layer management"""
        st.subheader("Semantic Layer")
        
        # Check if connected to a database
        if not st.session_state.get("connected_db"):
            st.warning("Please connect to a database first.")
            return
        
        tab1, tab2, tab3, tab4 = st.tabs(["Model Designer", "Business Metrics", "Natural Language Query", "Import/Export"])
        
        with tab1:
            self._model_designer_ui(db_manager)
        
        with tab2:
            self._business_metrics_ui()
        
        with tab3:
            self._nl_query_ui(db_manager)
            
        with tab4:
            self._import_export_ui()
    
    def _model_designer_ui(self, db_manager):
        """UI for designing semantic models"""
        st.subheader("Semantic Model Designer")
        
        # Get database schema
        schema = st.session_state.get("db_schema", {})
        
        # If schema is not available, try to retrieve it
        if not schema and st.session_state.get("connected_db"):
            schema = db_manager.get_database_schema()
            
        if not schema:
            st.warning("No schema information available. Please make sure you are connected to a database with tables.")
            st.info("If you've just connected to a database, try refreshing the page.")
            
            # Add debugging info
            if st.session_state.get("connected_db"):
                st.write("Connected database:", st.session_state.get("connected_db"))
                if st.session_state.get("current_connection"):
                    conn_type = st.session_state.get("current_connection").get("type", "Unknown")
                    st.write("Connection type:", conn_type)
                    
                    # Add button to force refresh schema
                    if st.button("Force Refresh Schema"):
                        schema = db_manager.get_database_schema()
                        if schema:
                            st.success("Schema loaded successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to load schema. Please check your database connection.")
            return
        
        # Sidebar for creating/selecting models
        with st.sidebar:
            st.subheader("Models")
            
            # Create new model
            new_model_name = st.text_input("New Model Name", key="new_model_name_input")
            if st.button("Create Model", key="create_model_btn") and new_model_name:
                if new_model_name in st.session_state.semantic_models:
                    st.error(f"Model '{new_model_name}' already exists")
                else:
                    st.session_state.semantic_models[new_model_name] = {
                        "entities": {},
                        "relationships": [],
                        "description": ""
                    }
                    st.session_state.current_model = new_model_name
                    st.success(f"Created model: {new_model_name}")
                    st.rerun()
            
            # Select existing model
            if st.session_state.semantic_models:
                model_options = list(st.session_state.semantic_models.keys())
                current_model = st.selectbox(
                    "Select Model",
                    options=model_options,
                    index=0 if "current_model" not in st.session_state else model_options.index(st.session_state.current_model),
                    key="model_designer_model_selector"
                )
                st.session_state.current_model = current_model
            else:
                st.info("No models created yet")
                return
        
        # Main content for model editing
        current_model = st.session_state.current_model
        model_data = st.session_state.semantic_models[current_model]
        
        # Model description
        model_description = st.text_area(
            "Model Description",
            value=model_data.get("description", ""),
            height=100,
            key="model_description_area"
        )
        model_data["description"] = model_description
        
        # Show available tables/collections from schema
        st.subheader("Available Data Sources")
        source_options = []
        if "tables" in schema:
            source_options = list(schema["tables"].keys())
        elif "collections" in schema:
            source_options = list(schema["collections"].keys())
        
        # Add entity to the model
        st.subheader("Add Entity to Model")
        col1, col2 = st.columns(2)
        
        with col1:
            entity_source = st.selectbox("Select Data Source", options=source_options, key="entity_source_selector")
        
        with col2:
            entity_name = st.text_input("Business Entity Name", value=entity_source.title(), key="entity_name_input")
        
        if st.button("Add Entity", key="add_entity_btn") and entity_source and entity_name:
            # Check if entity already exists
            if entity_name in model_data["entities"]:
                st.error(f"Entity '{entity_name}' already exists in the model")
            else:
                # Get columns/fields for the entity
                entity_fields = []
                if "tables" in schema:
                    entity_fields = [col["name"] for col in schema["tables"][entity_source]["columns"]]
                elif "collections" in schema:
                    entity_fields = [field["name"] for field in schema["collections"][entity_source]["fields"]]
                
                # Add entity to the model
                model_data["entities"][entity_name] = {
                    "source": entity_source,
                    "fields": {field: {"source": field, "display_name": field.replace("_", " ").title(), "visible": True} for field in entity_fields},
                    "display_name": entity_name,
                    "description": ""
                }
                
                st.success(f"Added entity: {entity_name}")
                st.rerun()
        
        # Show and edit entities in the model
        if model_data["entities"]:
            st.subheader("Model Entities")
            
            for entity_name, entity_data in model_data["entities"].items():
                with st.expander(f"{entity_name} ({entity_data['source']})", expanded=False):
                    # Entity display name
                    entity_display_name = st.text_input(
                        "Display Name",
                        value=entity_data.get("display_name", entity_name),
                        key=f"display_{entity_name}"
                    )
                    entity_data["display_name"] = entity_display_name
                    
                    # Entity description
                    entity_description = st.text_area(
                        "Description",
                        value=entity_data.get("description", ""),
                        height=100,
                        key=f"desc_{entity_name}"
                    )
                    entity_data["description"] = entity_description
                    
                    # Entity fields
                    st.subheader("Fields")
                    
                    for field_name, field_data in entity_data["fields"].items():
                        col1, col2, col3 = st.columns([3, 3, 1])
                        
                        with col1:
                            st.text(field_name)
                        
                        with col2:
                            field_display_name = st.text_input(
                                "Display Name",
                                value=field_data.get("display_name", field_name.replace("_", " ").title()),
                                key=f"display_{entity_name}_{field_name}"
                            )
                            field_data["display_name"] = field_display_name
                        
                        with col3:
                            field_visible = st.checkbox(
                                "Visible",
                                value=field_data.get("visible", True),
                                key=f"visible_{entity_name}_{field_name}"
                            )
                            field_data["visible"] = field_visible
                    
                    # Delete entity button
                    if st.button("Delete Entity", key=f"delete_{entity_name}"):
                        del model_data["entities"][entity_name]
                        # Also remove any relationships involving this entity
                        model_data["relationships"] = [r for r in model_data["relationships"] 
                                                      if r["from_entity"] != entity_name and r["to_entity"] != entity_name]
                        st.success(f"Deleted entity: {entity_name}")
                        st.rerun()
            
            # Add relationships between entities
            if len(model_data["entities"]) > 1:
                st.subheader("Manage Relationships")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    from_entity = st.selectbox("From Entity", options=list(model_data["entities"].keys()), key="rel_from")
                
                with col2:
                    to_entity = st.selectbox("To Entity", options=[e for e in model_data["entities"].keys() if e != from_entity], key="rel_to")
                
                with col3:
                    rel_type = st.selectbox("Relationship Type", options=["One-to-Many", "Many-to-One", "One-to-One", "Many-to-Many"], key="rel_type")
                
                # Get fields from both entities
                from_fields = list(model_data["entities"][from_entity]["fields"].keys())
                to_fields = list(model_data["entities"][to_entity]["fields"].keys())
                
                col1, col2 = st.columns(2)
                
                with col1:
                    from_field = st.selectbox("From Field", options=from_fields, key="rel_from_field")
                
                with col2:
                    to_field = st.selectbox("To Field", options=to_fields, key="rel_to_field")
                
                if st.button("Add Relationship", key="add_relationship_btn"):
                    # Check if relationship already exists
                    existing_rel = any(
                        r["from_entity"] == from_entity and r["to_entity"] == to_entity and 
                        r["from_field"] == from_field and r["to_field"] == to_field
                        for r in model_data["relationships"]
                    )
                    
                    if existing_rel:
                        st.error("This relationship already exists")
                    else:
                        model_data["relationships"].append({
                            "from_entity": from_entity,
                            "to_entity": to_entity,
                            "from_field": from_field,
                            "to_field": to_field,
                            "type": rel_type,
                            "description": f"Relates {from_entity} to {to_entity}"
                        })
                        st.success("Added relationship")
                
                # Show existing relationships
                if model_data["relationships"]:
                    st.subheader("Existing Relationships")
                    
                    for i, rel in enumerate(model_data["relationships"]):
                        st.write(f"{rel['from_entity']}.{rel['from_field']} → {rel['to_entity']}.{rel['to_field']} ({rel['type']})")
                        
                        if st.button("Delete", key=f"del_rel_{i}"):
                            model_data["relationships"].pop(i)
                            st.success("Deleted relationship")
                            st.rerun()
            
            # Generate SQL for the semantic model
            if st.button("Generate Model SQL"):
                sql = self._generate_model_sql(model_data, schema)
                
                st.subheader("Generated SQL")
                st.code(sql, language="sql")
    
    def _business_metrics_ui(self):
        """UI for defining business metrics based on semantic models"""
        st.subheader("Business Metrics Designer")
        
        # Check if any models exist
        if not st.session_state.semantic_models:
            st.warning("Please create a semantic model first")
            return
        
        # Sidebar for creating/selecting metrics
        with st.sidebar:
            st.subheader("Metrics")
            
            # Select model
            model_options = list(st.session_state.semantic_models.keys())
            selected_model = st.selectbox(
                "Select Model",
                options=model_options,
                index=0,
                key="metrics_model_selector"  # Updated key name
            )
            
            # Create new metric
            new_metric_name = st.text_input("New Metric Name", key="new_metric_name_input")
            if st.button("Create Metric", key="create_metric_btn") and new_metric_name:
                if selected_model not in st.session_state.semantic_metrics:
                    st.session_state.semantic_metrics[selected_model] = {}
                
                if new_metric_name in st.session_state.semantic_metrics[selected_model]:
                    st.error(f"Metric '{new_metric_name}' already exists")
                else:
                    st.session_state.semantic_metrics[selected_model][new_metric_name] = {
                        "name": new_metric_name,
                        "description": "",
                        "definition": "",
                        "type": "measure",
                        "format": "number",
                        "entity": "",
                        "expression": ""
                    }
                    st.session_state.current_metric = new_metric_name
                    st.success(f"Created metric: {new_metric_name}")
                    st.rerun()
            
            # Select existing metric
            if selected_model in st.session_state.semantic_metrics and st.session_state.semantic_metrics[selected_model]:
                metric_options = list(st.session_state.semantic_metrics[selected_model].keys())
                current_metric = st.selectbox(
                    "Select Metric",
                    options=metric_options,
                    index=0 if "current_metric" not in st.session_state else 
                          (metric_options.index(st.session_state.current_metric) 
                           if st.session_state.current_metric in metric_options else 0),
                    key="metrics_metric_selector"  # Updated key name
                )
                st.session_state.current_metric = current_metric
            else:
                st.info("No metrics created yet")
                return
        
        # Main content for metric editing
        current_model = selected_model
        current_metric = st.session_state.current_metric
        metric_data = st.session_state.semantic_metrics[current_model][current_metric]
        model_data = st.session_state.semantic_models[current_model]
        
        # Metric properties
        st.subheader(f"Edit Metric: {current_metric}")
        
        # Metric description
        metric_description = st.text_area(
            "Business Description",
            value=metric_data.get("description", ""),
            height=100,
            key="metric_description_area"  # Added unique key
        )
        metric_data["description"] = metric_description
        
        # Metric type
        metric_type = st.selectbox(
            "Metric Type",
            options=["Measure", "Dimension", "Calculated Measure"],
            index=["measure", "dimension", "calculated"].index(metric_data.get("type", "measure").lower()),
            key="metric_type_selector"  # Added unique key
        )
        metric_data["type"] = metric_type.lower()
        
        # Metric format
        format_options = ["Number", "Currency", "Percentage", "Date", "Text"]
        metric_format = st.selectbox(
            "Display Format",
            options=format_options,
            index=format_options.index(metric_data.get("format", "Number").capitalize()),
            key="metric_format_selector"  # Added unique key
        )
        metric_data["format"] = metric_format.lower()
        
        # Entity selection
        if "entities" not in model_data or not model_data["entities"]:
            st.warning("No entities defined in this model. Please add entities first.")
            entity_options = [""]
            entity = ""
        else:
            entity_options = list(model_data["entities"].keys())
            # Safely get the index
            if metric_data.get("entity") in entity_options:
                index = entity_options.index(metric_data["entity"])
            else:
                index = 0
                
            entity = st.selectbox(
                "Entity",
                options=entity_options,
                index=index,
                key="metric_entity_selector"  # Added unique key
            )
        
        metric_data["entity"] = entity
        
        # Field/Expression based on metric type
        if metric_type.lower() in ["measure", "dimension"]:
            # Check if the entity exists in the model data
            if entity not in model_data.get("entities", {}):
                st.error(f"Entity '{entity}' not found in the model. Please select a valid entity.")
                field_options = []
            else:
                # Check if the entity has fields
                if "fields" not in model_data["entities"][entity]:
                    st.warning(f"Entity '{entity}' has no fields defined. Please add fields to this entity first.")
                    field_options = []
                else:
                    # Get field options from the entity
                    field_options = list(model_data["entities"][entity]["fields"].keys())
            
            # Only show field selector if we have fields
            if field_options:
                field = st.selectbox(
                    "Field",
                    options=field_options,
                    index=field_options.index(metric_data["expression"]) if metric_data.get("expression") in field_options else 0,
                    key="metric_field_selector"  # Added unique key
                )
                metric_data["expression"] = field
            else:
                # If no fields are available, show a text input instead
                field = st.text_input(
                    "Field (enter manually)",
                    value=metric_data.get("expression", ""),
                    key="metric_field_text_input"
                )
                metric_data["expression"] = field
            
            # For measures, add aggregation
            if metric_type.lower() == "measure":
                agg_options = ["SUM", "AVG", "MIN", "MAX", "COUNT", "COUNT DISTINCT"]
                
                # Extract current aggregation if any
                current_agg = "SUM"
                if "aggregation" in metric_data:
                    current_agg = metric_data["aggregation"]
                
                aggregation = st.selectbox(
                    "Aggregation",
                    options=agg_options,
                    index=agg_options.index(current_agg) if current_agg in agg_options else 0,
                    key="metric_aggregation_selector"  # Added unique key
                )
                metric_data["aggregation"] = aggregation
        else:
            # Expression for calculated measures
            expression = st.text_area(
                "Expression",
                value=metric_data.get("expression", ""),
                height=100,
                help="Use other metrics in calculations e.g., [Revenue] / [Cost]",
                key="metric_expression_area"  # Added unique key
            )
            metric_data["expression"] = expression
        
        # Generate SQL for the metric
        if st.button("Generate Metric SQL", key="generate_metric_sql_btn"):
            sql = self._generate_metric_sql(current_model, metric_data)
            
            st.subheader("Generated SQL")
            st.code(sql, language="sql")
        
        # Delete metric button
        if st.button("Delete Metric", key="delete_metric_btn"):
            del st.session_state.semantic_metrics[current_model][current_metric]
            st.success(f"Deleted metric: {current_metric}")
            st.rerun()
    
    def _nl_query_ui(self, db_manager):
        """UI for natural language querying of semantic models"""
        st.subheader("Natural Language Business Query")
        
        # Check if any models exist
        if not st.session_state.semantic_models:
            st.warning("Please create a semantic model first")
            return
        
        # Check if any metrics exist
        has_metrics = False
        for model in st.session_state.semantic_models:
            if model in st.session_state.semantic_metrics and st.session_state.semantic_metrics[model]:
                has_metrics = True
                break
        
        if not has_metrics:
            st.warning("Please define some business metrics first")
            return
        
        # Select model
        model_options = [model for model in st.session_state.semantic_models 
                        if model in st.session_state.semantic_metrics and st.session_state.semantic_metrics[model]]
        
        selected_model = st.selectbox(
            "Select Business Model",
            options=model_options,
            index=0,
            key="nl_query_model_selector"
        )
        
        # Display available metrics
        metrics = st.session_state.semantic_metrics[selected_model]
        
        with st.expander("Available Business Metrics"):
            for metric_name, metric_data in metrics.items():
                st.write(f"**{metric_name}** ({metric_data['type'].capitalize()})")
                st.write(f"*{metric_data['description']}*")
        
        # Natural language query input
        nl_query = st.text_area(
            "Ask a business question in plain English",
            height=100,
            placeholder="e.g., What was the total revenue by product category last month?",
            key="nl_query_input"
        )
        
        # Process query
        if st.button("Get Answer", key="get_nl_answer_btn") and nl_query:
            with st.spinner("Analyzing your question..."):
                # In a real implementation, this would use an NLP model to convert
                # the natural language query to a structured query against the semantic layer.
                # Here we'll simulate this with a sample implementation.
                
                # Extract potential metric and dimension mentions
                mentioned_metrics = []
                mentioned_dimensions = []
                
                for metric_name, metric_data in metrics.items():
                    if metric_name.lower() in nl_query.lower() or metric_data["description"].lower() in nl_query.lower():
                        mentioned_metrics.append(metric_name)
                
                # Look for potential dimensions
                dimension_metrics = [m for m, data in metrics.items() if data["type"] == "dimension"]
                for dim in dimension_metrics:
                    if dim.lower() in nl_query.lower() or metrics[dim]["description"].lower() in nl_query.lower():
                        mentioned_dimensions.append(dim)
                
                # Generate a simulated query
                if mentioned_metrics:
                    # Generate SQL based on detected metrics and dimensions
                    sql = self._generate_nl_query_sql(selected_model, mentioned_metrics, mentioned_dimensions, nl_query)
                    
                    # Display the SQL for transparency
                    with st.expander("Generated SQL"):
                        st.code(sql, language="sql")
                    
                    # Execute the query
                    try:
                        result = db_manager.execute_query(sql)
                        
                        if isinstance(result, pd.DataFrame) and not result.empty:
                            st.subheader("Query Results")
                            st.dataframe(result)
                            
                            # Simple visualization based on result shape
                            if len(result.columns) >= 2 and len(result) > 1:
                                st.subheader("Visualization")
                                
                                if len(result.columns) == 2:
                                    # Bar chart for 2 columns (dimension + measure)
                                    st.bar_chart(result.set_index(result.columns[0]))
                                elif len(result.columns) == 3:
                                    # Line chart for 3 columns (typically time series)
                                    st.line_chart(result.set_index(result.columns[0]))
                        else:
                            st.info("No results found for your query")
                            
                    except Exception as e:
                        st.error(f"Error executing query: {str(e)}")
                else:
                    st.warning("Couldn't identify any metrics in your question. Please try again with a more specific question.")
    
    def _generate_model_sql(self, model_data: Dict[str, Any], schema: Dict[str, Any]) -> str:
        """Generate SQL view definition for a semantic model"""
        entities = model_data["entities"]
        relationships = model_data["relationships"]
        
        if not entities:
            return "-- No entities defined in the model"
        
        # Start with the first entity as the base table
        base_entity_name = list(entities.keys())[0]
        base_entity = entities[base_entity_name]
        base_table = base_entity["source"]
        
        # Build SQL with CTEs for each entity
        sql_parts = []
        
        # Add base entity CTE
        base_fields = []
        for field_name, field_data in base_entity["fields"].items():
            if field_data["visible"]:
                base_fields.append(f"{field_name} AS {field_data['display_name'].replace(' ', '_')}")
        
        base_cte = f"""WITH {base_table}_base AS (
    SELECT 
        {', '.join(base_fields)}
    FROM {base_table}
)"""
        sql_parts.append(base_cte)
        
        # Process relationships to build JOINs
        processed_entities = {base_entity_name}
        join_queue = []
        
        # Find all direct relationships to the base entity
        for rel in relationships:
            if rel["from_entity"] == base_entity_name and rel["to_entity"] not in processed_entities:
                join_queue.append({
                    "entity": rel["to_entity"],
                    "from_entity": base_entity_name,
                    "from_field": rel["from_field"],
                    "to_field": rel["to_field"],
                    "type": rel["type"]
                })
            elif rel["to_entity"] == base_entity_name and rel["from_entity"] not in processed_entities:
                join_queue.append({
                    "entity": rel["from_entity"],
                    "from_entity": base_entity_name,
                    "from_field": rel["to_field"],
                    "to_field": rel["from_field"],
                    "type": rel["type"]  # Need to invert this for correct join type
                })
        
        # Process the join queue
        while join_queue:
            join_info = join_queue.pop(0)
            entity_name = join_info["entity"]
            
            if entity_name in processed_entities:
                continue
            
            entity = entities[entity_name]
            table = entity["source"]
            
            # Add fields for this entity
            entity_fields = []
            for field_name, field_data in entity["fields"].items():
                if field_data["visible"]:
                    # Prefix with table name to avoid column name conflicts
                    entity_fields.append(f"{field_name} AS {entity_name}_{field_data['display_name'].replace(' ', '_')}")
            
            # Determine join type
            join_type = "LEFT JOIN"
            if join_info["type"] == "One-to-One":
                join_type = "INNER JOIN"
            elif join_info["type"] == "Many-to-Many":
                # Many-to-many requires a junction table, which is not handled in this simplified implementation
                join_type = "LEFT JOIN"
            
            # Create CTE for this entity
            entity_cte = f"""{table}_view AS (
    SELECT 
        {', '.join(entity_fields)}
    FROM {table}
)"""
            sql_parts.append(entity_cte)
            
            processed_entities.add(entity_name)
            
            # Add any relationships from this entity to the queue
            for rel in relationships:
                if rel["from_entity"] == entity_name and rel["to_entity"] not in processed_entities:
                    join_queue.append({
                        "entity": rel["to_entity"],
                        "from_entity": entity_name,
                        "from_field": rel["from_field"],
                        "to_field": rel["to_field"],
                        "type": rel["type"]
                    })
                elif rel["to_entity"] == entity_name and rel["from_entity"] not in processed_entities:
                    join_queue.append({
                        "entity": rel["from_entity"],
                        "from_entity": entity_name,
                        "from_field": rel["to_field"],
                        "to_field": rel["from_field"],
                        "type": rel["type"]  # Need to invert this for correct join type
                    })
        
        # Final SELECT with all JOINs
        select_parts = []
        from_clause = f"{base_table}_base b"
        
        # Add fields from base entity
        for field_name, field_data in base_entity["fields"].items():
            if field_data["visible"]:
                display_name = field_data["display_name"].replace(" ", "_")
                select_parts.append(f"b.{display_name}")
        
        # Add JOINs and fields from related entities
        join_clauses = []
        for entity_name in processed_entities:
            if entity_name == base_entity_name:
                continue
            
            entity = entities[entity_name]
            table = entity["source"]
            alias = f"{entity_name.lower()}"
            
            # Find the relationship to use for joining
            join_rel = None
            for rel in relationships:
                if (rel["from_entity"] == base_entity_name and rel["to_entity"] == entity_name) or \
                   (rel["to_entity"] == base_entity_name and rel["from_entity"] == entity_name):
                    join_rel = rel
                    break
            
            if join_rel:
                # Determine which side of the relationship this entity is on
                if join_rel["from_entity"] == entity_name:
                    join_clause = f"LEFT JOIN {table}_view {alias} ON b.{join_rel['to_field']} = {alias}.{entity_name}_{join_rel['from_field']}"
                else:
                    join_clause = f"LEFT JOIN {table}_view {alias} ON b.{join_rel['from_field']} = {alias}.{entity_name}_{join_rel['to_field']}"
                
                join_clauses.append(join_clause)
                
                # Add fields from this entity
                for field_name, field_data in entity["fields"].items():
                    if field_data["visible"]:
                        display_name = field_data["display_name"].replace(" ", "_")
                        select_parts.append(f"{alias}.{entity_name}_{display_name}")
        
        # Assemble the final query
        # Use raw string to avoid issue with \n
        joined_parts = ',\n    '.join(select_parts)
        final_select = f"""
SELECT 
    {joined_parts}
FROM {from_clause}
{' '.join(join_clauses)}
"""
        
        sql_parts.append(final_select)
        
        # Combine all parts
        model_sql = "\n,\n".join(sql_parts)
        
        return model_sql
    
    def _generate_metric_sql(self, model_name: str, metric_data: Dict[str, Any]) -> str:
        """Generate SQL for a specific metric"""
        # Get the semantic model
        model = st.session_state.semantic_models.get(model_name, {})
        
        if not model or not model.get("entities"):
            return "-- No model entities defined"
        
        # Get the entity for this metric
        entity_name = metric_data.get("entity", "")
        entity = model["entities"].get(entity_name, {})
        
        if not entity:
            return f"-- Entity {entity_name} not found in model"
        
        # Get the source table
        source_table = entity.get("source", "")
        
        # Generate SQL based on metric type
        metric_type = metric_data.get("type", "").lower()
        expression = metric_data.get("expression", "")
        
        if metric_type == "measure":
            # Get aggregation
            aggregation = metric_data.get("aggregation", "SUM").upper()
            
            sql = f"""
SELECT
    {aggregation}({source_table}.{expression}) AS {metric_data['name']}
FROM {source_table}
"""
            
            # Add any necessary joins based on the semantic model
            joins = self._generate_joins_for_entity(model, entity_name)
            if joins:
                sql += f"\n{joins}"
            
            return sql
            
        elif metric_type == "dimension":
            sql = f"""
SELECT DISTINCT
    {source_table}.{expression} AS {metric_data['name']}
FROM {source_table}
"""
            
            # Add any necessary joins based on the semantic model
            joins = self._generate_joins_for_entity(model, entity_name)
            if joins:
                sql += f"\n{joins}"
            
            return sql
            
        elif metric_type == "calculated":
            # For calculated measures, we need to parse the expression and replace
            # metric references with their SQL definitions
            
            # In a real implementation, this would properly parse the expression
            # and substitute the metrics. Here we'll do a simplified version.
            calc_expression = expression
            
            # Find all metric references in [brackets]
            metric_refs = re.findall(r'\[([^\]]+)\]', calc_expression)
            
            # Base query template
            sql = f"""
SELECT
    -- Expression: {expression}
"""
            
            # Replace each metric reference with its SQL
            for ref in metric_refs:
                if model_name in st.session_state.semantic_metrics and ref in st.session_state.semantic_metrics[model_name]:
                    ref_metric = st.session_state.semantic_metrics[model_name][ref]
                    ref_entity = model["entities"].get(ref_metric.get("entity", ""), {})
                    ref_source = ref_entity.get("source", "")
                    
                    if ref_metric.get("type") == "measure":
                        ref_agg = ref_metric.get("aggregation", "SUM").upper()
                        ref_expr = ref_metric.get("expression", "")
                        sql_fragment = f"{ref_agg}({ref_source}.{ref_expr})"
                    else:
                        sql_fragment = f"{ref_source}.{ref_metric.get('expression', '')}"
                    
                    # Replace in the calculated expression
                    calc_expression = calc_expression.replace(f"[{ref}]", sql_fragment)
            
            # Add the calculated expression to the SQL
            sql += f"    {calc_expression} AS {metric_data['name']}\n"
            
            # Add FROM and joins
            source_tables = set()
            for ref in metric_refs:
                if model_name in st.session_state.semantic_metrics and ref in st.session_state.semantic_metrics[model_name]:
                    ref_metric = st.session_state.semantic_metrics[model_name][ref]
                    ref_entity = model["entities"].get(ref_metric.get("entity", ""), {})
                    ref_source = ref_entity.get("source", "")
                    if ref_source:
                        source_tables.add(ref_source)
            
            if source_tables:
                main_table = list(source_tables)[0]
                sql += f"FROM {main_table}\n"
                
                # Add any necessary joins
                if len(source_tables) > 1:
                    # This is simplified - in a real implementation, would need to determine
                    # proper join paths between the tables
                    sql += "-- Multiple source tables detected - would need proper joins\n"
            
            return sql
        
        return f"-- Unknown metric type: {metric_type}"
    
    def _generate_joins_for_entity(self, model: Dict[str, Any], entity_name: str) -> str:
        """Generate JOIN clauses for a given entity based on model relationships"""
        if not model.get("relationships"):
            return ""
        
        joins = []
        relationships = model["relationships"]
        entities = model["entities"]
        
        # Find direct relationships to this entity
        for rel in relationships:
            if rel["from_entity"] == entity_name:
                to_entity = entities.get(rel["to_entity"], {})
                to_table = to_entity.get("source", "")
                
                if to_table:
                    join_type = "LEFT JOIN"
                    if rel["type"] == "One-to-One":
                        join_type = "INNER JOIN"
                    
                    join = f"{join_type} {to_table} ON {entities[entity_name]['source']}.{rel['from_field']} = {to_table}.{rel['to_field']}"
                    joins.append(join)
            
            elif rel["to_entity"] == entity_name:
                from_entity = entities.get(rel["from_entity"], {})
                from_table = from_entity.get("source", "")
                
                if from_table:
                    join_type = "LEFT JOIN"
                    if rel["type"] == "One-to-One":
                        join_type = "INNER JOIN"
                    
                    join = f"{join_type} {from_table} ON {entities[entity_name]['source']}.{rel['to_field']} = {from_table}.{rel['from_field']}"
                    joins.append(join)
        
        return "\n".join(joins)
    
    def _import_export_ui(self):
        """UI for importing and exporting semantic models"""
        st.subheader("Import/Export Semantic Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Export Model")
            
            if not st.session_state.semantic_models:
                st.info("No models available to export.")
            else:
                # Select model to export
                model_options = list(st.session_state.semantic_models.keys())
                export_model = st.selectbox(
                    "Select Model to Export",
                    options=model_options,
                    key="export_model_selector"
                )
                
                # Export format
                export_format = st.radio(
                    "Export Format",
                    options=["JSON", "YAML"],
                    key="export_format"
                )
                
                # Include metrics
                include_metrics = st.checkbox(
                    "Include Metrics",
                    value=True,
                    key="include_metrics_checkbox"
                )
                
                # Export button
                if st.button("Export Model", key="export_model_btn"):
                    # Prepare export data
                    export_data = {
                        "format_version": "1.0",
                        "model": st.session_state.semantic_models[export_model].copy()
                    }
                    
                    if include_metrics and export_model in st.session_state.semantic_metrics:
                        export_data["metrics"] = st.session_state.semantic_metrics[export_model].copy()
                    
                    # Convert to selected format
                    if export_format == "JSON":
                        import json
                        export_str = json.dumps(export_data, indent=2)
                        file_ext = "json"
                        mime_type = "application/json"
                    else:  # YAML
                        import yaml
                        export_str = yaml.dump(export_data, sort_keys=False)
                        file_ext = "yaml"
                        mime_type = "text/yaml"
                    
                    # Offer download
                    st.download_button(
                        label="Download Model",
                        data=export_str,
                        file_name=f"{export_model}_model.{file_ext}",
                        mime=mime_type,
                        key="download_model_btn"
                    )
        
        with col2:
            st.subheader("Import Model")
            
            # Upload file
            uploaded_file = st.file_uploader(
                "Upload Model File",
                type=["json", "yaml", "yml"],
                key="upload_model_file"
            )
            
            if uploaded_file is not None:
                try:
                    # Determine file type
                    file_type = uploaded_file.name.split(".")[-1].lower()
                    
                    # Parse file
                    if file_type == "json":
                        import json
                        import_data = json.load(uploaded_file)
                    else:  # YAML
                        import yaml
                        import_data = yaml.safe_load(uploaded_file)
                    
                    # Validate format
                    if "format_version" not in import_data or "model" not in import_data:
                        st.error("Invalid model file format.")
                    else:
                        # Display model info
                        st.write("**Model Information:**")
                        
                        # Get entity count
                        entity_count = len(import_data["model"].get("entities", {}))
                        st.write(f"Entities: {entity_count}")
                        
                        # Get relationship count
                        relationship_count = len(import_data["model"].get("relationships", []))
                        st.write(f"Relationships: {relationship_count}")
                        
                        # Get metric count if available
                        metric_count = 0
                        if "metrics" in import_data:
                            metric_count = len(import_data["metrics"])
                        st.write(f"Metrics: {metric_count}")
                        
                        # Import options
                        new_model_name = st.text_input(
                            "Model Name",
                            value=uploaded_file.name.split(".")[0],
                            key="import_model_name"
                        )
                        
                        # Import button
                        if st.button("Import Model", key="import_model_btn") and new_model_name:
                            if new_model_name in st.session_state.semantic_models:
                                # Confirm overwrite
                                if st.session_state.get("confirm_import_overwrite") != new_model_name:
                                    st.warning(f"Model '{new_model_name}' already exists. Importing will overwrite it. Confirm?")
                                    st.session_state.confirm_import_overwrite = new_model_name
                                    return
                            
                            # Import model
                            st.session_state.semantic_models[new_model_name] = import_data["model"]
                            
                            # Import metrics if available
                            if "metrics" in import_data:
                                st.session_state.semantic_metrics[new_model_name] = import_data["metrics"]
                            
                            st.success(f"Model '{new_model_name}' imported successfully.")
                            
                            # Clear confirmation state
                            if "confirm_import_overwrite" in st.session_state:
                                del st.session_state.confirm_import_overwrite
                            
                            st.rerun()
                
                except Exception as e:
                    st.error(f"Error importing model: {str(e)}")
    
    def _generate_nl_query_sql(self, model_name: str, metrics: List[str], dimensions: List[str], nl_query: str) -> str:
        """Generate SQL for a natural language query against semantic metrics"""
        model = st.session_state.semantic_models.get(model_name, {})
        model_metrics = st.session_state.semantic_metrics.get(model_name, {})
        
        if not model or not model_metrics:
            return "-- No model or metrics defined"
        
        # Start building SQL
        select_clause = []
        from_tables = set()
        where_clauses = []
        group_by_clause = []
        order_by_clause = []
        
        # Process dimensions first
        for dim_name in dimensions:
            if dim_name in model_metrics:
                dim = model_metrics[dim_name]
                entity_name = dim.get("entity", "")
                
                # Check if entity exists in the model
                if entity_name not in model.get("entities", {}):
                    continue
                    
                entity = model["entities"][entity_name]
                source_table = entity.get("source", "")
                
                # Check if expression exists
                if not dim.get("expression"):
                    continue
                
                if source_table:
                    from_tables.add(source_table)
                    select_clause.append(f"{source_table}.{dim['expression']} AS {dim_name}")
                    group_by_clause.append(f"{source_table}.{dim['expression']}")
                    
                    # Add to ORDER BY for time-related dimensions
                    time_indicators = ["date", "time", "year", "month", "day", "week", "quarter"]
                    if any(indicator in dim_name.lower() for indicator in time_indicators):
                        order_by_clause.append(f"{source_table}.{dim['expression']}")
        
        # Process metrics
        for metric_name in metrics:
            if metric_name in model_metrics:
                metric = model_metrics[metric_name]
                entity_name = metric.get("entity", "")
                
                # Check if entity exists in the model
                if entity_name not in model.get("entities", {}):
                    continue
                    
                entity = model["entities"][entity_name]
                source_table = entity.get("source", "")
                
                # Check if expression exists
                if not metric.get("expression"):
                    continue
                
                if source_table:
                    from_tables.add(source_table)
                    
                    # Generate SQL based on metric type
                    if metric.get("type") == "measure":
                        agg = metric.get("aggregation", "SUM").upper()
                        select_clause.append(f"{agg}({source_table}.{metric['expression']}) AS {metric_name}")
                    elif metric.get("type") == "dimension":
                        select_clause.append(f"{source_table}.{metric['expression']} AS {metric_name}")
                        if f"{source_table}.{metric['expression']}" not in group_by_clause:
                            group_by_clause.append(f"{source_table}.{metric['expression']}")
                    elif metric.get("type") == "calculated":
                        # Simplified handling for calculated metrics
                        select_clause.append(f"/* Calculated: {metric['expression']} */ NULL AS {metric_name}")
        
        # Check for time-related filters in the query
        time_filters = {
            "today": "date = CURRENT_DATE",
            "yesterday": "date = CURRENT_DATE - INTERVAL '1 day'",
            "this week": "date >= date_trunc('week', CURRENT_DATE)",
            "last week": "date >= date_trunc('week', CURRENT_DATE - INTERVAL '7 days') AND date < date_trunc('week', CURRENT_DATE)",
            "this month": "date >= date_trunc('month', CURRENT_DATE)",
            "last month": "date >= date_trunc('month', CURRENT_DATE - INTERVAL '1 month') AND date < date_trunc('month', CURRENT_DATE)",
            "this year": "date >= date_trunc('year', CURRENT_DATE)",
            "last year": "date >= date_trunc('year', CURRENT_DATE - INTERVAL '1 year') AND date < date_trunc('year', CURRENT_DATE)"
        }
        
        for filter_text, filter_sql in time_filters.items():
            if filter_text in nl_query.lower():
                # Find a date column to apply the filter to
                date_column = None
                for dim_name in dimensions:
                    if dim_name in model_metrics:
                        dim = model_metrics[dim_name]
                        if any(time_word in dim_name.lower() for time_word in ["date", "time", "day"]):
                            entity_name = dim.get("entity", "")
                            
                            # Check if entity exists in the model
                            if entity_name not in model.get("entities", {}):
                                continue
                                
                            # Check if expression exists
                            if not dim.get("expression"):
                                continue
                                
                            entity = model["entities"][entity_name]
                            source_table = entity.get("source", "")
                            
                            if source_table:
                                date_column = f"{source_table}.{dim['expression']}"
                                break
                
                if date_column:
                    # Replace the generic date column in the filter SQL
                    filter_sql = filter_sql.replace("date", date_column)
                    where_clauses.append(filter_sql)
        
        # Assemble SQL query
        sql = "SELECT\n    " + ",\n    ".join(select_clause)
        
        # FROM clause
        if from_tables:
            main_table = list(from_tables)[0]
            sql += f"\nFROM {main_table}"
            
            # Add joins for other tables
            if len(from_tables) > 1:
                # Simplified join logic - in a real implementation would need proper join paths
                for table in list(from_tables)[1:]:
                    sql += f"\n-- JOIN {table} ON ... /* Would need proper join conditions */"
        
        # WHERE clause
        if where_clauses:
            sql += "\nWHERE " + " AND ".join(where_clauses)
        
        # GROUP BY clause
        if group_by_clause:
            sql += "\nGROUP BY " + ", ".join(group_by_clause)
        
        # ORDER BY clause
        if order_by_clause:
            sql += "\nORDER BY " + ", ".join(order_by_clause)
        
        return sql

