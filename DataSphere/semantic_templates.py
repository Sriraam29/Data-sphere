import streamlit as st
import pandas as pd
import json
import re
from typing import Dict, List, Any, Optional, Tuple

class SemanticTemplates:
    def __init__(self):
        """Initialize the Semantic Templates module"""
        # Initialize session state for semantic templates
        if "semantic_templates" not in st.session_state:
            st.session_state.semantic_templates = self._get_default_templates()
    
    def semantic_templates_ui(self, semantic_layer):
        """UI for semantic layer templates"""
        st.subheader("Semantic Layer Templates")
        
        tab1, tab2, tab3 = st.tabs(["Apply Templates", "Manage Templates", "Import/Export"])
        
        with tab1:
            self._apply_templates_ui(semantic_layer)
        
        with tab2:
            self._manage_templates_ui(semantic_layer)
        
        with tab3:
            self._import_export_ui(semantic_layer)
    
    def _apply_templates_ui(self, semantic_layer):
        """UI for applying templates to the semantic layer"""
        st.subheader("Apply Template to Current Model")
        
        # Check if any models exist
        if not st.session_state.semantic_models:
            st.warning("Please create a semantic model first.")
            return
        
        # Select model
        model_options = list(st.session_state.semantic_models.keys())
        selected_model = st.selectbox(
            "Select Model to Apply Template",
            options=model_options,
            key="template_model_selector"
        )
        
        # Select template
        template_options = list(st.session_state.semantic_templates.keys())
        selected_template = st.selectbox(
            "Select Template",
            options=template_options,
            key="selected_template"
        )
        
        # Display template details
        if selected_template:
            template = st.session_state.semantic_templates[selected_template]
            
            st.write(f"**Template:** {selected_template}")
            st.write(f"**Description:** {template.get('description', 'No description')}")
            st.write(f"**Domain:** {template.get('domain', 'General')}")
            st.write(f"**Entities:** {', '.join(template.get('entities', {}).keys())}")
            
            # Preview metrics
            if "metrics" in template:
                st.write(f"**Metrics:** {len(template['metrics'])} predefined metrics")
                
                with st.expander("Preview Metrics"):
                    for metric_name, metric_data in template["metrics"].items():
                        st.write(f"**{metric_name}** ({metric_data.get('type', 'measure')})")
                        st.write(f"*{metric_data.get('description', '')}*")
            
            # Application options
            st.subheader("Application Options")
            
            apply_mode = st.radio(
                "Application Mode",
                options=["Merge with existing model", "Replace existing model"],
                key="apply_mode"
            )
            
            map_entities = st.checkbox(
                "Map template entities to database tables",
                value=True,
                key="map_entities_checkbox"
            )
            
            # Entity mapping
            if map_entities:
                st.subheader("Entity Mapping")
                
                # Get database schema
                schema = st.session_state.get("db_schema", {})
                
                if not schema:
                    st.warning("No schema information available. Please make sure you are connected to a database.")
                    return
                
                # Get tables from schema
                table_options = []
                if "tables" in schema:
                    table_options = list(schema["tables"].keys())
                elif "collections" in schema:
                    table_options = list(schema["collections"].keys())
                
                if not table_options:
                    st.warning("No tables found in the database schema.")
                    return
                
                # Create mapping UI for each entity in the template
                entity_mapping = {}
                
                for entity_name in template.get("entities", {}).keys():
                    entity_mapping[entity_name] = st.selectbox(
                        f"Map '{entity_name}' to",
                        options=["-- Skip --"] + table_options,
                        key=f"map_{entity_name}"
                    )
            
            # Apply template button
            if st.button("Apply Template", key="apply_template_btn"):
                if apply_mode == "Replace existing model":
                    # Confirm before replacing
                    if st.session_state.get("confirm_replace") != selected_model:
                        st.warning(f"This will replace the existing model '{selected_model}'. Are you sure?")
                        st.session_state.confirm_replace = selected_model
                        return
                
                # Apply the template
                self._apply_template(
                    semantic_layer,
                    selected_model,
                    selected_template,
                    apply_mode,
                    entity_mapping if map_entities else {}
                )
                
                st.success(f"Template '{selected_template}' applied to model '{selected_model}'.")
                
                # Clear confirmation state
                if "confirm_replace" in st.session_state:
                    del st.session_state.confirm_replace
                
                st.rerun()
    
    def _manage_templates_ui(self, semantic_layer):
        """UI for managing semantic layer templates"""
        st.subheader("Manage Templates")
        
        # Create new template section
        with st.expander("Create New Template", expanded=False):
            # Template basic info
            template_name = st.text_input("Template Name", key="new_template_name")
            template_description = st.text_area("Description", key="new_template_description")
            template_domain = st.selectbox(
                "Domain",
                options=["General", "Sales", "Marketing", "Finance", "HR", "Operations", "E-commerce", "Healthcare", "Education", "Other"],
                key="new_template_domain"
            )
            
            # Create from existing model option
            create_from_model = st.checkbox("Create from existing model", key="create_from_model_checkbox")
            
            if create_from_model:
                # Select model to use as base
                if not st.session_state.semantic_models:
                    st.warning("No semantic models available to use as a base.")
                else:
                    model_options = list(st.session_state.semantic_models.keys())
                    base_model = st.selectbox(
                        "Select Base Model",
                        options=model_options,
                        key="base_model_selector"
                    )
                    
                    if st.button("Create Template from Model", key="create_from_model_btn") and template_name:
                        if template_name in st.session_state.semantic_templates:
                            st.error(f"Template '{template_name}' already exists.")
                        else:
                            # Create new template from model
                            model_data = st.session_state.semantic_models[base_model]
                            
                            # Extract metrics if available
                            metrics = {}
                            if base_model in st.session_state.semantic_metrics:
                                metrics = st.session_state.semantic_metrics[base_model]
                            
                            # Create template
                            st.session_state.semantic_templates[template_name] = {
                                "description": template_description,
                                "domain": template_domain,
                                "entities": model_data.get("entities", {}),
                                "relationships": model_data.get("relationships", []),
                                "metrics": metrics,
                                "created_from": base_model
                            }
                            
                            st.success(f"Template '{template_name}' created from model '{base_model}'.")
                            st.rerun()
            else:
                # Manual template creation
                if st.button("Create Empty Template", key="create_empty_template_btn") and template_name:
                    if template_name in st.session_state.semantic_templates:
                        st.error(f"Template '{template_name}' already exists.")
                    else:
                        # Create new empty template
                        st.session_state.semantic_templates[template_name] = {
                            "description": template_description,
                            "domain": template_domain,
                            "entities": {},
                            "relationships": [],
                            "metrics": {}
                        }
                        
                        st.success(f"Empty template '{template_name}' created.")
                        st.rerun()
        
        # List existing templates
        st.subheader("Existing Templates")
        
        if not st.session_state.semantic_templates:
            st.info("No templates available.")
        else:
            # Convert templates to DataFrame for display
            templates_data = []
            for template_name, template in st.session_state.semantic_templates.items():
                templates_data.append({
                    "Name": template_name,
                    "Domain": template.get("domain", "General"),
                    "Description": template.get("description", ""),
                    "Entities": len(template.get("entities", {})),
                    "Metrics": len(template.get("metrics", {}))
                })
            
            templates_df = pd.DataFrame(templates_data)
            st.dataframe(templates_df)
            
            # Template details and actions
            selected_template = st.selectbox(
                "Select Template to Manage",
                options=list(st.session_state.semantic_templates.keys()),
                key="manage_template_selector"
            )
            
            if selected_template:
                template = st.session_state.semantic_templates[selected_template]
                
                st.subheader(f"Template Details: {selected_template}")
                
                # Basic info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Domain:** {template.get('domain', 'General')}")
                    st.write(f"**Entities:** {len(template.get('entities', {}))}")
                
                with col2:
                    st.write(f"**Relationships:** {len(template.get('relationships', []))}")
                    st.write(f"**Metrics:** {len(template.get('metrics', {}))}")
                
                st.write(f"**Description:** {template.get('description', 'No description')}")
                
                # Entities
                with st.expander("Entities", expanded=False):
                    for entity_name, entity_data in template.get("entities", {}).items():
                        st.write(f"**{entity_name}**")
                        st.write(f"Source: {entity_data.get('source', 'N/A')}")
                        st.write(f"Fields: {len(entity_data.get('fields', {}))}")
                        st.write("---")
                
                # Metrics
                with st.expander("Metrics", expanded=False):
                    for metric_name, metric_data in template.get("metrics", {}).items():
                        st.write(f"**{metric_name}** ({metric_data.get('type', 'measure')})")
                        st.write(f"*{metric_data.get('description', '')}*")
                        st.write(f"Entity: {metric_data.get('entity', 'N/A')}")
                        st.write(f"Expression: {metric_data.get('expression', 'N/A')}")
                        st.write("---")
                
                # Actions
                st.subheader("Actions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Edit template
                    if st.button("Edit Template", key=f"edit_{selected_template}"):
                        st.session_state.edit_template = selected_template
                        st.rerun()
                
                with col2:
                    # Delete template
                    if st.button("Delete Template", key=f"delete_{selected_template}"):
                        # Confirm before deleting
                        if st.session_state.get("confirm_delete") != selected_template:
                            st.warning(f"Are you sure you want to delete the template '{selected_template}'?")
                            st.session_state.confirm_delete = selected_template
                            return
                        
                        del st.session_state.semantic_templates[selected_template]
                        st.success(f"Template '{selected_template}' deleted.")
                        
                        # Clear confirmation state
                        del st.session_state.confirm_delete
                        
                        st.rerun()
                
                # Edit template form
                if hasattr(st.session_state, "edit_template") and st.session_state.edit_template == selected_template:
                    st.subheader(f"Edit Template: {selected_template}")
                    
                    # Basic info
                    template_description = st.text_area(
                        "Description",
                        value=template.get("description", ""),
                        key="edit_template_description"
                    )
                    
                    template_domain = st.selectbox(
                        "Domain",
                        options=["General", "Sales", "Marketing", "Finance", "HR", "Operations", "E-commerce", "Healthcare", "Education", "Other"],
                        index=["General", "Sales", "Marketing", "Finance", "HR", "Operations", "E-commerce", "Healthcare", "Education", "Other"].index(template.get("domain", "General")),
                        key="edit_template_domain"
                    )
                    
                    # Save changes
                    if st.button("Save Changes", key="save_template_changes_btn"):
                        # Update template
                        template["description"] = template_description
                        template["domain"] = template_domain
                        
                        st.success(f"Template '{selected_template}' updated.")
                        
                        # Clear edit state
                        del st.session_state.edit_template
                        
                        st.rerun()
                    
                    if st.button("Cancel Edit", key="cancel_template_edit_btn"):
                        del st.session_state.edit_template
                        st.rerun()
    
    def _import_export_ui(self, semantic_layer):
        """UI for importing and exporting semantic layer templates"""
        st.subheader("Import/Export Templates")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Export Templates")
            
            # Select templates to export
            template_options = list(st.session_state.semantic_templates.keys())
            export_templates = st.multiselect(
                "Select Templates to Export",
                options=template_options,
                key="export_templates_selector"
            )
            
            if export_templates:
                # Export selected templates
                export_data = {
                    "format_version": "1.0",
                    "templates": {}
                }
                
                for template_name in export_templates:
                    export_data["templates"][template_name] = st.session_state.semantic_templates[template_name]
                
                # Convert to JSON
                export_json = json.dumps(export_data, indent=2)
                
                # Offer download
                st.download_button(
                    label="Download Templates",
                    data=export_json,
                    file_name="semantic_templates.json",
                    mime="application/json",
                    key="download_templates_btn"
                )
        
        with col2:
            st.subheader("Import Templates")
            
            # Upload JSON file
            uploaded_file = st.file_uploader(
                "Upload Templates JSON",
                type=["json"],
                key="upload_templates_file"
            )
            
            if uploaded_file is not None:
                try:
                    # Read and parse JSON
                    import_data = json.load(uploaded_file)
                    
                    # Validate format
                    if "format_version" not in import_data or "templates" not in import_data:
                        st.error("Invalid template file format.")
                    else:
                        # Display templates to import
                        st.write(f"Found {len(import_data['templates'])} templates:")
                        
                        for template_name in import_data["templates"].keys():
                            st.write(f"- {template_name}")
                        
                        # Import options
                        import_mode = st.radio(
                            "Import Mode",
                            options=["Skip existing", "Overwrite existing"],
                            key="import_mode"
                        )
                        
                        # Import button
                        if st.button("Import Templates", key="import_templates_btn"):
                            # Process import
                            imported = 0
                            skipped = 0
                            
                            for template_name, template_data in import_data["templates"].items():
                                if template_name in st.session_state.semantic_templates and import_mode == "Skip existing":
                                    skipped += 1
                                else:
                                    st.session_state.semantic_templates[template_name] = template_data
                                    imported += 1
                            
                            st.success(f"Imported {imported} templates. Skipped {skipped} existing templates.")
                            st.rerun()
                
                except Exception as e:
                    st.error(f"Error importing templates: {str(e)}")
    
    def _apply_template(self, semantic_layer, model_name, template_name, apply_mode, entity_mapping):
        """Apply a template to a semantic model"""
        template = st.session_state.semantic_templates[template_name]
        model = st.session_state.semantic_models[model_name]
        
        if apply_mode == "Replace existing model":
            # Create a new model based on the template
            new_model = {
                "description": model.get("description", ""),
                "entities": {},
                "relationships": []
            }
            
            # Copy entities from template with mapping
            for entity_name, entity_data in template.get("entities", {}).items():
                if entity_name in entity_mapping and entity_mapping[entity_name] != "-- Skip --":
                    # Map to database table
                    mapped_table = entity_mapping[entity_name]
                    
                    # Get schema for the mapped table
                    schema = st.session_state.get("db_schema", {})
                    
                    if "tables" in schema and mapped_table in schema["tables"]:
                        # Create entity from table schema
                        entity_fields = {}
                        for col in schema["tables"][mapped_table]["columns"]:
                            field_name = col["name"]
                            entity_fields[field_name] = {
                                "source": field_name,
                                "display_name": field_name.replace("_", " ").title(),
                                "visible": True
                            }
                        
                        new_model["entities"][entity_name] = {
                            "source": mapped_table,
                            "fields": entity_fields,
                            "display_name": entity_data.get("display_name", entity_name),
                            "description": entity_data.get("description", "")
                        }
                    elif "collections" in schema and mapped_table in schema["collections"]:
                        # Create entity from collection schema
                        entity_fields = {}
                        for field in schema["collections"][mapped_table]["fields"]:
                            field_name = field["name"]
                            entity_fields[field_name] = {
                                "source": field_name,
                                "display_name": field_name.replace("_", " ").title(),
                                "visible": True
                            }
                        
                        new_model["entities"][entity_name] = {
                            "source": mapped_table,
                            "fields": entity_fields,
                            "display_name": entity_data.get("display_name", entity_name),
                            "description": entity_data.get("description", "")
                        }
                else:
                    # Copy entity as is
                    new_model["entities"][entity_name] = entity_data.copy()
            
            # Copy relationships
            for rel in template.get("relationships", []):
                # Only copy relationships where both entities exist in the new model
                if rel["from_entity"] in new_model["entities"] and rel["to_entity"] in new_model["entities"]:
                    new_model["relationships"].append(rel.copy())
            
            # Replace the model
            st.session_state.semantic_models[model_name] = new_model
            
            # Apply metrics if available
            if "metrics" in template and template["metrics"]:
                if model_name not in st.session_state.semantic_metrics:
                    st.session_state.semantic_metrics[model_name] = {}
                
                # Replace or merge metrics
                if apply_mode == "Replace existing model":
                    st.session_state.semantic_metrics[model_name] = template["metrics"].copy()
                else:
                    # Merge metrics
                    for metric_name, metric_data in template["metrics"].items():
                        if metric_name not in st.session_state.semantic_metrics[model_name]:
                            st.session_state.semantic_metrics[model_name][metric_name] = metric_data.copy()
        
        else:  # Merge with existing model
            # Merge entities
            for entity_name, entity_data in template.get("entities", {}).items():
                if entity_name in entity_mapping and entity_mapping[entity_name] != "-- Skip --":
                    # Map to database table
                    mapped_table = entity_mapping[entity_name]
                    
                    # Get schema for the mapped table
                    schema = st.session_state.get("db_schema", {})
                    
                    if "tables" in schema and mapped_table in schema["tables"]:
                        # Create entity from table schema
                        entity_fields = {}
                        for col in schema["tables"][mapped_table]["columns"]:
                            field_name = col["name"]
                            entity_fields[field_name] = {
                                "source": field_name,
                                "display_name": field_name.replace("_", " ").title(),
                                "visible": True
                            }
                        
                        if entity_name not in model["entities"]:
                            model["entities"][entity_name] = {
                                "source": mapped_table,
                                "fields": entity_fields,
                                "display_name": entity_data.get("display_name", entity_name),
                                "description": entity_data.get("description", "")
                            }
                    elif "collections" in schema and mapped_table in schema["collections"]:
                        # Create entity from collection schema
                        entity_fields = {}
                        for field in schema["collections"][mapped_table]["fields"]:
                            field_name = field["name"]
                            entity_fields[field_name] = {
                                "source": field_name,
                                "display_name": field_name.replace("_", " ").title(),
                                "visible": True
                            }
                        
                        if entity_name not in model["entities"]:
                            model["entities"][entity_name] = {
                                "source": mapped_table,
                                "fields": entity_fields,
                                "display_name": entity_data.get("display_name", entity_name),
                                "description": entity_data.get("description", "")
                            }
                elif entity_name not in model["entities"]:
                    # Add entity if it doesn't exist
                    model["entities"][entity_name] = entity_data.copy()
            
            # Merge relationships
            existing_rels = set()
            for rel in model["relationships"]:
                rel_key = f"{rel['from_entity']}.{rel['from_field']}-{rel['to_entity']}.{rel['to_field']}"
                existing_rels.add(rel_key)
            
            for rel in template.get("relationships", []):
                # Only add relationships where both entities exist in the model
                if rel["from_entity"] in model["entities"] and rel["to_entity"] in model["entities"]:
                    rel_key = f"{rel['from_entity']}.{rel['from_field']}-{rel['to_entity']}.{rel['to_field']}"
                    
                    if rel_key not in existing_rels:
                        model["relationships"].append(rel.copy())
                        existing_rels.add(rel_key)
            
            # Merge metrics if available
            if "metrics" in template and template["metrics"]:
                if model_name not in st.session_state.semantic_metrics:
                    st.session_state.semantic_metrics[model_name] = {}
                
                # Add metrics that don't exist
                for metric_name, metric_data in template["metrics"].items():
                    if metric_name not in st.session_state.semantic_metrics[model_name]:
                        # Check if the metric's entity exists in the model
                        entity = metric_data.get("entity", "")
                        if not entity or entity in model["entities"]:
                            st.session_state.semantic_metrics[model_name][metric_name] = metric_data.copy()
    
    def _get_default_templates(self) -> Dict:
        """Get default semantic layer templates"""
        return {
            "Sales Analytics": {
                "description": "Standard sales analytics model with customers, products, and orders",
                "domain": "Sales",
                "entities": {
                    "Customers": {
                        "display_name": "Customers",
                        "description": "Customer information",
                        "fields": {
                            "customer_id": {"source": "customer_id", "display_name": "Customer ID", "visible": True},
                            "customer_name": {"source": "customer_name", "display_name": "Customer Name", "visible": True},
                            "email": {"source": "email", "display_name": "Email", "visible": True},
                            "phone": {"source": "phone", "display_name": "Phone", "visible": True},
                            "address": {"source": "address", "display_name": "Address", "visible": True},
                            "city": {"source": "city", "display_name": "City", "visible": True},
                            "state": {"source": "state", "display_name": "State", "visible": True},
                            "country": {"source": "country", "display_name": "Country", "visible": True},
                            "postal_code": {"source": "postal_code", "display_name": "Postal Code", "visible": True},
                            "created_at": {"source": "created_at", "display_name": "Created At", "visible": True}
                        }
                    },
                    "Products": {
                        "display_name": "Products",
                        "description": "Product catalog",
                        "fields": {
                            "product_id": {"source": "product_id", "display_name": "Product ID", "visible": True},
                            "product_name": {"source": "product_name", "display_name": "Product Name", "visible": True},
                            "description": {"source": "description", "display_name": "Description", "visible": True},
                            "category": {"source": "category", "display_name": "Category", "visible": True},
                            "price": {"source": "price", "display_name": "Price", "visible": True},
                            "cost": {"source": "cost", "display_name": "Cost", "visible": True},
                            "sku": {"source": "sku", "display_name": "SKU", "visible": True},
                            "created_at": {"source": "created_at", "display_name": "Created At", "visible": True}
                        }
                    },
                    "Orders": {
                        "display_name": "Orders",
                        "description": "Customer orders",
                        "fields": {
                            "order_id": {"source": "order_id", "display_name": "Order ID", "visible": True},
                            "customer_id": {"source": "customer_id", "display_name": "Customer ID", "visible": True},
                            "order_date": {"source": "order_date", "display_name": "Order Date", "visible": True},
                            "status": {"source": "status", "display_name": "Status", "visible": True},
                            "total_amount": {"source": "total_amount", "display_name": "Total Amount", "visible": True}
                        }
                    },
                    "Order_Items": {
                        "display_name": "Order Items",
                        "description": "Items within orders",
                        "fields": {
                            "order_item_id": {"source": "order_item_id", "display_name": "Order Item ID", "visible": True},
                            "order_id": {"source": "order_id", "display_name": "Order ID", "visible": True},
                            "product_id": {"source": "product_id", "display_name": "Product ID", "visible": True},
                            "quantity": {"source": "quantity", "display_name": "Quantity", "visible": True},
                            "price": {"source": "price", "display_name": "Price", "visible": True},
                            "discount": {"source": "discount", "display_name": "Discount", "visible": True},
                            "total": {"source": "total", "display_name": "Total", "visible": True}
                        }
                    }
                },
                "relationships": [
                    {
                        "from_entity": "Orders",
                        "to_entity": "Customers",
                        "from_field": "customer_id",
                        "to_field": "customer_id",
                        "type": "Many-to-One",
                        "description": "Orders belong to customers"
                    },
                    {
                        "from_entity": "Order_Items",
                        "to_entity": "Orders",
                        "from_field": "order_id",
                        "to_field": "order_id",
                        "type": "Many-to-One",
                        "description": "Order items belong to orders"
                    },
                    {
                        "from_entity": "Order_Items",
                        "to_entity": "Products",
                        "from_field": "product_id",
                        "to_field": "product_id",
                        "type": "Many-to-One",
                        "description": "Order items reference products"
                    }
                ],
                "metrics": {
                    "Total Revenue": {
                        "name": "Total Revenue",
                        "description": "Sum of all order totals",
                        "type": "measure",
                        "format": "currency",
                        "entity": "Orders",
                        "expression": "total_amount",
                        "aggregation": "SUM"
                    },
                    "Order Count": {
                        "name": "Order Count",
                        "description": "Number of orders",
                        "type": "measure",
                        "format": "number",
                        "entity": "Orders",
                        "expression": "order_id",
                        "aggregation": "COUNT"
                    },
                    "Average Order Value": {
                        "name": "Average Order Value",
                        "description": "Average amount per order",
                        "type": "measure",
                        "format": "currency",
                        "entity": "Orders",
                        "expression": "total_amount",
                        "aggregation": "AVG"
                    },
                    "Product Category": {
                        "name": "Product Category",
                        "description": "Product category dimension",
                        "type": "dimension",
                        "format": "text",
                        "entity": "Products",
                        "expression": "category"
                    },
                    "Order Date": {
                        "name": "Order Date",
                        "description": "Date of order",
                        "type": "dimension",
                        "format": "date",
                        "entity": "Orders",
                        "expression": "order_date"
                    },
                    "Customer Location": {
                        "name": "Customer Location",
                        "description": "Customer's country",
                        "type": "dimension",
                        "format": "text",
                        "entity": "Customers",
                        "expression": "country"
                    },
                    "Profit": {
                        "name": "Profit",
                        "description": "Revenue minus cost",
                        "type": "calculated",
                        "format": "currency",
                        "entity": "",
                        "expression": "[Total Revenue] - SUM(Order_Items.quantity * Products.cost)"
                    }
                }
            },
            "Marketing Analytics": {
                "description": "Marketing analytics model with campaigns, channels, and conversions",
                "domain": "Marketing",
                "entities": {
                    "Campaigns": {
                        "display_name": "Campaigns",
                        "description": "Marketing campaigns",
                        "fields": {
                            "campaign_id": {"source": "campaign_id", "display_name": "Campaign ID", "visible": True},
                            "campaign_name": {"source": "campaign_name", "display_name": "Campaign Name", "visible": True},
                            "description": {"source": "description", "display_name": "Description", "visible": True},
                            "start_date": {"source": "start_date", "display_name": "Start Date", "visible": True},
                            "end_date": {"source": "end_date", "display_name": "End Date", "visible": True},
                            "budget": {"source": "budget", "display_name": "Budget", "visible": True},
                            "status": {"source": "status", "display_name": "Status", "visible": True}
                        }
                    },
                    "Channels": {
                        "display_name": "Channels",
                        "description": "Marketing channels",
                        "fields": {
                            "channel_id": {"source": "channel_id", "display_name": "Channel ID", "visible": True},
                            "channel_name": {"source": "channel_name", "display_name": "Channel Name", "visible": True},
                            "type": {"source": "type", "display_name": "Type", "visible": True},
                            "description": {"source": "description", "display_name": "Description", "visible": True}
                        }
                    },
                    "Ads": {
                        "display_name": "Ads",
                        "description": "Individual advertisements",
                        "fields": {
                            "ad_id": {"source": "ad_id", "display_name": "Ad ID", "visible": True},
                            "campaign_id": {"source": "campaign_id", "display_name": "Campaign ID", "visible": True},
                            "channel_id": {"source": "channel_id", "display_name": "Channel ID", "visible": True},
                            "ad_name": {"source": "ad_name", "display_name": "Ad Name", "visible": True},
                            "content": {"source": "content", "display_name": "Content", "visible": True},
                            "start_date": {"source": "start_date", "display_name": "Start Date", "visible": True},
                            "end_date": {"source": "end_date", "display_name": "End Date", "visible": True},
                            "cost": {"source": "cost", "display_name": "Cost", "visible": True}
                        }
                    },
                    "Conversions": {
                        "display_name": "Conversions",
                        "description": "Marketing conversions",
                        "fields": {
                            "conversion_id": {"source": "conversion_id", "display_name": "Conversion ID", "visible": True},
                            "ad_id": {"source": "ad_id", "display_name": "Ad ID", "visible": True},
                            "user_id": {"source": "user_id", "display_name": "User ID", "visible": True},
                            "conversion_date": {"source": "conversion_date", "display_name": "Conversion Date", "visible": True},
                            "conversion_value": {"source": "conversion_value", "display_name": "Conversion Value", "visible": True},
                            "conversion_type": {"source": "conversion_type", "display_name": "Conversion Type", "visible": True}
                        }
                    }
                },
                "relationships": [
                    {
                        "from_entity": "Ads",
                        "to_entity": "Campaigns",
                        "from_field": "campaign_id",
                        "to_field": "campaign_id",
                        "type": "Many-to-One",
                        "description": "Ads belong to campaigns"
                    },
                    {
                        "from_entity": "Ads",
                        "to_entity": "Channels",
                        "from_field": "channel_id",
                        "to_field": "channel_id",
                        "type": "Many-to-One",
                        "description": "Ads are delivered through channels"
                    },
                    {
                        "from_entity": "Conversions",
                        "to_entity": "Ads",
                        "from_field": "ad_id",
                        "to_field": "ad_id",
                        "type": "Many-to-One",
                        "description": "Conversions come from ads"
                    }
                ],
                "metrics": {
                    "Total Ad Spend": {
                        "name": "Total Ad Spend",
                        "description": "Total amount spent on ads",
                        "type": "measure",
                        "format": "currency",
                        "entity": "Ads",
                        "expression": "cost",
                        "aggregation": "SUM"
                    },
                    "Conversion Count": {
                        "name": "Conversion Count",
                        "description": "Number of conversions",
                        "type": "measure",
                        "format": "number",
                        "entity": "Conversions",
                        "expression": "conversion_id",
                        "aggregation": "COUNT"
                    },
                    "Conversion Value": {
                        "name": "Conversion Value",
                        "description": "Total value of conversions",
                        "type": "measure",
                        "format": "currency",
                        "entity": "Conversions",
                        "expression": "conversion_value",
                        "aggregation": "SUM"
                    },
                    "Campaign Name": {
                        "name": "Campaign Name",
                        "description": "Name of the campaign",
                        "type": "dimension",
                        "format": "text",
                        "entity": "Campaigns",
                        "expression": "campaign_name"
                    },
                    "Channel Type": {
                        "name": "Channel Type",
                        "description": "Type of marketing channel",
                        "type": "dimension",
                        "format": "text",
                        "entity": "Channels",
                        "expression": "type"
                    },
                    "Conversion Date": {
                        "name": "Conversion Date",
                        "description": "Date of conversion",
                        "type": "dimension",
                        "format": "date",
                        "entity": "Conversions",
                        "expression": "conversion_date"
                    },
                    "ROI": {
                        "name": "ROI",
                        "description": "Return on investment",
                        "type": "calculated",
                        "format": "percentage",
                        "entity": "",
                        "expression": "([Conversion Value] - [Total Ad Spend]) / [Total Ad Spend]"
                    },
                    "Cost per Conversion": {
                        "name": "Cost per Conversion",
                        "description": "Average cost per conversion",
                        "type": "calculated",
                        "format": "currency",
                        "entity": "",
                        "expression": "[Total Ad Spend] / [Conversion Count]"
                    }
                }
            },
            "Financial Analytics": {
                "description": "Financial analytics model with accounts, transactions, and budgets",
                "domain": "Finance",
                "entities": {
                    "Accounts": {
                        "display_name": "Accounts",
                        "description": "Financial accounts",
                        "fields": {
                            "account_id": {"source": "account_id", "display_name": "Account ID", "visible": True},
                            "account_name": {"source": "account_name", "display_name": "Account Name", "visible": True},
                            "account_type": {"source": "account_type", "display_name": "Account Type", "visible": True},
                            "balance": {"source": "balance", "display_name": "Balance", "visible": True},
                            "currency": {"source": "currency", "display_name": "Currency", "visible": True},
                            "created_at": {"source": "created_at", "display_name": "Created At", "visible": True}
                        }
                    },
                    "Transactions": {
                        "display_name": "Transactions",
                        "description": "Financial transactions",
                        "fields": {
                            "transaction_id": {"source": "transaction_id", "display_name": "Transaction ID", "visible": True},
                            "account_id": {"source": "account_id", "display_name": "Account ID", "visible": True},
                            "transaction_date": {"source": "transaction_date", "display_name": "Transaction Date", "visible": True},
                            "amount": {"source": "amount", "display_name": "Amount", "visible": True},
                            "type": {"source": "type", "display_name": "Type", "visible": True},
                            "category": {"source": "category", "display_name": "Category", "visible": True},
                            "description": {"source": "description", "display_name": "Description", "visible": True}
                        }
                    },
                    "Categories": {
                        "display_name": "Categories",
                        "description": "Transaction categories",
                        "fields": {
                            "category_id": {"source": "category_id", "display_name": "Category ID", "visible": True},
                            "category_name": {"source": "category_name", "display_name": "Category Name", "visible": True},
                            "parent_category_id": {"source": "parent_category_id", "display_name": "Parent Category ID", "visible": True},
                            "type": {"source": "type", "display_name": "Type", "visible": True}
                        }
                    },
                    "Budgets": {
                        "display_name": "Budgets",
                        "description": "Financial budgets",
                        "fields": {
                            "budget_id": {"source": "budget_id", "display_name": "Budget ID", "visible": True},
                            "category_id": {"source": "category_id", "display_name": "Category ID", "visible": True},
                            "amount": {"source": "amount", "display_name": "Amount", "visible": True},
                            "start_date": {"source": "start_date", "display_name": "Start Date", "visible": True},
                            "end_date": {"source": "end_date", "display_name": "End Date", "visible": True},
                            "name": {"source": "name", "display_name": "Name", "visible": True}
                        }
                    }
                },
                "relationships": [
                    {
                        "from_entity": "Transactions",
                        "to_entity": "Accounts",
                        "from_field": "account_id",
                        "to_field": "account_id",
                        "type": "Many-to-One",
                        "description": "Transactions belong to accounts"
                    },
                    {
                        "from_entity": "Transactions",
                        "to_entity": "Categories",
                        "from_field": "category",
                        "to_field": "category_id",
                        "type": "Many-to-One",
                        "description": "Transactions have categories"
                    },
                    {
                        "from_entity": "Budgets",
                        "to_entity": "Categories",
                        "from_field": "category_id",
                        "to_field": "category_id",
                        "type": "Many-to-One",
                        "description": "Budgets are for categories"
                    },
                    {
                        "from_entity": "Categories",
                        "to_entity": "Categories",
                        "from_field": "parent_category_id",
                        "to_field": "category_id",
                        "type": "Many-to-One",
                        "description": "Categories can have parent categories"
                    }
                ],
                "metrics": {
                    "Total Income": {
                        "name": "Total Income",
                        "description": "Sum of all income transactions",
                        "type": "measure",
                        "format": "currency",
                        "entity": "Transactions",
                        "expression": "amount",
                        "aggregation": "SUM",
                        "filter": "type = 'income'"
                    },
                    "Total Expenses": {
                        "name": "Total Expenses",
                        "description": "Sum of all expense transactions",
                        "type": "measure",
                        "format": "currency",
                        "entity": "Transactions",
                        "expression": "amount",
                        "aggregation": "SUM",
                        "filter": "type = 'expense'"
                    },
                    "Transaction Count": {
                        "name": "Transaction Count",
                        "description": "Number of transactions",
                        "type": "measure",
                        "format": "number",
                        "entity": "Transactions",
                        "expression": "transaction_id",
                        "aggregation": "COUNT"
                    },
                    "Account Type": {
                        "name": "Account Type",
                        "description": "Type of account",
                        "type": "dimension",
                        "format": "text",
                        "entity": "Accounts",
                        "expression": "account_type"
                    },
                    "Transaction Date": {
                        "name": "Transaction Date",
                        "description": "Date of transaction",
                        "type": "dimension",
                        "format": "date",
                        "entity": "Transactions",
                        "expression": "transaction_date"
                    },
                    "Category Name": {
                        "name": "Category Name",
                        "description": "Name of category",
                        "type": "dimension",
                        "format": "text",
                        "entity": "Categories",
                        "expression": "category_name"
                    },
                    "Net Income": {
                        "name": "Net Income",
                        "description": "Income minus expenses",
                        "type": "calculated",
                        "format": "currency",
                        "entity": "",
                        "expression": "[Total Income] - [Total Expenses]"
                    },
                    "Budget Variance": {
                        "name": "Budget Variance",
                        "description": "Difference between budget and actual spending",
                        "type": "calculated",
                        "format": "currency",
                        "entity": "",
                        "expression": "SUM(Budgets.amount) - [Total Expenses]"
                    }
                }
            }
        }