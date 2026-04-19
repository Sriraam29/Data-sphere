import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import json
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime, timedelta
import uuid

class AdvancedVisualization:
    def __init__(self):
        """Initialize the Advanced Visualization module"""
        # Initialize session state for dashboards
        if "dashboards" not in st.session_state:
            st.session_state.dashboards = {}
        
        if "current_dashboard" not in st.session_state:
            st.session_state.current_dashboard = None
        
        if "dashboard_filters" not in st.session_state:
            st.session_state.dashboard_filters = {}
    
    def visualization_ui(self, db_manager):
        """UI for advanced visualization dashboard"""
        st.subheader("Advanced Visualization Dashboard")
        
        # Check if connected to a database
        if not st.session_state.get("connected_db"):
            st.warning("Please connect to a database first.")
            return
        
        tab1, tab2, tab3 = st.tabs(["Dashboard Builder", "Data Explorer", "Forecasting"])
        
        with tab1:
            self._dashboard_builder_ui(db_manager)
        
        with tab2:
            self._data_explorer_ui(db_manager)
        
        with tab3:
            self._forecasting_ui(db_manager)
    
    def _dashboard_builder_ui(self, db_manager):
        """UI for building custom dashboards"""
        st.subheader("Dashboard Builder")
        
        # Check if connected to a database and has schema
        if not st.session_state.get("connected_db"):
            st.warning("Please connect to a database first.")
            return
            
        # Get database schema if not already available
        schema = st.session_state.get("db_schema", {})
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
        
        # Dashboard management in sidebar
        with st.sidebar:
            st.subheader("Dashboards")
            
            # Create new dashboard
            new_dashboard_name = st.text_input("New Dashboard Name")
            if st.button("Create Dashboard") and new_dashboard_name:
                if new_dashboard_name in st.session_state.dashboards:
                    st.error(f"Dashboard '{new_dashboard_name}' already exists")
                else:
                    dashboard_id = str(uuid.uuid4())
                    st.session_state.dashboards[new_dashboard_name] = {
                        "id": dashboard_id,
                        "name": new_dashboard_name,
                        "description": "",
                        "charts": [],
                        "layout": "vertical",  # or "grid"
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "filters": []
                    }
                    st.session_state.current_dashboard = new_dashboard_name
                    st.success(f"Created dashboard: {new_dashboard_name}")
                    st.rerun()
            
            # Select existing dashboard
            if st.session_state.dashboards:
                dashboard_options = list(st.session_state.dashboards.keys())
                selected_dashboard = st.selectbox(
                    "Select Dashboard",
                    options=dashboard_options,
                    index=dashboard_options.index(st.session_state.current_dashboard) if st.session_state.current_dashboard in dashboard_options else 0
                )
                st.session_state.current_dashboard = selected_dashboard
            else:
                st.info("No dashboards created yet")
                return
        
        # Main dashboard editing area
        current_dashboard = st.session_state.current_dashboard
        dashboard_data = st.session_state.dashboards[current_dashboard]
        
        # Dashboard settings
        st.subheader(f"Edit Dashboard: {current_dashboard}")
        
        # Dashboard description
        dashboard_description = st.text_area(
            "Dashboard Description",
            value=dashboard_data.get("description", ""),
            height=100
        )
        dashboard_data["description"] = dashboard_description
        
        # Dashboard layout
        layout_options = ["Vertical", "Grid"]
        selected_layout = st.selectbox(
            "Dashboard Layout",
            options=layout_options,
            index=layout_options.index("Vertical" if dashboard_data.get("layout") == "vertical" else "Grid")
        )
        dashboard_data["layout"] = selected_layout.lower()
        
        # Dashboard filters
        with st.expander("Dashboard Filters"):
            st.info("Dashboard filters apply to all charts in the dashboard")
            
            # Add new filter
            filter_type = st.selectbox(
                "Filter Type",
                options=["Date Range", "Dropdown", "Multi-select", "Slider"]
            )
            
            filter_name = st.text_input("Filter Name", key="new_filter_name")
            filter_column = st.text_input("Database Column", key="new_filter_column")
            
            if st.button("Add Filter"):
                if filter_name and filter_column:
                    # Add to dashboard filters
                    dashboard_data["filters"].append({
                        "name": filter_name,
                        "type": filter_type.lower().replace(" ", "_"),
                        "column": filter_column,
                        "default_value": None
                    })
                    st.success(f"Added {filter_type} filter")
                    st.rerun()
            
            # Show existing filters
            if dashboard_data["filters"]:
                st.subheader("Existing Filters")
                
                for i, filter_data in enumerate(dashboard_data["filters"]):
                    st.write(f"**{filter_data['name']}** ({filter_data['type']}) - Column: {filter_data['column']}")
                    
                    if st.button("Remove", key=f"remove_filter_{i}"):
                        dashboard_data["filters"].pop(i)
                        st.success("Filter removed")
                        st.rerun()
        
        # Add new chart to dashboard
        st.subheader("Add Chart to Dashboard")
        
        chart_types = ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Table", "Card", "Heatmap"]
        chart_type = st.selectbox("Chart Type", options=chart_types)
        
        chart_title = st.text_input("Chart Title")
        
        # SQL query for chart data
        chart_query = st.text_area(
            "SQL Query",
            height=150,
            help="Enter SQL query to fetch chart data. Use :filter_name for filter placeholders."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            chart_width = st.slider("Chart Width", 1, 12, 6, help="Width from 1 to 12 columns (12 is full width)")
        
        with col2:
            chart_height = st.slider("Chart Height", 100, 800, 300, help="Height in pixels")
        
        if st.button("Add Chart to Dashboard") and chart_title and chart_query:
            # Add chart to dashboard
            dashboard_data["charts"].append({
                "id": str(uuid.uuid4()),
                "type": chart_type.lower().replace(" ", "_"),
                "title": chart_title,
                "query": chart_query,
                "width": chart_width,
                "height": chart_height,
                "options": {}
            })
            
            st.success("Chart added to dashboard")
            st.rerun()
        
        # Preview dashboard
        if dashboard_data["charts"]:
            st.subheader("Dashboard Preview")
            
            # Display dashboard filters
            if dashboard_data["filters"]:
                st.subheader("Filters")
                
                cols = st.columns(min(len(dashboard_data["filters"]), 3))
                
                for i, filter_data in enumerate(dashboard_data["filters"]):
                    col_idx = i % len(cols)
                    
                    with cols[col_idx]:
                        filter_key = f"dash_{dashboard_data['id']}_{filter_data['name']}"
                        
                        if filter_data["type"] == "date_range":
                            start_date = st.date_input(
                                f"{filter_data['name']} - Start",
                                value=(datetime.now() - timedelta(days=30)).date(),
                                key=f"{filter_key}_start"
                            )
                            end_date = st.date_input(
                                f"{filter_data['name']} - End",
                                value=datetime.now().date(),
                                key=f"{filter_key}_end"
                            )
                            st.session_state.dashboard_filters[filter_key] = (start_date, end_date)
                        
                        elif filter_data["type"] == "dropdown":
                            # In a real implementation, would fetch options from database
                            options = ["Option 1", "Option 2", "Option 3"]
                            selected = st.selectbox(
                                filter_data["name"],
                                options=options,
                                key=filter_key
                            )
                            st.session_state.dashboard_filters[filter_key] = selected
                        
                        elif filter_data["type"] == "multi_select":
                            # In a real implementation, would fetch options from database
                            options = ["Option 1", "Option 2", "Option 3"]
                            selected = st.multiselect(
                                filter_data["name"],
                                options=options,
                                key=filter_key
                            )
                            st.session_state.dashboard_filters[filter_key] = selected
                        
                        elif filter_data["type"] == "slider":
                            # In a real implementation, would determine range from data
                            value = st.slider(
                                filter_data["name"],
                                min_value=0,
                                max_value=100,
                                value=50,
                                key=filter_key
                            )
                            st.session_state.dashboard_filters[filter_key] = value
            
            # Display dashboard charts
            if dashboard_data["layout"] == "vertical":
                # Vertical layout
                for i, chart in enumerate(dashboard_data["charts"]):
                    with st.container():
                        st.subheader(chart["title"])
                        
                        # Execute query to get chart data
                        if not chart["query"]:
                            st.info("No query defined for this chart")
                            continue
                        
                        try:
                            # Apply dashboard filters to query
                            query = chart["query"]
                            
                            # In a real implementation, would replace filter placeholders in query
                            # Here we'll just execute the query as is for demonstration
                            result = db_manager.execute_query(query)
                            
                            if isinstance(result, pd.DataFrame) and not result.empty:
                                # Display chart based on type
                                chart_height = chart.get("height", 300)
                                
                                if chart["type"] == "bar_chart":
                                    if len(result.columns) >= 2:
                                        st.bar_chart(result.set_index(result.columns[0]), height=chart_height)
                                    else:
                                        st.warning("Bar chart requires at least 2 columns (x and y)")
                                
                                elif chart["type"] == "line_chart":
                                    if len(result.columns) >= 2:
                                        st.line_chart(result.set_index(result.columns[0]), height=chart_height)
                                    else:
                                        st.warning("Line chart requires at least 2 columns (x and y)")
                                
                                elif chart["type"] == "pie_chart":
                                    # Create a matplotlib pie chart
                                    if len(result.columns) >= 2:
                                        fig, ax = plt.subplots()
                                        ax.pie(result[result.columns[1]], labels=result[result.columns[0]], autopct='%1.1f%%')
                                        ax.axis('equal')
                                        st.pyplot(fig)
                                    else:
                                        st.warning("Pie chart requires at least 2 columns (labels and values)")
                                
                                elif chart["type"] == "scatter_plot":
                                    if len(result.columns) >= 2:
                                        # For scatter plots with more than 2 columns, use the 3rd column for the point size
                                        if len(result.columns) >= 3:
                                            fig, ax = plt.subplots()
                                            ax.scatter(result[result.columns[0]], result[result.columns[1]], 
                                                     s=result[result.columns[2]] * 10)  # Scale the sizes
                                            ax.set_xlabel(result.columns[0])
                                            ax.set_ylabel(result.columns[1])
                                            st.pyplot(fig)
                                        else:
                                            # Basic scatter plot with two columns
                                            st.scatter_chart(result.set_index(result.columns[0]), height=chart_height)
                                    else:
                                        st.warning("Scatter plot requires at least 2 columns (x and y)")
                                
                                elif chart["type"] == "table":
                                    st.dataframe(result)
                                
                                elif chart["type"] == "card":
                                    # Card view for a single value
                                    if not result.empty:
                                        value = result.iloc[0, 0]
                                        st.metric(chart["title"], value)
                                    else:
                                        st.warning("No data available for card")
                                
                                elif chart["type"] == "heatmap":
                                    if len(result.columns) >= 3:
                                        try:
                                            # Check for duplicate indices
                                            if result.duplicated(subset=[result.columns[0], result.columns[1]]).any():
                                                st.warning("Data contains duplicate x-y combinations. Using the mean value for duplicates.")
                                                # Use pivot_table instead of pivot to handle duplicates
                                                pivot_table = pd.pivot_table(
                                                    result,
                                                    index=result.columns[0],
                                                    columns=result.columns[1],
                                                    values=result.columns[2],
                                                    aggfunc='mean'  # Use mean for duplicate values
                                                )
                                            else:
                                                # No duplicates, use regular pivot
                                                pivot_table = result.pivot(
                                                    index=result.columns[0],
                                                    columns=result.columns[1],
                                                    values=result.columns[2]
                                                )
                                            
                                            # Handle empty pivot table
                                            if pivot_table.empty:
                                                st.warning("No data to display in heatmap after pivoting.")
                                                return
                                                
                                            # Create the heatmap
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            heatmap = ax.pcolor(pivot_table)
                                            ax.set_xlabel(result.columns[1])
                                            ax.set_ylabel(result.columns[0])
                                            
                                            # Set tick positions and labels safely
                                            if len(pivot_table.columns) > 0:
                                                ax.set_xticks(np.arange(0.5, len(pivot_table.columns)))
                                                ax.set_xticklabels(pivot_table.columns)
                                            if len(pivot_table.index) > 0:
                                                ax.set_yticks(np.arange(0.5, len(pivot_table.index)))
                                                ax.set_yticklabels(pivot_table.index)
                                                
                                            plt.colorbar(heatmap)
                                            st.pyplot(fig)
                                        except Exception as e:
                                            st.error(f"Error creating heatmap: {str(e)}")
                                            # Fallback to a simple table view
                                            st.write("Showing data in table format instead:")
                                            st.dataframe(result)
                                    else:
                                        st.warning("Heatmap requires at least 3 columns (x, y, and value)")
                            else:
                                st.info("No data returned for this chart")
                            
                        except Exception as e:
                            st.error(f"Error executing query: {str(e)}")
                        
                        # Option to remove chart
                        if st.button("Remove Chart", key=f"remove_chart_{i}"):
                            dashboard_data["charts"].pop(i)
                            st.success("Chart removed")
                            st.rerun()
            
            else:
                # Grid layout
                num_charts = len(dashboard_data["charts"])
                rows = (num_charts + 1) // 2  # 2 charts per row
                
                for row in range(rows):
                    cols = st.columns(2)
                    
                    for col in range(2):
                        chart_idx = row * 2 + col
                        
                        if chart_idx < num_charts:
                            chart = dashboard_data["charts"][chart_idx]
                            
                            with cols[col]:
                                st.subheader(chart["title"])
                                
                                # Execute query to get chart data
                                if not chart["query"]:
                                    st.info("No query defined for this chart")
                                    continue
                                
                                try:
                                    # Execute query
                                    result = db_manager.execute_query(chart["query"])
                                    
                                    if isinstance(result, pd.DataFrame) and not result.empty:
                                        # Display chart based on type (simplified)
                                        if chart["type"] == "bar_chart":
                                            st.bar_chart(result.set_index(result.columns[0]))
                                        elif chart["type"] == "line_chart":
                                            st.line_chart(result.set_index(result.columns[0]))
                                        elif chart["type"] == "table":
                                            st.dataframe(result)
                                        else:
                                            st.write("Chart type not supported in grid view")
                                    else:
                                        st.info("No data returned for this chart")
                                
                                except Exception as e:
                                    st.error(f"Error executing query: {str(e)}")
                                
                                # Option to remove chart
                                if st.button("Remove Chart", key=f"remove_chart_{chart_idx}"):
                                    dashboard_data["charts"].pop(chart_idx)
                                    st.success("Chart removed")
                                    st.rerun()
        
        # Delete dashboard button
        if st.button("Delete Dashboard"):
            if st.session_state.current_dashboard in st.session_state.dashboards:
                del st.session_state.dashboards[st.session_state.current_dashboard]
                st.session_state.current_dashboard = None if not st.session_state.dashboards else list(st.session_state.dashboards.keys())[0]
                st.success("Dashboard deleted")
                st.rerun()
    
    def _data_explorer_ui(self, db_manager):
        """UI for interactive data exploration"""
        st.subheader("Interactive Data Explorer")
        
        # Table selector
        table_options = []
        
        # Get available tables from connected database
        schema = st.session_state.get("db_schema", {})
        
        if "tables" in schema:
            table_options = list(schema["tables"].keys())
        elif "collections" in schema:
            table_options = list(schema["collections"].keys())
        
        if not table_options:
            st.warning("No tables found in the connected database")
            return
        
        selected_table = st.selectbox("Select Table/Collection", options=table_options)
        
        if not selected_table:
            return
        
        # Get columns for the selected table
        columns = []
        if "tables" in schema and selected_table in schema["tables"]:
            columns = [col["name"] for col in schema["tables"][selected_table]["columns"]]
        elif "collections" in schema and selected_table in schema["collections"]:
            columns = [field["name"] for field in schema["collections"][selected_table]["fields"]]
        
        if not columns:
            st.warning(f"No columns found for table {selected_table}")
            return
        
        # Data exploration options
        exploration_options = ["Column Statistics", "Distribution Analysis", "Correlation Analysis", "Anomaly Detection"]
        selected_exploration = st.selectbox("Exploration Type", options=exploration_options)
        
        if selected_exploration == "Column Statistics":
            self._column_statistics_ui(db_manager, selected_table, columns)
        
        elif selected_exploration == "Distribution Analysis":
            self._distribution_analysis_ui(db_manager, selected_table, columns)
        
        elif selected_exploration == "Correlation Analysis":
            self._correlation_analysis_ui(db_manager, selected_table, columns)
        
        elif selected_exploration == "Anomaly Detection":
            self._anomaly_detection_ui(db_manager, selected_table, columns)
    
    def _column_statistics_ui(self, db_manager, table: str, columns: List[str]):
        """UI for column statistics"""
        st.subheader("Column Statistics")
        
        # Column selector
        selected_columns = st.multiselect("Select Columns", options=columns)
        
        if not selected_columns:
            st.info("Please select at least one column")
            return
        
        if st.button("Calculate Statistics"):
            with st.spinner("Calculating statistics..."):
                try:
                    # Generate SQL to calculate statistics for each column
                    sql_parts = []
                    
                    for col in selected_columns:
                        sql_parts.extend([
                            f"COUNT({col}) AS {col}_count",
                            f"MIN({col}) AS {col}_min",
                            f"MAX({col}) AS {col}_max",
                            f"AVG({col}::float) AS {col}_avg",
                            f"STDDEV({col}::float) AS {col}_stddev",
                            f"COUNT(DISTINCT {col}) AS {col}_unique"
                        ])
                    
                    sql = f"""
                    SELECT
                        {','.join(sql_parts)}
                    FROM {table}
                    """
                    
                    # Execute query
                    result = db_manager.execute_query(sql)
                    
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        # Reshape the result for better display
                        stats = {}
                        metrics = ["count", "min", "max", "avg", "stddev", "unique"]
                        
                        for col in selected_columns:
                            col_stats = {}
                            for metric in metrics:
                                col_key = f"{col}_{metric}"
                                if col_key in result.columns:
                                    col_stats[metric] = result.iloc[0][col_key]
                            
                            stats[col] = col_stats
                        
                        # Display statistics
                        for col, col_stats in stats.items():
                            st.subheader(f"Statistics for {col}")
                            
                            # Convert to dataframe for display
                            stats_df = pd.DataFrame(
                                [[metric, value] for metric, value in col_stats.items()],
                                columns=["Metric", "Value"]
                            )
                            
                            st.dataframe(stats_df)
                            
                            # Check for missing values
                            if "count" in col_stats:
                                # Execute query to get total rows
                                total_rows_result = db_manager.execute_query(f"SELECT COUNT(*) AS total FROM {table}")
                                
                                if isinstance(total_rows_result, pd.DataFrame) and not total_rows_result.empty:
                                    total_rows = total_rows_result.iloc[0]["total"]
                                    missing_count = total_rows - col_stats["count"]
                                    missing_percent = (missing_count / total_rows) * 100 if total_rows > 0 else 0
                                    
                                    st.write(f"Missing Values: {missing_count} ({missing_percent:.2f}%)")
                                    
                                    if missing_percent > 0:
                                        st.progress(missing_percent / 100)
                    else:
                        st.warning("No statistics could be calculated")
                
                except Exception as e:
                    st.error(f"Error calculating statistics: {str(e)}")
    
    def _distribution_analysis_ui(self, db_manager, table: str, columns: List[str]):
        """UI for distribution analysis"""
        st.subheader("Distribution Analysis")
        
        # Column selector
        selected_column = st.selectbox("Select Column", options=columns)
        
        if not selected_column:
            return
        
        # Analysis options
        analysis_type = st.selectbox(
            "Analysis Type",
            options=["Histogram", "Box Plot", "Time Series (if date/time)"]
        )
        
        # Parameters
        if analysis_type == "Histogram":
            bins = st.slider("Number of Bins", 5, 100, 20)
        
        if st.button("Generate Analysis"):
            with st.spinner("Analyzing data distribution..."):
                try:
                    if analysis_type == "Histogram":
                        # For histogram, get the data and let matplotlib determine bins
                        sql = f"""
                        SELECT {selected_column}
                        FROM {table}
                        WHERE {selected_column} IS NOT NULL
                        ORDER BY {selected_column}
                        """
                        
                        result = db_manager.execute_query(sql)
                        
                        if isinstance(result, pd.DataFrame) and not result.empty:
                            # Create histogram
                            fig, ax = plt.subplots()
                            ax.hist(result[selected_column], bins=bins)
                            ax.set_xlabel(selected_column)
                            ax.set_ylabel("Frequency")
                            ax.set_title(f"Distribution of {selected_column}")
                            st.pyplot(fig)
                            
                            # Show summary statistics
                            st.subheader("Summary Statistics")
                            summary = result[selected_column].describe()
                            st.dataframe(summary)
                        else:
                            st.warning("No data available for histogram")
                    
                    elif analysis_type == "Box Plot":
                        # Get data for box plot
                        sql = f"""
                        SELECT {selected_column}
                        FROM {table}
                        WHERE {selected_column} IS NOT NULL
                        """
                        
                        result = db_manager.execute_query(sql)
                        
                        if isinstance(result, pd.DataFrame) and not result.empty:
                            # Create box plot
                            fig, ax = plt.subplots()
                            ax.boxplot(result[selected_column])
                            ax.set_xlabel(selected_column)
                            ax.set_title(f"Box Plot of {selected_column}")
                            plt.xticks([1], [selected_column])
                            st.pyplot(fig)
                            
                            # Show potential outliers
                            q1 = result[selected_column].quantile(0.25)
                            q3 = result[selected_column].quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            
                            outliers = result[(result[selected_column] < lower_bound) | (result[selected_column] > upper_bound)]
                            
                            if not outliers.empty:
                                st.subheader("Potential Outliers")
                                st.write(f"Found {len(outliers)} potential outliers out of {len(result)} values")
                                st.dataframe(outliers.head(10) if len(outliers) > 10 else outliers)
                        else:
                            st.warning("No data available for box plot")
                    
                    elif analysis_type == "Time Series (if date/time)":
                        # Try to determine if this is a time/date column
                        # This is a simplified check - would need more robust type checking in practice
                        time_indicators = ["date", "time", "timestamp", "created", "updated", "at", "day", "month", "year"]
                        
                        is_time_column = any(indicator in selected_column.lower() for indicator in time_indicators)
                        
                        if not is_time_column:
                            st.warning(f"{selected_column} doesn't appear to be a time/date column")
                            return
                        
                        # Group by date part for the time series
                        sql = f"""
                        SELECT
                            DATE_TRUNC('day', {selected_column}) AS date_group,
                            COUNT(*) AS count
                        FROM {table}
                        WHERE {selected_column} IS NOT NULL
                        GROUP BY date_group
                        ORDER BY date_group
                        """
                        
                        result = db_manager.execute_query(sql)
                        
                        if isinstance(result, pd.DataFrame) and not result.empty:
                            # Create time series plot
                            fig, ax = plt.subplots()
                            ax.plot(result["date_group"], result["count"])
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Count")
                            ax.set_title(f"Time Series of {selected_column}")
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Check for seasonality
                            if len(result) >= 30:  # Need sufficient data points
                                st.subheader("Seasonality Analysis")
                                
                                # Extract day of week and month for basic seasonality
                                result["dow"] = pd.to_datetime(result["date_group"]).dt.day_name()
                                result["month"] = pd.to_datetime(result["date_group"]).dt.month_name()
                                
                                # Day of week aggregation
                                dow_agg = result.groupby("dow")["count"].mean().reset_index()
                                dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                                dow_agg["dow"] = pd.Categorical(dow_agg["dow"], categories=dow_order, ordered=True)
                                dow_agg = dow_agg.sort_values("dow")
                                
                                # Month aggregation
                                month_agg = result.groupby("month")["count"].mean().reset_index()
                                month_order = ["January", "February", "March", "April", "May", "June", 
                                              "July", "August", "September", "October", "November", "December"]
                                month_agg["month"] = pd.Categorical(month_agg["month"], categories=month_order, ordered=True)
                                month_agg = month_agg.sort_values("month")
                                
                                # Plot day of week pattern
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                                
                                ax1.bar(dow_agg["dow"], dow_agg["count"])
                                ax1.set_xlabel("Day of Week")
                                ax1.set_ylabel("Average Count")
                                ax1.set_title("Day of Week Pattern")
                                plt.setp(ax1.get_xticklabels(), rotation=45)
                                
                                ax2.bar(month_agg["month"], month_agg["count"])
                                ax2.set_xlabel("Month")
                                ax2.set_ylabel("Average Count")
                                ax2.set_title("Month Pattern")
                                plt.setp(ax2.get_xticklabels(), rotation=45)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                        else:
                            st.warning("No time series data available")
                
                except Exception as e:
                    st.error(f"Error analyzing distribution: {str(e)}")
    
    def _correlation_analysis_ui(self, db_manager, table: str, columns: List[str]):
        """UI for correlation analysis"""
        st.subheader("Correlation Analysis")
        
        # Column selectors
        selected_columns = st.multiselect(
            "Select Numeric Columns",
            options=columns,
            help="Select numeric columns to analyze correlations"
        )
        
        if len(selected_columns) < 2:
            st.info("Please select at least 2 columns")
            return
        
        if st.button("Calculate Correlations"):
            with st.spinner("Calculating correlations..."):
                try:
                    # Get data for correlation analysis
                    sql = f"""
                    SELECT {', '.join(selected_columns)}
                    FROM {table}
                    WHERE {' AND '.join([f'{col} IS NOT NULL' for col in selected_columns])}
                    """
                    
                    result = db_manager.execute_query(sql)
                    
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        # Calculate correlation matrix
                        corr_matrix = result.corr()
                        
                        # Plot correlation heatmap
                        fig, ax = plt.subplots(figsize=(10, 8))
                        im = ax.imshow(corr_matrix, cmap="coolwarm")
                        
                        # Add labels
                        ax.set_xticks(np.arange(len(selected_columns)))
                        ax.set_yticks(np.arange(len(selected_columns)))
                        ax.set_xticklabels(selected_columns)
                        ax.set_yticklabels(selected_columns)
                        
                        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                        
                        # Add colorbar
                        cbar = ax.figure.colorbar(im, ax=ax)
                        cbar.ax.set_ylabel("Correlation Coefficient", rotation=-90, va="bottom")
                        
                        # Loop over data dimensions and create text annotations
                        for i in range(len(selected_columns)):
                            for j in range(len(selected_columns)):
                                text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                                               ha="center", va="center", color="black")
                        
                        ax.set_title("Correlation Matrix")
                        fig.tight_layout()
                        
                        st.pyplot(fig)
                        
                        # Highlight strong correlations
                        st.subheader("Strong Correlations")
                        
                        strong_corrs = []
                        for i in range(len(selected_columns)):
                            for j in range(i+1, len(selected_columns)):
                                corr = corr_matrix.iloc[i, j]
                                if abs(corr) > 0.5:  # Threshold for strong correlation
                                    strong_corrs.append({
                                        "Column 1": selected_columns[i],
                                        "Column 2": selected_columns[j],
                                        "Correlation": corr
                                    })
                        
                        if strong_corrs:
                            strong_df = pd.DataFrame(strong_corrs)
                            # Sort by absolute correlation
                            strong_df["Abs Correlation"] = strong_df["Correlation"].abs()
                            strong_df = strong_df.sort_values("Abs Correlation", ascending=False).drop("Abs Correlation", axis=1)
                            
                            st.dataframe(strong_df)
                            
                            # Scatter plots for top 3 correlations
                            if len(strong_corrs) > 0:
                                st.subheader("Top Correlation Scatter Plots")
                                
                                for i, corr in enumerate(strong_corrs[:3]):
                                    col1, col2 = corr["Column 1"], corr["Column 2"]
                                    
                                    fig, ax = plt.subplots()
                                    ax.scatter(result[col1], result[col2], alpha=0.5)
                                    ax.set_xlabel(col1)
                                    ax.set_ylabel(col2)
                                    ax.set_title(f"Correlation: {corr['Correlation']:.2f}")
                                    
                                    # Add trend line
                                    z = np.polyfit(result[col1], result[col2], 1)
                                    p = np.poly1d(z)
                                    ax.plot(result[col1], p(result[col1]), "r--")
                                    
                                    st.pyplot(fig)
                        else:
                            st.info("No strong correlations found (threshold: 0.5)")
                    else:
                        st.warning("No data available for correlation analysis")
                
                except Exception as e:
                    st.error(f"Error calculating correlations: {str(e)}")
    
    def _anomaly_detection_ui(self, db_manager, table: str, columns: List[str]):
        """UI for anomaly detection"""
        st.subheader("Anomaly Detection")
        
        # Column selector
        selected_column = st.selectbox("Select Column for Anomaly Detection", options=columns)
        
        if not selected_column:
            return
        
        # Method selector
        method = st.selectbox(
            "Detection Method",
            options=["Z-Score", "IQR (Interquartile Range)", "Moving Average"]
        )
        
        # Parameters
        if method == "Z-Score":
            threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)
        elif method == "IQR (Interquartile Range)":
            iqr_factor = st.slider("IQR Factor", 1.0, 3.0, 1.5, 0.1)
        elif method == "Moving Average":
            window_size = st.slider("Window Size", 3, 30, 7)
            deviation_threshold = st.slider("Deviation Threshold", 1.0, 5.0, 2.0, 0.1)
        
        if st.button("Detect Anomalies"):
            with st.spinner("Detecting anomalies..."):
                try:
                    # Get data for anomaly detection
                    sql = f"""
                    SELECT {selected_column}
                    FROM {table}
                    WHERE {selected_column} IS NOT NULL
                    ORDER BY {selected_column}
                    """
                    
                    result = db_manager.execute_query(sql)
                    
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        if method == "Z-Score":
                            # Z-Score method
                            mean = result[selected_column].mean()
                            std = result[selected_column].std()
                            
                            result["z_score"] = (result[selected_column] - mean) / std
                            anomalies = result[abs(result["z_score"]) > threshold]
                            
                            # Plot data with anomalies
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(result.index, result[selected_column], 'b-', label='Data')
                            ax.scatter(anomalies.index, anomalies[selected_column], color='red', label='Anomalies')
                            
                            ax.axhline(mean + threshold * std, color='r', linestyle='--', label=f'Upper Threshold ({threshold} σ)')
                            ax.axhline(mean - threshold * std, color='r', linestyle='--', label=f'Lower Threshold ({threshold} σ)')
                            
                            ax.set_xlabel("Data Point Index")
                            ax.set_ylabel(selected_column)
                            ax.set_title(f"Z-Score Anomaly Detection for {selected_column}")
                            ax.legend()
                            
                            st.pyplot(fig)
                            
                            st.subheader("Detected Anomalies")
                            st.write(f"Found {len(anomalies)} anomalies out of {len(result)} data points ({len(anomalies)/len(result)*100:.2f}%)")
                            
                            if not anomalies.empty:
                                st.dataframe(anomalies.head(20) if len(anomalies) > 20 else anomalies)
                        
                        elif method == "IQR (Interquartile Range)":
                            # IQR method
                            q1 = result[selected_column].quantile(0.25)
                            q3 = result[selected_column].quantile(0.75)
                            iqr = q3 - q1
                            
                            lower_bound = q1 - iqr_factor * iqr
                            upper_bound = q3 + iqr_factor * iqr
                            
                            anomalies = result[(result[selected_column] < lower_bound) | (result[selected_column] > upper_bound)]
                            
                            # Plot data with anomalies
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(result.index, result[selected_column], 'b-', label='Data')
                            ax.scatter(anomalies.index, anomalies[selected_column], color='red', label='Anomalies')
                            
                            ax.axhline(upper_bound, color='r', linestyle='--', label=f'Upper Bound (Q3 + {iqr_factor} * IQR)')
                            ax.axhline(lower_bound, color='r', linestyle='--', label=f'Lower Bound (Q1 - {iqr_factor} * IQR)')
                            ax.axhline(q1, color='g', linestyle=':', label='Q1')
                            ax.axhline(q3, color='g', linestyle=':', label='Q3')
                            
                            ax.set_xlabel("Data Point Index")
                            ax.set_ylabel(selected_column)
                            ax.set_title(f"IQR Anomaly Detection for {selected_column}")
                            ax.legend()
                            
                            st.pyplot(fig)
                            
                            st.subheader("Detected Anomalies")
                            st.write(f"Found {len(anomalies)} anomalies out of {len(result)} data points ({len(anomalies)/len(result)*100:.2f}%)")
                            
                            if not anomalies.empty:
                                st.dataframe(anomalies.head(20) if len(anomalies) > 20 else anomalies)
                        
                        elif method == "Moving Average":
                            # Moving Average method
                            result["moving_avg"] = result[selected_column].rolling(window=window_size).mean()
                            result["moving_std"] = result[selected_column].rolling(window=window_size).std()
                            
                            # Skip initial NaN values due to window
                            result = result.iloc[window_size-1:]
                            
                            # Calculate deviation
                            result["deviation"] = abs(result[selected_column] - result["moving_avg"]) / result["moving_std"]
                            anomalies = result[result["deviation"] > deviation_threshold]
                            
                            # Plot data with anomalies
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(result.index, result[selected_column], 'b-', label='Data')
                            ax.plot(result.index, result["moving_avg"], 'g-', label=f'Moving Avg (window={window_size})')
                            ax.scatter(anomalies.index, anomalies[selected_column], color='red', label='Anomalies')
                            
                            # Plot confidence bands
                            ax.fill_between(
                                result.index,
                                result["moving_avg"] - deviation_threshold * result["moving_std"],
                                result["moving_avg"] + deviation_threshold * result["moving_std"],
                                color='gray', alpha=0.2, label=f'±{deviation_threshold} σ Band'
                            )
                            
                            ax.set_xlabel("Data Point Index")
                            ax.set_ylabel(selected_column)
                            ax.set_title(f"Moving Average Anomaly Detection for {selected_column}")
                            ax.legend()
                            
                            st.pyplot(fig)
                            
                            st.subheader("Detected Anomalies")
                            st.write(f"Found {len(anomalies)} anomalies out of {len(result)} data points ({len(anomalies)/len(result)*100:.2f}%)")
                            
                            if not anomalies.empty:
                                anomaly_display = anomalies[["moving_avg", "moving_std", "deviation", selected_column]]
                                st.dataframe(anomaly_display.head(20) if len(anomaly_display) > 20 else anomaly_display)
                    else:
                        st.warning("No data available for anomaly detection")
                
                except Exception as e:
                    st.error(f"Error detecting anomalies: {str(e)}")
    
    def _forecasting_ui(self, db_manager):
        """UI for time series forecasting"""
        st.subheader("Time Series Forecasting")
        
        # Table selector
        table_options = []
        
        # Get available tables from connected database
        schema = st.session_state.get("db_schema", {})
        
        if "tables" in schema:
            table_options = list(schema["tables"].keys())
        elif "collections" in schema:
            table_options = list(schema["collections"].keys())
        
        if not table_options:
            st.warning("No tables found in the connected database")
            return
        
        selected_table = st.selectbox("Select Table/Collection", options=table_options, key="forecast_table")
        
        if not selected_table:
            return
        
        # Get columns for the selected table
        columns = []
        if "tables" in schema and selected_table in schema["tables"]:
            columns = [col["name"] for col in schema["tables"][selected_table]["columns"]]
        elif "collections" in schema and selected_table in schema["collections"]:
            columns = [field["name"] for field in schema["collections"][selected_table]["fields"]]
        
        # Column selectors
        time_column = st.selectbox(
            "Select Time/Date Column",
            options=[col for col in columns if any(time_ind in col.lower() for time_ind in ["date", "time", "day", "created", "updated"])],
            help="Select the column that contains time/date information"
        )
        
        value_column = st.selectbox(
            "Select Value Column to Forecast",
            options=[col for col in columns if col != time_column],
            help="Select the numeric column to forecast"
        )
        
        # Forecast parameters
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_periods = st.slider("Forecast Periods", 1, 100, 30, help="Number of periods to forecast")
        
        with col2:
            forecast_method = st.selectbox(
                "Forecasting Method",
                options=["Simple Moving Average", "Exponential Smoothing", "Linear Regression"]
            )
        
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                try:
                    # Get time series data
                    sql = f"""
                    SELECT {time_column}, {value_column}
                    FROM {selected_table}
                    WHERE {time_column} IS NOT NULL AND {value_column} IS NOT NULL
                    ORDER BY {time_column}
                    """
                    
                    result = db_manager.execute_query(sql)
                    
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        # Convert to time series if needed
                        if pd.api.types.is_numeric_dtype(result[time_column]):
                            # It's already numeric, treat as periods
                            time_series = result
                        else:
                            # Try to convert to datetime
                            time_series = result.copy()
                            time_series[time_column] = pd.to_datetime(time_series[time_column])
                        
                        # Generate forecast
                        if forecast_method == "Simple Moving Average":
                            window_size = st.slider("Window Size", 2, min(30, len(time_series)-1), 7)
                            
                            # Calculate moving average
                            time_series["moving_avg"] = time_series[value_column].rolling(window=window_size).mean()
                            
                            # Generate forecast
                            last_value = time_series[value_column].iloc[-window_size:].mean()
                            forecast_values = [last_value] * forecast_periods
                            
                            # Get last time point
                            last_time = time_series[time_column].iloc[-1]
                            
                            # Generate future time points
                            if pd.api.types.is_numeric_dtype(time_series[time_column]):
                                # Numeric time
                                time_diff = time_series[time_column].iloc[-1] - time_series[time_column].iloc[-2]
                                forecast_times = [last_time + i * time_diff for i in range(1, forecast_periods + 1)]
                            else:
                                # Datetime
                                time_diff = time_series[time_column].iloc[-1] - time_series[time_column].iloc[-2]
                                forecast_times = [last_time + i * time_diff for i in range(1, forecast_periods + 1)]
                            
                            # Create forecast dataframe
                            forecast_df = pd.DataFrame({
                                time_column: forecast_times,
                                value_column: forecast_values,
                                "type": "forecast"
                            })
                            
                            # Add type to original data
                            time_series["type"] = "actual"
                            
                            # Combine for plotting
                            combined = pd.concat([time_series[[time_column, value_column, "type"]], forecast_df])
                            
                            # Plot
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Plot original data
                            actual_data = combined[combined["type"] == "actual"]
                            ax.plot(actual_data[time_column], actual_data[value_column], 'b-', label='Actual')
                            
                            # Plot moving average
                            ax.plot(time_series[time_column], time_series["moving_avg"], 'g-', label=f'Moving Avg (window={window_size})')
                            
                            # Plot forecast
                            forecast_data = combined[combined["type"] == "forecast"]
                            ax.plot(forecast_data[time_column], forecast_data[value_column], 'r--', label='Forecast')
                            
                            ax.set_xlabel(time_column)
                            ax.set_ylabel(value_column)
                            ax.set_title(f"Time Series Forecast for {value_column}")
                            ax.legend()
                            
                            # Format x-axis for datetime
                            if not pd.api.types.is_numeric_dtype(combined[time_column]):
                                plt.xticks(rotation=45)
                            
                            st.pyplot(fig)
                            
                            # Show forecast values
                            st.subheader("Forecast Values")
                            st.dataframe(forecast_df[[time_column, value_column]])
                        
                        elif forecast_method == "Exponential Smoothing":
                            alpha = st.slider("Smoothing Factor (Alpha)", 0.01, 0.99, 0.3, 0.01)
                            
                            # Perform simple exponential smoothing
                            time_series["exp_smooth"] = time_series[value_column].ewm(alpha=alpha).mean()
                            
                            # Generate forecast
                            last_smooth = time_series["exp_smooth"].iloc[-1]
                            forecast_values = [last_smooth] * forecast_periods
                            
                            # Get last time point
                            last_time = time_series[time_column].iloc[-1]
                            
                            # Generate future time points
                            if pd.api.types.is_numeric_dtype(time_series[time_column]):
                                # Numeric time
                                time_diff = time_series[time_column].iloc[-1] - time_series[time_column].iloc[-2]
                                forecast_times = [last_time + i * time_diff for i in range(1, forecast_periods + 1)]
                            else:
                                # Datetime
                                time_diff = time_series[time_column].iloc[-1] - time_series[time_column].iloc[-2]
                                forecast_times = [last_time + i * time_diff for i in range(1, forecast_periods + 1)]
                            
                            # Create forecast dataframe
                            forecast_df = pd.DataFrame({
                                time_column: forecast_times,
                                value_column: forecast_values,
                                "type": "forecast"
                            })
                            
                            # Add type to original data
                            time_series["type"] = "actual"
                            
                            # Combine for plotting
                            combined = pd.concat([time_series[[time_column, value_column, "type"]], forecast_df])
                            
                            # Plot
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Plot original data
                            actual_data = combined[combined["type"] == "actual"]
                            ax.plot(actual_data[time_column], actual_data[value_column], 'b-', label='Actual')
                            
                            # Plot smoothed data
                            ax.plot(time_series[time_column], time_series["exp_smooth"], 'g-', label=f'Exp Smoothing (α={alpha})')
                            
                            # Plot forecast
                            forecast_data = combined[combined["type"] == "forecast"]
                            ax.plot(forecast_data[time_column], forecast_data[value_column], 'r--', label='Forecast')
                            
                            ax.set_xlabel(time_column)
                            ax.set_ylabel(value_column)
                            ax.set_title(f"Exponential Smoothing Forecast for {value_column}")
                            ax.legend()
                            
                            # Format x-axis for datetime
                            if not pd.api.types.is_numeric_dtype(combined[time_column]):
                                plt.xticks(rotation=45)
                            
                            st.pyplot(fig)
                            
                            # Show forecast values
                            st.subheader("Forecast Values")
                            st.dataframe(forecast_df[[time_column, value_column]])
                        
                        elif forecast_method == "Linear Regression":
                            # Add numeric time index for regression
                            time_series["time_idx"] = np.arange(len(time_series))
                            
                            # Perform linear regression
                            x = time_series["time_idx"].values.reshape(-1, 1)
                            y = time_series[value_column].values
                            
                            # Basic linear regression implementation
                            n = len(x)
                            x_mean = np.mean(x)
                            y_mean = np.mean(y)
                            
                            # Calculate slope and intercept
                            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
                            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
                            
                            slope = numerator / denominator
                            intercept = y_mean - slope * x_mean
                            
                            # Calculate regression line
                            time_series["trend"] = intercept + slope * time_series["time_idx"]
                            
                            # Generate forecast indices
                            forecast_indices = np.arange(len(time_series), len(time_series) + forecast_periods)
                            
                            # Generate forecast values
                            forecast_values = intercept + slope * forecast_indices
                            
                            # Get last time point
                            last_time = time_series[time_column].iloc[-1]
                            
                            # Generate future time points
                            if pd.api.types.is_numeric_dtype(time_series[time_column]):
                                # Numeric time
                                time_diff = time_series[time_column].iloc[-1] - time_series[time_column].iloc[-2]
                                forecast_times = [last_time + i * time_diff for i in range(1, forecast_periods + 1)]
                            else:
                                # Datetime
                                time_diff = time_series[time_column].iloc[-1] - time_series[time_column].iloc[-2]
                                forecast_times = [last_time + i * time_diff for i in range(1, forecast_periods + 1)]
                            
                            # Create forecast dataframe
                            forecast_df = pd.DataFrame({
                                time_column: forecast_times,
                                value_column: forecast_values,
                                "type": "forecast"
                            })
                            
                            # Add type to original data
                            time_series["type"] = "actual"
                            
                            # Combine for plotting
                            combined = pd.concat([time_series[[time_column, value_column, "type"]], forecast_df])
                            
                            # Plot
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Plot original data
                            actual_data = combined[combined["type"] == "actual"]
                            ax.plot(actual_data[time_column], actual_data[value_column], 'b-', label='Actual')
                            
                            # Plot trend line
                            ax.plot(time_series[time_column], time_series["trend"], 'g-', label='Linear Trend')
                            
                            # Plot forecast
                            forecast_data = combined[combined["type"] == "forecast"]
                            ax.plot(forecast_data[time_column], forecast_data[value_column], 'r--', label='Forecast')
                            
                            ax.set_xlabel(time_column)
                            ax.set_ylabel(value_column)
                            ax.set_title(f"Linear Regression Forecast for {value_column}")
                            ax.legend()
                            
                            # Format x-axis for datetime
                            if not pd.api.types.is_numeric_dtype(combined[time_column]):
                                plt.xticks(rotation=45)
                            
                            st.pyplot(fig)
                            
                            # Show forecast values
                            st.subheader("Forecast Values")
                            st.dataframe(forecast_df[[time_column, value_column]])
                            
                            # Show model details
                            st.subheader("Model Details")
                            st.write(f"Linear Regression Equation: y = {slope[0]:.4f}x + {intercept:.4f}")
                            
                            # Calculate R-squared
                            y_pred = intercept + slope * x
                            ss_total = sum((y - y_mean) ** 2)
                            ss_residual = sum((y - y_pred) ** 2)
                            r_squared = 1 - (ss_residual / ss_total)
                            
                            st.write(f"R² (Coefficient of Determination): {r_squared[0]:.4f}")
                    else:
                        st.warning("No data available for forecasting")
                
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
                    st.error(f"Detailed error: {type(e).__name__}")
                    import traceback
                    st.error(traceback.format_exc())