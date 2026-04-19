import streamlit as st
import pandas as pd
import re
import sqlparse
from typing import Dict, List, Tuple, Any, Optional

class QueryOptimizer:
    def __init__(self):
        """Initialize the Query Optimizer"""
        pass
    
    def optimize_query_ui(self, db_manager):
        """UI for query optimization features"""
        st.subheader("Query Optimization")
        
        # Get current query
        current_query = st.session_state.get("current_query", "")
        
        if not current_query:
            st.info("No query to optimize. Please create a query first.")
            return
        
        # Display original query
        with st.expander("Original Query", expanded=True):
            st.code(current_query, language="sql")
        
        # Query optimization features
        tab1, tab2, tab3 = st.tabs(["Query Analysis", "Execution Plan", "Index Recommendations"])
        
        with tab1:
            self._query_analysis_ui(current_query, db_manager)
        
        with tab2:
            self._execution_plan_ui(current_query, db_manager)
        
        with tab3:
            self._index_recommendations_ui(current_query, db_manager)
    
    def _query_analysis_ui(self, query: str, db_manager):
        """Analyze and provide insights on query structure"""
        st.subheader("Query Analysis")
        
        if st.button("Analyze Query"):
            with st.spinner("Analyzing query..."):
                # Parse query
                parsed = sqlparse.parse(query)[0]
                query_type = parsed.get_type()
                
                # Get tables and columns used in query
                tables, columns = self._extract_tables_and_columns(query)
                
                # Identify query complexity
                complexity_score, complexity_factors = self._assess_query_complexity(query)
                
                # Display analysis results
                st.subheader("Analysis Results")
                
                st.write("**Query Type:**", query_type)
                
                st.write("**Tables Referenced:**")
                for table in tables:
                    st.write(f"- {table}")
                
                st.write("**Columns Referenced:**")
                for col in columns:
                    st.write(f"- {col}")
                
                st.write(f"**Complexity Score:** {complexity_score}/10")
                
                st.write("**Complexity Factors:**")
                for factor, details in complexity_factors.items():
                    if details["found"]:
                        st.warning(f"- {factor}: {details['explanation']}")
                
                # Display formatting improvements
                formatted_query = sqlparse.format(
                    query, reindent=True, keyword_case='upper'
                )
                
                with st.expander("Formatted Query"):
                    st.code(formatted_query, language="sql")
    
    def _execution_plan_ui(self, query: str, db_manager):
        """Generate and display query execution plan"""
        st.subheader("Execution Plan")
        
        if st.button("Generate Execution Plan"):
            with st.spinner("Generating execution plan..."):
                try:
                    # Get database type to determine explain syntax
                    current_connection = st.session_state.get("current_connection") or {}
                    db_type = current_connection.get("type", "").lower()
                    
                    # Generate appropriate EXPLAIN command based on database type
                    if db_type == "postgresql":
                        explain_query = f"EXPLAIN (ANALYZE, COSTS, VERBOSE, BUFFERS, FORMAT JSON) {query}"
                    elif db_type == "mysql":
                        explain_query = f"EXPLAIN FORMAT=JSON {query}"
                    elif db_type == "sqlite":
                        explain_query = f"EXPLAIN QUERY PLAN {query}"
                    else:
                        st.error(f"Execution plan not supported for {db_type} databases")
                        return
                    
                    # Execute explain query
                    plan_result = db_manager.execute_query(explain_query)
                    
                    if plan_result is not None:
                        # Display execution plan
                        st.json(plan_result)
                        
                        # Analyze execution plan
                        performance_insights = self._analyze_execution_plan(plan_result, db_type)
                        
                        # Display performance insights
                        st.subheader("Performance Insights")
                        for category, insights in performance_insights.items():
                            with st.expander(category):
                                for insight in insights:
                                    if insight["type"] == "warning":
                                        st.warning(insight["message"])
                                    elif insight["type"] == "info":
                                        st.info(insight["message"])
                                    elif insight["type"] == "success":
                                        st.success(insight["message"])
                    else:
                        st.error("Failed to generate execution plan")
                except Exception as e:
                    st.error(f"Error generating execution plan: {str(e)}")
    
    def _index_recommendations_ui(self, query: str, db_manager):
        """Recommend indexes based on query patterns"""
        st.subheader("Index Recommendations")
        
        if st.button("Generate Index Recommendations"):
            with st.spinner("Analyzing query for index recommendations..."):
                try:
                    # Extract tables and query conditions
                    tables, columns = self._extract_tables_and_columns(query)
                    where_conditions = self._extract_where_conditions(query)
                    join_conditions = self._extract_join_conditions(query)
                    order_by_columns = self._extract_order_by_columns(query)
                    
                    # Generate recommendations
                    recommendations = self._generate_index_recommendations(
                        tables, columns, where_conditions, join_conditions, order_by_columns
                    )
                    
                    # Display recommendations
                    if recommendations:
                        st.subheader("Recommended Indexes")
                        for table, indexes in recommendations.items():
                            st.write(f"**Table: {table}**")
                            for idx, details in indexes.items():
                                st.info(f"Index on: {', '.join(details['columns'])}")
                                st.write(f"Reason: {details['reason']}")
                                st.code(details['create_statement'], language="sql")
                                st.write("---")
                    else:
                        st.info("No index recommendations found for this query")
                except Exception as e:
                    st.error(f"Error generating index recommendations: {str(e)}")
    
    def _extract_tables_and_columns(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract tables and columns from a SQL query"""
        # Parse the query
        parsed = sqlparse.parse(query)[0]
        
        # Extract tables (simplified approach - would need more robust parsing for complex queries)
        tables = []
        from_seen = False
        for token in parsed.tokens:
            if token.is_keyword and token.value.upper() == 'FROM':
                from_seen = True
            elif from_seen and token.ttype is None:  # Table names after FROM
                # Extract table names from the token
                table_token_str = token.value
                # Remove any JOIN keywords
                table_token_str = re.sub(r'\b(JOIN|INNER|LEFT|RIGHT|OUTER|CROSS|NATURAL)\b', ',', table_token_str, flags=re.IGNORECASE)
                # Split by commas and get table names
                table_names = [t.strip().split(' ')[-1] for t in table_token_str.split(',') if t.strip()]
                tables.extend(table_names)
                from_seen = False
        
        # Extract columns (simplified approach)
        columns = []
        select_seen = False
        for token in parsed.tokens:
            if token.is_keyword and token.value.upper() == 'SELECT':
                select_seen = True
            elif select_seen and token.ttype is None:  # Column names after SELECT
                # Extract column names from the token
                column_token_str = token.value
                # Split by commas and get column names
                column_names = [c.strip().split(' ')[-1] for c in column_token_str.split(',') if c.strip()]
                columns.extend(column_names)
                select_seen = False
        
        return tables, columns
    
    def _extract_where_conditions(self, query: str) -> List[str]:
        """Extract WHERE conditions from a SQL query"""
        # Find the WHERE clause
        where_match = re.search(r'\bWHERE\b(.*?)(?:\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|$)', query, re.IGNORECASE | re.DOTALL)
        
        if where_match:
            where_clause = where_match.group(1).strip()
            # Split by AND/OR to get individual conditions
            conditions = re.split(r'\b(AND|OR)\b', where_clause, flags=re.IGNORECASE)
            # Take only the condition parts, not the AND/OR separators
            conditions = [cond.strip() for i, cond in enumerate(conditions) if i % 2 == 0 and cond.strip()]
            return conditions
        
        return []
    
    def _extract_join_conditions(self, query: str) -> List[str]:
        """Extract JOIN conditions from a SQL query"""
        # Find all JOIN ... ON clauses
        join_matches = re.finditer(r'\b(JOIN|INNER JOIN|LEFT JOIN|RIGHT JOIN|FULL JOIN|OUTER JOIN)\b(.*?)\bON\b(.*?)(?:\bJOIN\b|\bWHERE\b|\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|$)', 
                                 query, re.IGNORECASE | re.DOTALL)
        
        join_conditions = []
        for match in join_matches:
            table = match.group(2).strip()
            condition = match.group(3).strip()
            join_conditions.append({
                'table': table,
                'condition': condition
            })
        
        return join_conditions
    
    def _extract_order_by_columns(self, query: str) -> List[str]:
        """Extract ORDER BY columns from a SQL query"""
        # Find the ORDER BY clause
        order_by_match = re.search(r'\bORDER BY\b(.*?)(?:\bLIMIT\b|$)', query, re.IGNORECASE | re.DOTALL)
        
        if order_by_match:
            order_by_clause = order_by_match.group(1).strip()
            # Split by commas to get individual columns
            columns = [col.strip() for col in order_by_clause.split(',')]
            return columns
        
        return []
    
    def _assess_query_complexity(self, query: str) -> Tuple[int, Dict[str, Any]]:
        """Assess the complexity of a SQL query"""
        complexity_score = 0
        complexity_factors = {
            "Nested Subqueries": {
                "found": False,
                "explanation": "Nested subqueries can significantly impact performance"
            },
            "Multiple Joins": {
                "found": False,
                "explanation": "Queries with multiple joins may require optimization"
            },
            "Complex Aggregations": {
                "found": False,
                "explanation": "Aggregations like COUNT, SUM, AVG may require optimization"
            },
            "Window Functions": {
                "found": False,
                "explanation": "Window functions can be resource-intensive"
            },
            "Complex Filtering": {
                "found": False,
                "explanation": "Many WHERE conditions or complex predicates"
            },
            "ORDER BY without INDEX": {
                "found": False,
                "explanation": "Sorting without proper indexes can be slow"
            },
            "DISTINCT Operator": {
                "found": False,
                "explanation": "DISTINCT requires additional processing"
            },
            "Large Table Operations": {
                "found": False,
                "explanation": "Operations on large tables may require optimization"
            }
        }
        
        # Check for nested subqueries
        if query.lower().count("select") > 1:
            complexity_factors["Nested Subqueries"]["found"] = True
            complexity_score += 2
        
        # Check for multiple joins
        join_count = len(re.findall(r'\bjoin\b', query, re.IGNORECASE))
        if join_count > 1:
            complexity_factors["Multiple Joins"]["found"] = True
            complexity_score += min(join_count, 3)  # Cap at 3 points
        
        # Check for complex aggregations
        aggregation_funcs = ['sum', 'avg', 'count', 'max', 'min']
        aggregation_count = sum(query.lower().count(f"{func}(") for func in aggregation_funcs)
        if aggregation_count > 0:
            complexity_factors["Complex Aggregations"]["found"] = True
            complexity_score += min(aggregation_count, 2)  # Cap at 2 points
        
        # Check for window functions
        if re.search(r'\bover\s*\(', query, re.IGNORECASE):
            complexity_factors["Window Functions"]["found"] = True
            complexity_score += 2
        
        # Check for complex filtering
        where_conditions = self._extract_where_conditions(query)
        if len(where_conditions) > 2:
            complexity_factors["Complex Filtering"]["found"] = True
            complexity_score += min(len(where_conditions) - 2, 2)  # Cap at 2 points
        
        # Check for DISTINCT operator
        if re.search(r'\bdistinct\b', query, re.IGNORECASE):
            complexity_factors["DISTINCT Operator"]["found"] = True
            complexity_score += 1
        
        # Cap the total score at 10
        return min(complexity_score, 10), complexity_factors
    
    def _analyze_execution_plan(self, plan_result, db_type: str) -> Dict[str, List[Dict[str, str]]]:
        """Analyze execution plan and provide performance insights"""
        # This is a placeholder implementation
        # In a real-world implementation, this would parse the specific format
        # of execution plans from different database systems
        
        insights = {
            "Major Concerns": [],
            "Optimization Opportunities": [],
            "Good Practices": []
        }
        
        # Sample insights (would be derived from actual plan analysis)
        insights["Major Concerns"].append({
            "type": "warning",
            "message": "Table scan detected. Consider adding indexes to improve performance."
        })
        
        insights["Optimization Opportunities"].append({
            "type": "info",
            "message": "Multiple join operations could be optimized by reordering joins."
        })
        
        insights["Good Practices"].append({
            "type": "success",
            "message": "Proper use of indexes for primary key lookups."
        })
        
        return insights
    
    def _generate_index_recommendations(self, tables, columns, where_conditions, join_conditions, order_by_columns) -> Dict[str, Dict[str, Any]]:
        """Generate index recommendations based on query analysis"""
        recommendations = {}
        
        # Process WHERE conditions for potential indexes
        for condition in where_conditions:
            # Simple heuristic to extract column names from conditions
            # In a real implementation, this would need more sophisticated parsing
            column_match = re.search(r'(\w+)\s*(?:=|>|<|>=|<=|!=|<>|LIKE|IN)', condition)
            if column_match:
                column = column_match.group(1)
                # Determine which table this column belongs to (simplified)
                # In a real implementation, would use schema information
                for table in tables:
                    if not table in recommendations:
                        recommendations[table] = {}
                    
                    index_name = f"idx_{table}_{column}"
                    if not index_name in recommendations[table]:
                        recommendations[table][index_name] = {
                            "columns": [column],
                            "reason": f"Used in WHERE condition: {condition}",
                            "create_statement": f"CREATE INDEX {index_name} ON {table} ({column});"
                        }
        
        # Process JOIN conditions for potential indexes
        for join in join_conditions:
            # Simple heuristic to extract column names from join conditions
            # In a real implementation, this would need more sophisticated parsing
            table = join["table"]
            condition = join["condition"]
            
            # Extract the column names from the join condition
            column_match = re.search(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', condition)
            if column_match:
                table1, col1, table2, col2 = column_match.groups()
                
                # Add recommendation for first table
                if not table1 in recommendations:
                    recommendations[table1] = {}
                
                index_name = f"idx_{table1}_{col1}"
                if not index_name in recommendations[table1]:
                    recommendations[table1][index_name] = {
                        "columns": [col1],
                        "reason": f"Used in JOIN condition: {condition}",
                        "create_statement": f"CREATE INDEX {index_name} ON {table1} ({col1});"
                    }
                
                # Add recommendation for second table
                if not table2 in recommendations:
                    recommendations[table2] = {}
                
                index_name = f"idx_{table2}_{col2}"
                if not index_name in recommendations[table2]:
                    recommendations[table2][index_name] = {
                        "columns": [col2],
                        "reason": f"Used in JOIN condition: {condition}",
                        "create_statement": f"CREATE INDEX {index_name} ON {table2} ({col2});"
                    }
        
        # Process ORDER BY columns for potential indexes
        for order_col in order_by_columns:
            # Remove any ASC/DESC specifiers
            order_col = re.sub(r'\s+(?:ASC|DESC)$', '', order_col, flags=re.IGNORECASE).strip()
            
            # Determine which table this column belongs to (simplified)
            # In a real implementation, would use schema information
            for table in tables:
                if not table in recommendations:
                    recommendations[table] = {}
                
                index_name = f"idx_{table}_{order_col}"
                if not index_name in recommendations[table]:
                    recommendations[table][index_name] = {
                        "columns": [order_col],
                        "reason": f"Used in ORDER BY clause",
                        "create_statement": f"CREATE INDEX {index_name} ON {table} ({order_col});"
                    }
        
        return recommendations