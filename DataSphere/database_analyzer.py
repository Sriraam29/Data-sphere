import streamlit as st
import pandas as pd
import json
import os
import re
from datetime import datetime
from sqlalchemy import create_engine, inspect, text
from collections import defaultdict

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


class DatabaseAnalyzer:
    """Comprehensive database analysis and reporting module.
    
    Analyzes the entire database and generates a detailed report including:
    - Table summaries with row counts and column statistics
    - Data type analysis and distribution
    - Duplicate/similar column detection across tables
    - Data quality analysis (nulls, uniqueness, patterns)
    - AI-generated insights and recommendations
    """

    def __init__(self):
        if "analysis_report" not in st.session_state:
            st.session_state.analysis_report = None
        if "analysis_in_progress" not in st.session_state:
            st.session_state.analysis_in_progress = False

    # ------------------------------------------------------------------ #
    #  Main UI
    # ------------------------------------------------------------------ #
    def database_analysis_ui(self, db_manager):
        """Main UI for the Database Analysis & Report feature."""
        st.subheader("🔍 Database Analysis & Intelligence Report")

        if not st.session_state.get("connected_db"):
            st.warning("Please connect to a database first.")
            return

        if not st.session_state.get("db_schema"):
            st.warning("No database schema available. Please reconnect to your database.")
            return

        db_type = st.session_state.current_connection.get("type", "")
        if db_type == "MongoDB":
            st.info("Database analysis is currently available for SQL databases (PostgreSQL, MySQL, SQLite).")
            return

        # ---------- action buttons ---------- #
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            run_analysis = st.button("🚀 Run Full Analysis", key="run_db_analysis_btn", type="primary")
        with col2:
            if st.session_state.analysis_report:
                if st.button("🗑️ Clear Report", key="clear_analysis_btn"):
                    st.session_state.analysis_report = None
                    st.rerun()
        with col3:
            pass  # spacer

        if run_analysis:
            self._run_analysis(db_manager)

        # ---------- display report ---------- #
        if st.session_state.analysis_report:
            self._display_report(st.session_state.analysis_report, db_manager)

    # ------------------------------------------------------------------ #
    #  Core analysis engine
    # ------------------------------------------------------------------ #
    def _run_analysis(self, db_manager):
        """Execute a comprehensive database analysis."""
        connection_string = db_manager._build_connection_string(st.session_state.current_connection)
        engine = create_engine(connection_string)
        schema = st.session_state.db_schema

        report = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "database_name": st.session_state.connected_db,
            "db_type": st.session_state.current_connection.get("type", "SQL"),
            "tables": {},
            "cross_table": {},
            "ai_insights": None,
        }

        tables = schema.get("tables", {})
        total_tables = len(tables)
        if total_tables == 0:
            st.warning("No tables found in the database.")
            return

        progress_bar = st.progress(0, text="Starting analysis…")
        status_text = st.empty()

        for idx, (table_name, table_info) in enumerate(tables.items()):
            progress = (idx + 1) / (total_tables + 1)  # +1 for cross-table step
            status_text.text(f"Analyzing table: {table_name} ({idx + 1}/{total_tables})")
            progress_bar.progress(progress, text=f"Analyzing {table_name}…")

            table_report = self._analyze_table(engine, table_name, table_info)
            report["tables"][table_name] = table_report

        # Cross-table analysis
        status_text.text("Running cross-table analysis…")
        progress_bar.progress((total_tables) / (total_tables + 1), text="Cross-table analysis…")
        report["cross_table"] = self._cross_table_analysis(report["tables"], schema)

        # AI insights
        status_text.text("Generating AI insights…")
        progress_bar.progress(1.0, text="Generating AI insights…")
        report["ai_insights"] = self._generate_ai_insights(report)

        engine.dispose()
        st.session_state.analysis_report = report

        progress_bar.empty()
        status_text.empty()
        st.success("✅ Database analysis complete!")
        st.rerun()

    def _analyze_table(self, engine, table_name, table_info):
        """Analyze a single table and return a detailed report dict."""
        result = {
            "columns": [],
            "row_count": 0,
            "size_estimate": None,
            "data_quality_score": 0,
        }

        try:
            with engine.connect() as conn:
                # Row count
                count_df = pd.read_sql(f"SELECT COUNT(*) AS cnt FROM \"{table_name}\"", conn)
                row_count = int(count_df["cnt"].iloc[0])
                result["row_count"] = row_count

                if row_count == 0:
                    # Still capture column metadata
                    for col in table_info["columns"]:
                        result["columns"].append({
                            "name": col["name"],
                            "type": col["type"],
                            "null_count": 0,
                            "null_pct": 0.0,
                            "distinct_count": 0,
                            "distinct_pct": 0.0,
                            "sample_values": [],
                            "is_unique": True,
                            "min_value": None,
                            "max_value": None,
                            "avg_value": None,
                            "is_primary_key": col["name"] in table_info.get("primary_keys", []),
                        })
                    result["data_quality_score"] = 100
                    return result

                # Per-column analysis
                quality_scores = []
                for col in table_info["columns"]:
                    col_report = self._analyze_column(engine, table_name, col, row_count, table_info)
                    result["columns"].append(col_report)
                    quality_scores.append(col_report.get("quality_score", 100))

                result["data_quality_score"] = round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else 100

        except Exception as e:
            result["error"] = str(e)

        return result

    def _analyze_column(self, engine, table_name, col_info, row_count, table_info):
        """Analyze a single column."""
        col_name = col_info["name"]
        col_type = col_info["type"].upper()
        is_pk = col_name in table_info.get("primary_keys", [])

        report = {
            "name": col_name,
            "type": col_info["type"],
            "is_primary_key": is_pk,
            "null_count": 0,
            "null_pct": 0.0,
            "distinct_count": 0,
            "distinct_pct": 0.0,
            "sample_values": [],
            "is_unique": False,
            "min_value": None,
            "max_value": None,
            "avg_value": None,
            "most_common": [],
            "quality_score": 100,
            "data_category": self._classify_column(col_name, col_type),
        }

        try:
            with engine.connect() as conn:
                # Null count and distinct count in one query
                stats_query = f"""
                    SELECT 
                        COUNT(*) - COUNT("{col_name}") AS null_count,
                        COUNT(DISTINCT "{col_name}") AS distinct_count
                    FROM "{table_name}"
                """
                stats_df = pd.read_sql(stats_query, conn)
                null_count = int(stats_df["null_count"].iloc[0])
                distinct_count = int(stats_df["distinct_count"].iloc[0])

                report["null_count"] = null_count
                report["null_pct"] = round((null_count / row_count) * 100, 2) if row_count else 0
                report["distinct_count"] = distinct_count
                report["distinct_pct"] = round((distinct_count / row_count) * 100, 2) if row_count else 0
                report["is_unique"] = distinct_count == row_count

                # Quality score: penalize high null rates
                null_penalty = min(report["null_pct"], 50)  # cap at 50
                report["quality_score"] = max(0, 100 - null_penalty)

                # Check if numeric
                is_numeric = any(t in col_type for t in [
                    "INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC",
                    "REAL", "SERIAL", "BIGINT", "SMALLINT", "TINYINT", "NUMBER"
                ])

                # Min / Max (works for most types)
                try:
                    minmax_query = f"""
                        SELECT MIN("{col_name}") AS min_val, MAX("{col_name}") AS max_val
                        FROM "{table_name}"
                    """
                    minmax_df = pd.read_sql(minmax_query, conn)
                    report["min_value"] = str(minmax_df["min_val"].iloc[0]) if minmax_df["min_val"].iloc[0] is not None else None
                    report["max_value"] = str(minmax_df["max_val"].iloc[0]) if minmax_df["max_val"].iloc[0] is not None else None
                except Exception:
                    pass

                # Average for numeric columns
                if is_numeric:
                    try:
                        avg_query = f'SELECT AVG(CAST("{col_name}" AS FLOAT)) AS avg_val FROM "{table_name}"'
                        avg_df = pd.read_sql(avg_query, conn)
                        val = avg_df["avg_val"].iloc[0]
                        report["avg_value"] = round(float(val), 4) if val is not None else None
                    except Exception:
                        pass

                # Sample values (top 5 distinct)
                try:
                    sample_query = f"""
                        SELECT DISTINCT "{col_name}" AS val 
                        FROM "{table_name}" 
                        WHERE "{col_name}" IS NOT NULL 
                        LIMIT 5
                    """
                    sample_df = pd.read_sql(sample_query, conn)
                    report["sample_values"] = [str(v) for v in sample_df["val"].tolist()]
                except Exception:
                    pass

                # Most common values (top 5)
                try:
                    freq_query = f"""
                        SELECT "{col_name}" AS val, COUNT(*) AS freq
                        FROM "{table_name}"
                        WHERE "{col_name}" IS NOT NULL
                        GROUP BY "{col_name}"
                        ORDER BY freq DESC
                        LIMIT 5
                    """
                    freq_df = pd.read_sql(freq_query, conn)
                    report["most_common"] = [
                        {"value": str(row["val"]), "count": int(row["freq"])}
                        for _, row in freq_df.iterrows()
                    ]
                except Exception:
                    pass

        except Exception as e:
            report["error"] = str(e)

        return report

    def _classify_column(self, col_name, col_type):
        """Classify a column into a semantic category based on name and type."""
        name = col_name.lower()
        ctype = col_type.upper()

        if any(k in name for k in ["id", "_pk", "key"]) and "INT" in ctype:
            return "identifier"
        if any(k in name for k in ["email", "mail"]):
            return "email"
        if any(k in name for k in ["phone", "tel", "mobile", "fax"]):
            return "phone"
        if any(k in name for k in ["date", "time", "created", "updated", "modified", "_at"]):
            return "datetime"
        if any(k in name for k in ["price", "cost", "amount", "salary", "revenue", "total", "fee", "rate"]):
            return "monetary"
        if any(k in name for k in ["name", "first_name", "last_name", "full_name", "username"]):
            return "name"
        if any(k in name for k in ["address", "street", "city", "state", "country", "zip", "postal"]):
            return "address"
        if any(k in name for k in ["url", "link", "website", "href"]):
            return "url"
        if any(k in name for k in ["description", "comment", "note", "bio", "text", "body", "content"]):
            return "text"
        if any(k in name for k in ["status", "state", "type", "category", "flag", "is_", "has_"]):
            return "categorical"
        if any(k in name for k in ["count", "quantity", "qty", "num", "number"]):
            return "count"
        if "BOOL" in ctype:
            return "boolean"
        if any(t in ctype for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL"]):
            return "numeric"
        if any(t in ctype for t in ["VARCHAR", "TEXT", "CHAR", "STRING"]):
            return "text"
        return "other"

    # ------------------------------------------------------------------ #
    #  Cross-table analysis
    # ------------------------------------------------------------------ #
    def _cross_table_analysis(self, tables_report, schema):
        """Analyze relationships and similarities across tables."""
        result = {
            "duplicate_columns": [],
            "similar_columns": [],
            "potential_joins": [],
            "naming_inconsistencies": [],
            "table_relationships": [],
        }

        # Collect all columns with their table context
        all_columns = []
        for table_name, table_report in tables_report.items():
            for col in table_report.get("columns", []):
                all_columns.append({
                    "table": table_name,
                    "name": col["name"],
                    "type": col["type"],
                    "distinct_count": col.get("distinct_count", 0),
                    "sample_values": col.get("sample_values", []),
                    "category": col.get("data_category", "other"),
                })

        # Find duplicate/similar column names across tables
        name_groups = defaultdict(list)
        for col in all_columns:
            normalized = col["name"].lower().strip()
            name_groups[normalized].append(col)

        for name, cols in name_groups.items():
            if len(cols) > 1:
                tables_list = [c["table"] for c in cols]
                types_list = [c["type"] for c in cols]
                type_match = len(set(t.upper() for t in types_list)) == 1

                result["duplicate_columns"].append({
                    "column_name": name,
                    "found_in_tables": tables_list,
                    "types": types_list,
                    "types_match": type_match,
                    "could_be_join_key": type_match,
                })

        # Find similar column names (fuzzy matching)
        col_names = list(set(c["name"].lower() for c in all_columns))
        for i in range(len(col_names)):
            for j in range(i + 1, len(col_names)):
                name_a, name_b = col_names[i], col_names[j]
                if name_a == name_b:
                    continue
                similarity = self._name_similarity(name_a, name_b)
                if similarity > 0.7 and similarity < 1.0:
                    tables_a = [c["table"] for c in all_columns if c["name"].lower() == name_a]
                    tables_b = [c["table"] for c in all_columns if c["name"].lower() == name_b]
                    result["similar_columns"].append({
                        "column_a": name_a,
                        "column_b": name_b,
                        "similarity": round(similarity, 2),
                        "tables_a": tables_a,
                        "tables_b": tables_b,
                    })

        # Detect relationships from foreign keys in schema
        tables_schema = schema.get("tables", {})
        for table_name, table_info in tables_schema.items():
            for fk in table_info.get("foreign_keys", []):
                result["table_relationships"].append({
                    "from_table": table_name,
                    "from_columns": fk["constrained_columns"],
                    "to_table": fk["referred_table"],
                    "to_columns": fk["referred_columns"],
                })

        # Potential joins (columns with same name and compatible types
        # that are NOT already connected by foreign keys)
        existing_fk_pairs = set()
        for rel in result["table_relationships"]:
            existing_fk_pairs.add((rel["from_table"], rel["to_table"]))
            existing_fk_pairs.add((rel["to_table"], rel["from_table"]))

        for dup in result["duplicate_columns"]:
            if dup["could_be_join_key"] and len(dup["found_in_tables"]) >= 2:
                tables = dup["found_in_tables"]
                for i in range(len(tables)):
                    for j in range(i + 1, len(tables)):
                        pair = (tables[i], tables[j])
                        if pair not in existing_fk_pairs:
                            result["potential_joins"].append({
                                "table_a": tables[i],
                                "table_b": tables[j],
                                "join_column": dup["column_name"],
                                "reason": f"Both tables have column '{dup['column_name']}' with matching type",
                            })

        return result

    def _name_similarity(self, a, b):
        """Simple Jaccard-based name similarity on character bigrams."""
        if not a or not b:
            return 0.0
        bigrams_a = set(a[i:i+2] for i in range(len(a) - 1))
        bigrams_b = set(b[i:i+2] for i in range(len(b) - 1))
        if not bigrams_a or not bigrams_b:
            return 0.0
        intersection = bigrams_a & bigrams_b
        union = bigrams_a | bigrams_b
        return len(intersection) / len(union)

    # ------------------------------------------------------------------ #
    #  AI insights generation
    # ------------------------------------------------------------------ #
    def _generate_ai_insights(self, report):
        """Generate AI-powered insights about the database using the configured LLM."""
        # Build a concise summary for the LLM
        summary_lines = []
        summary_lines.append(f"Database: {report['database_name']} ({report['db_type']})")
        summary_lines.append(f"Total tables: {len(report['tables'])}")
        summary_lines.append("")

        for table_name, t in report["tables"].items():
            summary_lines.append(f"--- Table: {table_name} ({t['row_count']} rows, quality score: {t['data_quality_score']}%) ---")
            for col in t.get("columns", []):
                parts = [
                    f"  {col['name']} ({col['type']})",
                    f"category={col.get('data_category', '?')}",
                    f"nulls={col['null_pct']}%",
                    f"distinct={col['distinct_count']}",
                    f"unique={col['is_unique']}",
                ]
                if col.get("min_value"):
                    parts.append(f"min={col['min_value']}")
                if col.get("max_value"):
                    parts.append(f"max={col['max_value']}")
                if col.get("avg_value") is not None:
                    parts.append(f"avg={col['avg_value']}")
                summary_lines.append(" | ".join(parts))
            summary_lines.append("")

        # Cross-table findings
        cross = report.get("cross_table", {})
        if cross.get("duplicate_columns"):
            summary_lines.append("--- Duplicate Columns Across Tables ---")
            for dup in cross["duplicate_columns"]:
                summary_lines.append(
                    f"  '{dup['column_name']}' in tables: {', '.join(dup['found_in_tables'])} "
                    f"(types match: {dup['types_match']})"
                )
        if cross.get("table_relationships"):
            summary_lines.append("--- Foreign Key Relationships ---")
            for rel in cross["table_relationships"]:
                summary_lines.append(
                    f"  {rel['from_table']}.{', '.join(rel['from_columns'])} -> "
                    f"{rel['to_table']}.{', '.join(rel['to_columns'])}"
                )

        summary_text = "\n".join(summary_lines)

        prompt = f"""You are an expert database analyst. Analyze the following database summary and produce a comprehensive intelligence report.

DATABASE SUMMARY:
{summary_text}

Please provide the following sections in your report (use markdown formatting):

## 📊 Executive Summary
A brief overview of the database — what it appears to store and its overall health.

## 🏗️ Schema Design Assessment
- Structure quality
- Normalization assessment
- Naming convention consistency

## 📈 Data Quality Findings
- Tables/columns with high null rates and why that matters
- Columns that should be unique but aren't
- Data type appropriateness

## 🔗 Relationship Analysis
- Existing relationships and their purpose
- Missing relationships or potential joins
- Duplicate columns and what they mean

## 💡 Key Insights & Patterns
- What stories does this data tell?
- Interesting patterns in the data distribution
- Column categories and what they reveal about the business domain

## ⚠️ Issues & Recommendations
- Data quality improvements
- Schema design improvements
- Performance optimization suggestions
- Missing indexes or constraints

## 🎯 Use Cases
- What kinds of analytics questions can this database answer?
- Suggested queries or dashboards

Keep the report concise but meaningful. Use bullet points for clarity.
"""

        # Try configured LLM
        model_provider = st.session_state.get("model_provider", "Groq" if st.session_state.get("groq_api_key") else "Gemini")

        try:
            if model_provider == "Groq" and st.session_state.get("groq_api_key"):
                client = Groq(api_key=st.session_state.groq_api_key)
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are an expert database analyst providing detailed database intelligence reports."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=3000,
                )
                return response.choices[0].message.content.strip()

            elif model_provider == "Gemini" and st.session_state.get("gemini_api_key") and GENAI_AVAILABLE:
                try:
                    client = genai.Client(api_key=st.session_state.gemini_api_key)
                    response = client.models.generate_content(
                        model="gemini-2.0-flash-exp",
                        contents=prompt
                    )
                    return response.text.strip()
                except AttributeError:
                    genai.configure(api_key=st.session_state.gemini_api_key)
                    model = genai.GenerativeModel("gemini-pro")
                    response = model.generate_content(prompt)
                    return response.text.strip()

        except Exception as e:
            return f"⚠️ Could not generate AI insights: {str(e)}\n\nPlease ensure your API key is configured in the Query Generator section."

        return "⚠️ No AI model configured. Please set up an API key in the Query Generator section to get AI-powered insights."

    # ------------------------------------------------------------------ #
    #  Report display
    # ------------------------------------------------------------------ #
    def _display_report(self, report, db_manager):
        """Display the analysis report in the Streamlit UI."""

        st.markdown("---")

        # Header
        st.markdown(f"""
        ### 📋 Database Intelligence Report
        **Database:** {report['database_name']} ({report['db_type']})  
        **Generated:** {report['generated_at']}  
        **Tables analyzed:** {len(report['tables'])}
        """)

        # Quick stats
        total_rows = sum(t["row_count"] for t in report["tables"].values())
        total_columns = sum(len(t["columns"]) for t in report["tables"].values())
        avg_quality = round(
            sum(t["data_quality_score"] for t in report["tables"].values()) / len(report["tables"]), 1
        ) if report["tables"] else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Tables", len(report["tables"]))
        col2.metric("📏 Total Rows", f"{total_rows:,}")
        col3.metric("📐 Total Columns", total_columns)
        col4.metric("✅ Avg Quality", f"{avg_quality}%")

        # Tabs for different report sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Table Details",
            "🔗 Cross-Table Analysis",
            "🤖 AI Insights",
            "📈 Data Quality",
            "📥 Export Report",
        ])

        with tab1:
            self._display_table_details(report)

        with tab2:
            self._display_cross_table(report)

        with tab3:
            self._display_ai_insights(report)

        with tab4:
            self._display_data_quality(report)

        with tab5:
            self._display_export(report)

    def _display_table_details(self, report):
        """Display detailed analysis for each table."""
        st.subheader("Table-by-Table Analysis")

        for table_name, t in report["tables"].items():
            quality_color = "🟢" if t["data_quality_score"] >= 80 else "🟡" if t["data_quality_score"] >= 50 else "🔴"
            with st.expander(f"{quality_color} **{table_name}** — {t['row_count']:,} rows | Quality: {t['data_quality_score']}%"):
                if t.get("error"):
                    st.error(f"Error analyzing table: {t['error']}")
                    continue

                # Column details table
                col_data = []
                for col in t["columns"]:
                    # Ensure all values are strings to prevent PyArrow mixed-type errors
                    avg_val = col.get("avg_value")
                    avg_display = str(round(avg_val, 4)) if avg_val is not None else ""
                    
                    col_data.append({
                        "Column": col["name"],
                        "Type": col["type"],
                        "Category": col.get("data_category", ""),
                        "PK": "✅" if col.get("is_primary_key") else "",
                        "Unique": "✅" if col.get("is_unique") else "",
                        "Nulls": f"{col['null_pct']}%",
                        "Distinct": str(col["distinct_count"]),
                        "Min": str(col.get("min_value", "")) if col.get("min_value") is not None else "",
                        "Max": str(col.get("max_value", "")) if col.get("max_value") is not None else "",
                        "Avg": avg_display,
                        "Sample Values": ", ".join(col.get("sample_values", [])[:3]),
                    })

                col_df = pd.DataFrame(col_data)
                st.dataframe(col_df, width="stretch", hide_index=True)

                # Most common values for interesting columns
                interesting_cols = [c for c in t["columns"] if c.get("most_common") and c.get("data_category") in
                                    ["categorical", "name", "address", "text", "other"] and not c.get("is_unique")]
                if interesting_cols:
                    st.markdown("**🏷️ Most Common Values:**")
                    for col in interesting_cols[:5]:  # limit to 5 columns
                        if col["most_common"]:
                            values_str = " | ".join(
                                f"`{v['value']}` ({v['count']})" for v in col["most_common"][:3]
                            )
                            st.markdown(f"- **{col['name']}**: {values_str}")

    def _display_cross_table(self, report):
        """Display cross-table analysis."""
        cross = report.get("cross_table", {})

        # Duplicate columns
        st.subheader("🔄 Duplicate Columns Across Tables")
        if cross.get("duplicate_columns"):
            dup_data = []
            for dup in cross["duplicate_columns"]:
                dup_data.append({
                    "Column Name": dup["column_name"],
                    "Found In Tables": ", ".join(dup["found_in_tables"]),
                    "Types": ", ".join(dup["types"]),
                    "Types Match": "✅" if dup["types_match"] else "❌",
                    "Potential Join Key": "✅" if dup["could_be_join_key"] else "❌",
                })
            st.dataframe(pd.DataFrame(dup_data), width="stretch", hide_index=True)
        else:
            st.info("No duplicate column names found across tables.")

        # Similar columns
        st.subheader("🔍 Similar Column Names")
        if cross.get("similar_columns"):
            sim_data = []
            for sim in cross["similar_columns"][:20]:  # limit display
                sim_data.append({
                    "Column A": sim["column_a"],
                    "Tables A": ", ".join(sim["tables_a"]),
                    "Column B": sim["column_b"],
                    "Tables B": ", ".join(sim["tables_b"]),
                    "Similarity": f"{sim['similarity']:.0%}",
                })
            st.dataframe(pd.DataFrame(sim_data), width="stretch", hide_index=True)
        else:
            st.info("No similar column names detected.")

        # Existing relationships
        st.subheader("🔗 Foreign Key Relationships")
        if cross.get("table_relationships"):
            rel_data = []
            for rel in cross["table_relationships"]:
                rel_data.append({
                    "From Table": rel["from_table"],
                    "From Columns": ", ".join(rel["from_columns"]),
                    "To Table": rel["to_table"],
                    "To Columns": ", ".join(rel["to_columns"]),
                })
            st.dataframe(pd.DataFrame(rel_data), width="stretch", hide_index=True)
        else:
            st.info("No foreign key relationships found.")

        # Potential joins
        st.subheader("💡 Potential Join Opportunities")
        if cross.get("potential_joins"):
            join_data = []
            for pj in cross["potential_joins"]:
                join_data.append({
                    "Table A": pj["table_a"],
                    "Table B": pj["table_b"],
                    "Join Column": pj["join_column"],
                    "Reason": pj["reason"],
                })
            st.dataframe(pd.DataFrame(join_data), width="stretch", hide_index=True)
        else:
            st.info("No additional join opportunities detected beyond existing foreign keys.")

    def _display_ai_insights(self, report):
        """Display AI-generated insights."""
        st.subheader("🤖 AI-Generated Database Intelligence")

        if report.get("ai_insights"):
            st.markdown(report["ai_insights"])
        else:
            st.info("No AI insights available. Configure an API key to enable AI-powered analysis.")

        # Regenerate button
        if st.button("🔄 Regenerate AI Insights", key="regen_ai_insights_btn"):
            with st.spinner("Regenerating insights…"):
                report["ai_insights"] = self._generate_ai_insights(report)
                st.session_state.analysis_report = report
            st.rerun()

    def _display_data_quality(self, report):
        """Display data quality analysis."""
        st.subheader("📈 Data Quality Dashboard")

        # Table quality scores
        quality_data = []
        for table_name, t in report["tables"].items():
            quality_data.append({
                "Table": table_name,
                "Rows": t["row_count"],
                "Columns": len(t["columns"]),
                "Quality Score": t["data_quality_score"],
                "Status": "🟢 Good" if t["data_quality_score"] >= 80 else "🟡 Fair" if t["data_quality_score"] >= 50 else "🔴 Needs Attention",
            })

        quality_df = pd.DataFrame(quality_data)
        st.dataframe(quality_df, width="stretch", hide_index=True)

        # Columns with issues
        st.subheader("⚠️ Columns Needing Attention")
        issues = []
        for table_name, t in report["tables"].items():
            for col in t.get("columns", []):
                if col["null_pct"] > 50:
                    issues.append({
                        "Table": table_name,
                        "Column": col["name"],
                        "Issue": "High null rate",
                        "Details": f"{col['null_pct']}% null values",
                        "Severity": "🔴 High",
                    })
                elif col["null_pct"] > 20:
                    issues.append({
                        "Table": table_name,
                        "Column": col["name"],
                        "Issue": "Moderate null rate",
                        "Details": f"{col['null_pct']}% null values",
                        "Severity": "🟡 Medium",
                    })
                if col["distinct_count"] == 1 and t["row_count"] > 1:
                    issues.append({
                        "Table": table_name,
                        "Column": col["name"],
                        "Issue": "Single value column",
                        "Details": f"Only 1 distinct value across {t['row_count']} rows",
                        "Severity": "🟡 Medium",
                    })

        if issues:
            issues_df = pd.DataFrame(issues)
            st.dataframe(issues_df, width="stretch", hide_index=True)
        else:
            st.success("✅ No significant data quality issues found!")

        # Data type distribution
        st.subheader("📊 Column Category Distribution")
        category_counts = defaultdict(int)
        for t in report["tables"].values():
            for col in t.get("columns", []):
                category_counts[col.get("data_category", "other")] += 1

        if category_counts:
            cat_df = pd.DataFrame([
                {"Category": cat.title(), "Count": count}
                for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])
            ])
            st.dataframe(cat_df, width="stretch", hide_index=True)

    def _display_export(self, report):
        """Display export options for the report."""
        st.subheader("📥 Export Analysis Report")

        # JSON export
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="📄 Download Full Report (JSON)",
            data=report_json,
            file_name=f"db_analysis_{report['database_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_json_report",
        )

        # CSV export - table summary
        summary_data = []
        for table_name, t in report["tables"].items():
            for col in t.get("columns", []):
                summary_data.append({
                    "Table": table_name,
                    "Column": col["name"],
                    "Type": col["type"],
                    "Category": col.get("data_category", ""),
                    "Is PK": col.get("is_primary_key", False),
                    "Is Unique": col.get("is_unique", False),
                    "Null %": col["null_pct"],
                    "Distinct Count": col["distinct_count"],
                    "Min": col.get("min_value", ""),
                    "Max": col.get("max_value", ""),
                    "Avg": col.get("avg_value", ""),
                    "Table Row Count": t["row_count"],
                    "Table Quality Score": t["data_quality_score"],
                })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Column Analysis (CSV)",
                data=csv,
                file_name=f"column_analysis_{report['database_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_csv_report",
            )

        # AI insights as text
        if report.get("ai_insights"):
            st.download_button(
                label="🤖 Download AI Insights (Markdown)",
                data=report["ai_insights"],
                file_name=f"ai_insights_{report['database_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key="download_ai_insights",
            )
