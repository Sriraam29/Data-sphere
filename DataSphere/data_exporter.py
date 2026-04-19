import streamlit as st
import pandas as pd
import io
import base64
import json
import csv

class DataExporter:
    def __init__(self):
        pass
    
    def export_ui(self, data):
        """UI for exporting data"""
        if not isinstance(data, pd.DataFrame):
            st.warning("No data available to export.")
            return
        
        # Export format selection
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "Excel", "JSON"],
            key="export_format"
        )
        
        # Export button
        if st.button("Export Data", key="export_btn"):
            if export_format == "CSV":
                self._export_csv(data)
            elif export_format == "Excel":
                self._export_excel(data)
            elif export_format == "JSON":
                self._export_json(data)
    
    def _export_csv(self, data):
        """Export data to CSV"""
        csv_data = data.to_csv(index=False)
        b64 = base64.b64encode(csv_data.encode()).decode()
        
        href = f'<a href="data:file/csv;base64,{b64}" download="query_results.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    def _export_excel(self, data):
        """Export data to Excel"""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            data.to_excel(writer, index=False, sheet_name='Results')
        
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="query_results.xlsx">Download Excel</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    def _export_json(self, data):
        """Export data to JSON"""
        json_data = data.to_json(orient='records')
        b64 = base64.b64encode(json_data.encode()).decode()
        
        href = f'<a href="data:file/json;base64,{b64}" download="query_results.json">Download JSON</a>'
        st.markdown(href, unsafe_allow_html=True)
