# DataSphere Analytics

DataSphere Analytics is a text-to-SQL platform that converts natural language to database queries using AI. Connect to PostgreSQL, MySQL, SQLite, or MongoDB databases and visualize your data with interactive dashboards.

## Features

- **Text-to-SQL Conversion**: Convert natural language questions to SQL queries using AI
- **Multi-Database Support**: Connect to PostgreSQL, MySQL, SQLite, and MongoDB
- **Schema Visualization**: View database schema with interactive ER diagrams
- **Query Builder**: Build and execute SQL queries with a user-friendly interface
- **Data Export**: Export query results in various formats (CSV, Excel, JSON)
- **Query Optimization**: Analyze and optimize SQL queries for better performance
- **Advanced Visualization**: Create interactive dashboards with multiple chart types
- **Data Storytelling**: Generate insights from your data with AI assistance
- **Semantic Layer**: Create business-friendly data models
- **Collaboration**: Share queries and dashboards with team members
- **Cloud Storage**: Integration with AWS S3, Google Cloud Storage, and Azure

## Setup Instructions

### Using requirements.txt (Recommended for most users)

```bash
# Clone the repository
git clone https://github.com/mdyasir8055/Data-Sphere-Analytics.git
cd DataSphereAnalytics

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Using pyproject.toml (For package development)

```bash
# Clone the repository
git clone https://github.com/mdyasir8055/Data-Sphere-Analytics.git

# Install with pip in development mode
pip install -e .

# Run the application
streamlit run app.py
```

## Environment Variables

Create a `.env` file with your API keys (optional):
```
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

## Requirements

- Python 3.11+
- Database system (PostgreSQL, MySQL, SQLite, or MongoDB)
- Graphviz (for ER diagram visualization)

The application will be available at http://localhost:5000 by default.

## Graphviz Setup

For schema visualization features, Graphviz must be installed and properly configured:

1. Download and install Graphviz from [https://graphviz.org/download/](https://graphviz.org/download/)

2. Add Graphviz to your system PATH:

   **Windows:**
   - After installation, add the Graphviz bin directory to your system PATH
   - Typically: `C:\Program Files\Graphviz\bin`
   - Open System Properties > Advanced > Environment Variables
   - Edit the PATH variable and add the Graphviz bin directory
   - Restart your terminal or command prompt

   **macOS:**
   ```bash
   brew install graphviz
   ```

   **Linux:**
   ```bash
   sudo apt-get install graphviz
   ```

3. Verify installation:
   ```bash
   dot -v
   ```

This step is essential for the schema visualization features to work properly.