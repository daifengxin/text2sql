# Text2SQL AI Assistant

This project implements a command-line interface (CLI) application that allows users to interact with SQLite databases using natural language queries, powered by OpenAI's GPT-4o model. It can also load data from CSV files, automatically infer table schemas, and handle potential schema conflicts.

## Features

*   **CSV Loading**: Load data from CSV files into SQLite tables.
*   **Dynamic Schema Inference**: Automatically determine column names and data types (TEXT, INTEGER, REAL) from CSV files and create corresponding SQLite tables.
*   **Schema Conflict Handling**: When loading a CSV into an existing table, detects schema mismatches and prompts the user to:
    *   Overwrite the existing table.
    *   Rename the import to a new table.
    *   Skip the current file.
*   **Direct SQL Execution**: Execute raw SQL queries against the database.
*   **Natural Language Queries**: Use OpenAI (GPT-4o by default) to translate natural language questions into SQL queries and execute them.
*   **Interactive CLI**: Provides commands for loading data, listing tables, checking schema, executing SQL, and asking questions in natural language.
*   **Logging**: Logs operations and errors to `error_log.txt` and the console.

## Project Structure

```
.text2sql/
├── data/                 # Directory for CSV input files
│   ├── products.csv
│   └── sales.csv
├── .env                  # Stores OpenAI API Key (!!! Add to .gitignore !!!)
├── database.db           # SQLite database file (created automatically)
├── error_log.txt         # Log file (created automatically)
├── main.py               # Main application script
├── prd.md                # Project requirements document (original)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup

1.  **Clone the Repository**: (If applicable)
    ```bash
    git clone <repository_url>
    cd text2sql
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up OpenAI API Key**:
    *   Create a file named `.env` in the project root directory.
    *   Add your OpenAI API key to the `.env` file:
        ```dotenv
        # .env
        OPENAI_API_KEY='your_openai_api_key_here'
        ```
    *   Replace `'your_openai_api_key_here'` with your actual key.
    *   **Important**: Add `.env` to your `.gitignore` file to avoid committing your key.

5.  **Prepare Data** (Optional):
    *   Place your CSV files in the `data/` directory. Example files (`products.csv`, `sales.csv`) are included.

## Usage

Run the main script from the project root directory:

```bash
python main.py
```

This will start the interactive CLI. You will see a welcome message and a list of available commands.

**Available Commands:**

*   `load <csv_file_path> <table_name>`: Load data from a CSV file into a specified table. If only the filename is provided, it assumes the file is in the `data/` directory. Handles schema inference and conflicts.
    *   Example: `load products.csv products`
*   `ask <natural_language_query>`: Ask a question in natural language. The application will generate and execute the corresponding SQL query using OpenAI.
    *   Example: `ask show all products with price greater than 50`
*   `schema`: Display the current database schema (tables and columns) as seen by the AI.
*   `sql <sql_query>`: Execute a raw SQL query directly.
    *   Example: `sql SELECT * FROM sales WHERE quantity > 1`
*   `tables`: List all user-created tables in the database.
*   `help`: Show the list of available commands.
*   `exit`: Exit the application.

## How it Works

1.  **Initialization**: Connects to the SQLite database (`database.db`) and sets up logging.
2.  **CLI Loop**: Waits for user commands.
3.  **`load`**: Reads the CSV using Pandas, infers the schema (column names and data types), compares with any existing table schema, handles conflicts via user input (overwrite/rename/skip), creates/updates the table, and loads data using `pandas.DataFrame.to_sql`.
4.  **`sql`**: Executes the provided SQL using the `sqlite3` library and displays results or status.
5.  **`ask`**: Retrieves the current database schema, constructs a prompt for OpenAI including the schema and user question, sends the request to the specified GPT model, extracts the SQL from the response, and executes it using the `execute_sql` function.

## Dependencies

*   `pandas`: For reading and handling CSV data.
*   `openai`: For interacting with the OpenAI API.
*   `python-dotenv`: For loading the API key from the `.env` file. 