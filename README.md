# Text2SQL Assistant

An interactive database assistant that loads CSV files into an SQLite database 
and allows users to query the database using natural language (via OpenAI) 
or direct SQL commands.

## Features

*   Load data from CSV files into SQLite tables.
*   Automatically infer schema from CSVs.
*   Handle schema conflicts interactively (overwrite, rename, skip).
*   Query the database using natural language questions (requires OpenAI API key).
*   Execute raw SQL queries.
*   List available tables.
*   View database schema.

## Installation

```bash
pip install text2sql-assistant
```

## Usage

```bash
text2sql
```

This will start the interactive command-line interface. Type `help` for available commands.

## Configuration

An OpenAI API key is required for the `ask` command. Set the `OPENAI_API_KEY` 
environment variable. You can place it in a `.env` file in the directory where 
you run the command:

```
OPENAI_API_KEY="your_api_key_here"
```

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