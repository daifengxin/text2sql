import sqlite3
import pandas as pd
import os
import logging
import sys
from openai import OpenAI # Added for Step 5
import re # Added for Step 5 to extract SQL
from dotenv import load_dotenv # Added to load .env file

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Database file path
DB_FILE = 'database.db'
# Data directory path
DATA_DIR = 'data'
# Log file path
LOG_FILE = 'error_log.txt'
# OpenAI API Key (Read from environment variable)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Added for Step 5
# OpenAI Model to use
OPENAI_MODEL = "gpt-4o" # Added for Step 5

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler(sys.stdout) # Also print logs to console
                    ])

def create_connection(db_file):
    """ Create a database connection to the SQLite database specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logging.info(f"Connected to SQLite database: {db_file}")
    except sqlite3.Error as e:
        logging.error(f"Error connecting to database {db_file}: {e}")
    return conn

def create_table(conn, create_table_sql):
    """ Create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
        logging.info(f"Executed: {create_table_sql}")
    except sqlite3.Error as e:
        logging.error(f"Error executing SQL: {create_table_sql} - {e}")

# Map pandas dtypes to SQLite types
def map_dtype_to_sqlite(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return 'INTEGER'
    elif pd.api.types.is_float_dtype(dtype):
        return 'REAL'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return 'TEXT' # Store dates/timestamps as TEXT
    else:
        return 'TEXT' # Default to TEXT for strings and other types

def get_table_schema(conn, table_name):
    """ Get the schema of an existing table. """
    try:
        cursor = conn.cursor()
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        schema = cursor.fetchall()
        # Return a list of tuples: (column_name, column_type)
        return [(col[1], col[2].upper()) for col in schema]
    except sqlite3.Error as e:
        logging.error(f"Error getting schema for table {table_name}: {e}")
        return None

def sanitize_column_name(col_name):
    """ Basic sanitization for SQL column names. """
    # Remove potentially problematic characters, replace spaces
    name = ''.join(c for c in col_name if c.isalnum() or c == '_' or c == ' ')
    name = name.replace(' ', '_').strip()
    if not name: # Handle empty names after sanitization
        name = 'unnamed_col'
    # Ensure it doesn't start with a number (SQLite specific issue? Best practice anyway)
    if name[0].isdigit():
        name = '_' + name
    return name

def create_and_load_table_from_csv(conn, csv_file_path, table_name):
    """
    Infer schema, create/check table, handle conflicts, and load data.
    :param conn: Connection object
    :param csv_file_path: Path to the CSV file
    :param table_name: Name of the target table
    """
    if not os.path.exists(csv_file_path):
        logging.error(f"CSV file not found at {csv_file_path}")
        return

    try:
        df = pd.read_csv(csv_file_path)
        if df.empty:
            logging.warning(f"CSV file {csv_file_path} is empty. Skipping.")
            return

        # --- Infer Schema from DataFrame ---
        inferred_schema_list = []
        sanitized_columns = []
        for col_name in df.columns:
            safe_col_name = sanitize_column_name(col_name)
            sanitized_columns.append(safe_col_name)
            sql_type = map_dtype_to_sqlite(df[col_name].dtype)
            inferred_schema_list.append((safe_col_name, sql_type))

        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        table_exists = cursor.fetchone()

        # Rename DataFrame columns to match sanitized names *before* potential conflict handling
        df.columns = sanitized_columns

        proceed_with_loading = True
        target_table_for_load = table_name

        if table_exists:
            logging.info(f"Table '{table_name}' already exists. Checking schema.")
            existing_schema = get_table_schema(conn, table_name)

            # Compare Schemas (simple comparison: number of columns, names, types)
            if existing_schema and inferred_schema_list == existing_schema:
                logging.info(f"Schema for table '{table_name}' matches CSV. Appending data.")
                # Schema matches, just append data
                load_action = 'append'
            else:
                logging.warning(f"Schema conflict detected for table '{table_name}'.")
                print(f"Schema mismatch for table '{table_name}':")
                print(f"  Existing: {existing_schema}")
                print(f"  CSV File: {inferred_schema_list}")

                while True:
                    action = input("Choose action: [O]verwrite table, [R]ename CSV import, [S]kip file: ").upper()
                    if action in ['O', 'R', 'S']:
                        break
                    print("Invalid choice. Please enter O, R, or S.")

                if action == 'O':
                    logging.info(f"User chose to OVERWRITE table '{table_name}'.")
                    try:
                        cursor.execute(f'DROP TABLE "{table_name}"')
                        logging.info(f"Dropped existing table '{table_name}'.")
                        # Need to recreate table with new schema
                        column_definitions = [f'"{name}" {type}' for name, type in inferred_schema_list]
                        create_table_sql = f"CREATE TABLE \"{table_name}\" ({', '.join(column_definitions)});"
                        create_table(conn, create_table_sql)
                        load_action = 'replace' # Or append to the newly created empty table
                    except sqlite3.Error as e:
                        logging.error(f"Failed to drop table '{table_name}': {e}")
                        proceed_with_loading = False

                elif action == 'R':
                    while True:
                        new_table_name = input("Enter new table name for this CSV data: ").strip()
                        if new_table_name:
                             # Basic check: ensure new name is not the conflicting name
                            if new_table_name == table_name:
                                print(f"New name cannot be the same as the conflicting table '{table_name}'.")
                                continue
                             # Check if the *new* name already exists (optional but good practice)
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (new_table_name,))
                            if cursor.fetchone():
                                print(f"Table '{new_table_name}' already exists. Choose a different name.")
                                continue
                            break # Valid new name received
                        else:
                           print("Table name cannot be empty.")

                    logging.info(f"User chose to RENAME CSV import to '{new_table_name}'.")
                    target_table_for_load = new_table_name
                    # Create the *new* table with the inferred schema
                    column_definitions = [f'"{name}" {type}' for name, type in inferred_schema_list]
                    create_table_sql = f"CREATE TABLE \"{target_table_for_load}\" ({', '.join(column_definitions)});"
                    create_table(conn, create_table_sql)
                    load_action = 'append' # Append to the newly created table

                elif action == 'S':
                    logging.info(f"User chose to SKIP file '{csv_file_path}' due to schema conflict with table '{table_name}'.")
                    proceed_with_loading = False

        else:
            # Table does not exist, create it
            logging.info(f"Table '{table_name}' does not exist. Creating table.")
            column_definitions = [f'"{name}" {type}' for name, type in inferred_schema_list]
            create_table_sql = f"CREATE TABLE \"{table_name}\" ({', '.join(column_definitions)});"
            create_table(conn, create_table_sql)
            load_action = 'append'

        # --- Load Data (if not skipped) ---
        if proceed_with_loading:
            try:
                logging.info(f"Loading data into table '{target_table_for_load}' (action: {load_action})")
                # Use if_exists='append' as we handle overwrite by dropping/recreating
                # or are loading into a new/renamed table.
                df.to_sql(target_table_for_load, conn, if_exists='append', index=False)
                logging.info(f"Successfully loaded data from {os.path.basename(csv_file_path)} to table '{target_table_for_load}'")
            except Exception as e:
                 logging.error(f"Error loading data from {csv_file_path} to table {target_table_for_load}: {e}")

    except pd.errors.EmptyDataError:
        logging.warning(f"CSV file {csv_file_path} is empty. Skipping.") # Already logged above, but ok
    except Exception as e:
        logging.error(f"Error processing CSV {csv_file_path} for table {table_name}: {e}")

def show_tables(conn):
    """ List all user-defined tables in the database. """
    try:
        cursor = conn.cursor()
        # Query sqlite_master table for user tables (type='table' and name not starting with 'sqlite_')
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = cursor.fetchall()
        if tables:
            print("Available tables:")
            for table in tables:
                print(f"- {table[0]}")
        else:
            print("No user tables found in the database.")
    except sqlite3.Error as e:
        logging.error(f"Error listing tables: {e}")

def execute_sql(conn, sql_query):
    """ Execute a given SQL query and print the results. """
    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)

        # Check if the query was a SELECT or similar that returns rows
        if cursor.description:
            rows = cursor.fetchall()
            if rows:
                # Print header
                col_names = [description[0] for description in cursor.description]
                print(" | ".join(col_names))
                print("-" * (len(" | ".join(col_names)))) # Separator line
                # Print rows
                for row in rows:
                    print(" | ".join(map(str, row)))
                logging.info(f"Executed query: {sql_query}. Returned {len(rows)} rows.")
            else:
                print("Query executed successfully, but returned no results.")
                logging.info(f"Executed query: {sql_query}. Returned 0 rows.")
        else:
            # For non-SELECT queries (INSERT, UPDATE, DELETE, CREATE, etc.)
            conn.commit() # Commit changes for non-select queries
            rowcount = cursor.rowcount # Number of rows affected for DML
            print(f"Query executed successfully. {rowcount if rowcount != -1 else 'No'} rows affected.")
            logging.info(f"Executed non-query statement: {sql_query}. Affected rows: {rowcount}")

    except sqlite3.Error as e:
        logging.error(f"Error executing SQL query \"{sql_query}\": {e}")
        print(f"Error executing SQL: {e}") # Also show error to user

def get_database_schema_string(conn):
    """ Generates a string representation of the database schema for the LLM. """
    schema_str = "Database Schema:\n"
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = cursor.fetchall()
        if not tables:
            return "Database Schema:\nNo user tables found.\n"

        for table_tuple in tables:
            table_name = table_tuple[0]
            schema_str += f"- Table: {table_name}\n"
            schema_info = get_table_schema(conn, table_name)
            if schema_info:
                schema_str += "  Columns:\n"
                for col_name, col_type in schema_info:
                    schema_str += f"    - {col_name} ({col_type})\n"
            else:
                 schema_str += "  (Could not retrieve schema info)\n"
        return schema_str
    except sqlite3.Error as e:
        logging.error(f"Error retrieving database schema: {e}")
        return f"Database Schema:\nError retrieving schema: {e}\n"

def generate_sql_with_openai(user_query, db_schema):
    """ Sends the user query and schema to OpenAI to generate SQL. """
    if not OPENAI_API_KEY:
        logging.error("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
        print("Error: OpenAI API key is not configured.")
        return None

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""
You are an AI assistant tasked with converting user queries into SQL statements. The database uses SQLite.

{db_schema}
User Query: "{user_query}"

Your task is:
1. Generate a single, executable SQLite query that accurately answers the user's question based *only* on the provided schema.
2. Ensure the SQL is compatible with SQLite syntax.
3. Do NOT add any explanation or commentary outside the SQL query itself. Only output the raw SQL query.
4. If the query cannot be answered with the given schema, respond with 'CANNOT_ANSWER'.

SQL Query:
"""

        logging.info(f"Sending prompt to OpenAI ({OPENAI_MODEL}):\n{prompt}")

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates natural language to SQL."}, # Optional: System message
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Lower temperature for more deterministic SQL
            max_tokens=150 # Adjust as needed
        )

        generated_text = response.choices[0].message.content.strip()
        logging.info(f"OpenAI ({OPENAI_MODEL}) response: \n{generated_text}")

        # Basic check for refusal
        if "CANNOT_ANSWER" in generated_text:
             print("AI indicates the query cannot be answered with the current database schema.")
             logging.warning(f"OpenAI indicated query cannot be answered for: {user_query}")
             return None

        # Attempt to extract SQL - often the model might wrap it in backticks or add minor comments
        # This regex tries to find common SQL patterns or code blocks
        sql_match = re.search(r"```(?:sql)?\s*(.*?)\s*```|(\bSELECT\b.*?);?", generated_text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            sql_query = sql_match.group(1) or sql_match.group(2)
            sql_query = sql_query.strip()
            print(f"\n🤖 Generated SQL:\n{sql_query}")
            logging.info(f"Extracted SQL: {sql_query}")
            return sql_query
        else:
            # If no clear SQL found, maybe the direct response is the SQL?
            # Check if it looks like a basic SQL command
            if generated_text.upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER")): 
                 print(f"\n🤖 Generated SQL (using direct response):\n{generated_text}")
                 logging.info(f"Using direct response as SQL: {generated_text}")
                 return generated_text
            else:
                print(f"Error: Could not extract a valid SQL query from the AI response: {generated_text}")
                logging.error(f"Failed to extract SQL from OpenAI response for query: {user_query}. Response: {generated_text}")
                return None

    except Exception as e:
        logging.error(f"Error interacting with OpenAI API: {e}")
        print(f"Error communicating with AI: {e}")
        return None

def print_help():
    """ Prints the help message for the CLI. """
    print("\nAvailable Commands:")
    print("  load <csv_file_path> <table_name> - Load data from CSV into a table.")
    print("                                     Handles schema inference and conflicts.")
    print("  ask <natural_language_query>     - Ask a question in natural language (uses AI).")
    print("  schema                           - Show the current database schema.")
    print("  sql <sql_query>                  - Execute a raw SQL query against the database.")
    print("  tables                           - List all tables in the database.")
    print("  help                             - Show this help message.")
    print("  exit                             - Exit the application.")
    print()

def main():
    if not os.path.exists(DATA_DIR):
        logging.error(f"Data directory '{DATA_DIR}' not found. Please create it and place CSV files inside.")
        return

    conn = create_connection(DB_FILE)
    if conn is None:
        logging.error("Failed to establish database connection. Exiting.")
        return

    print("Welcome to the Interactive Database Assistant!")
    print_help()

    # --- Interactive Loop ---
    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue

            parts = user_input.split(maxsplit=1) # Split command from arguments
            command = parts[0].lower()
            args_str = parts[1] if len(parts) > 1 else ""

            if command == "exit":
                break
            elif command == "help":
                print_help()
            elif command == "tables":
                show_tables(conn)
            elif command == "load":
                load_parts = args_str.split()
                if len(load_parts) == 2:
                    csv_path, table_name = load_parts
                    # Construct full path if only filename is given, assuming it's in DATA_DIR
                    if not os.path.dirname(csv_path) and os.path.exists(os.path.join(DATA_DIR, csv_path)):
                        csv_path = os.path.join(DATA_DIR, csv_path)
                        logging.info(f"Assuming CSV path refers to file in {DATA_DIR}: {csv_path}")
                    create_and_load_table_from_csv(conn, csv_path, table_name)
                else:
                    print("Usage: load <csv_file_path> <table_name>")
                    logging.warning("Invalid 'load' command usage.")
            elif command == "ask":
                if not OPENAI_API_KEY:
                    print("OpenAI API key not configured. Cannot use 'ask' command.")
                    logging.warning("'ask' command attempted without API key configured.")
                    continue
                if args_str:
                    print(f"\n🤔 Thinking about your question: '{args_str}'...")
                    schema_string = get_database_schema_string(conn)
                    logging.info("Retrieved schema for AI query.")
                    generated_sql = generate_sql_with_openai(args_str, schema_string)
                    if generated_sql:
                        # Optional: Ask user for confirmation before executing?
                        # confirm = input("Execute this SQL? [Y/n]: ").lower()
                        # if confirm == 'y' or confirm == '':
                        print("\n🚀 Executing generated SQL...")
                        execute_sql(conn, generated_sql)
                        # else:
                        #     print("SQL execution cancelled.")
                        #     logging.info("User cancelled execution of generated SQL.")
                else:
                    print("Usage: ask <natural_language_query>")
                    logging.warning("Empty 'ask' command.")
            elif command == "schema":
                 schema_string = get_database_schema_string(conn)
                 print(schema_string)
            elif command == "sql":
                if args_str:
                    execute_sql(conn, args_str)
                else:
                    print("Usage: sql <sql_query>")
                    logging.warning("Empty 'sql' command.")
            else:
                print(f"Unknown command: '{command}'. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nExiting...") # Handle Ctrl+C gracefully
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred in the main loop: {e}")
            print(f"An unexpected error occurred: {e}")

    # --- Cleanup ---
    if conn:
        conn.close()
        logging.info("Database connection closed.")
    print("Goodbye!")

if __name__ == '__main__':
    main()
