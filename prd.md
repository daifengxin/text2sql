Using LLM as a UX
Approaches
Use LLM as a copilot
Use LLM as a team member 
Use LLM as part of the SW
Exercise
Develop an equivalent to Excel or Google Sheet application where the interaction with the application is a chat interaction
Elements
Sqllite
OpenAI
Pandas
Steps
Step 1:  Load CSV Files into SQLLite
Step 2:  Create Tables Dynamically from CSV
Step 3:  Handle Schema Conflicts
Step 4:  Simulate AI using input (the input to be schemas)
Step 5:  Add AI to generate SQL
Step 1
Objective: Understand the structure of CSV and how it maps to SQL tables.
Activities:
Manually create a table in SQLite.
Use pandas.read_csv() to load data.
Use dataframe.to_sql() to insert data into SQLite.
Run basic queries using sqlite3 or DB browser.
Key SQL Concepts: SELECT, WHERE, LIMIT
Step 2
Objective: Automate table creation by inferring schema from CSV.
Activities:
Write a function to inspect column names and data types.
Generate and execute a CREATE TABLE statement dynamically.
Use pandas and Python string formatting to build SQL.
Key Concepts: Data type mapping (TEXT, INTEGER, REAL)
Step 3
Objective: Learn to build robust systems that validate inputs.
Activities:
Use PRAGMA table_info() to detect existing table schema.
Prompt user on schema conflict: overwrite, rename, or skip.
Implement error logging to a file (error_log.txt).
Key Concepts: Defensive coding, logging, user input control
Step 4
Objective: Create a simple, interactive assistant using Python CLI.
Activities:
Use a loop with input() to simulate chatbot-like interaction.
Allow users to load CSV files, run SQL queries, or exit.
Provide table listing functionality using sqlite_master.
Key Concepts: Control flow, CLI design, user experience
Step 5
Objective: Enable interaction through plain language using ChatGPT or another LLM.
Activities:
Pass table schema and user request to an LLM.
Let AI generate SQL and execute it.
Display results and optionally the generated SQL.
Key Concepts: Prompt engineering, schema context, LLM integration
Examples of a prompt
You are an AI assistant tasked with converting user queries into SQL statements. The database uses SQLite and contains the following tables: - sales (sale_id, product_id, quantity, sale_date, revenue) - products (product_id, product_name, category, price) - employees (employee_id, name, department, hire_date) - customers (customer_id, customer_name, location) User Query: "Show me the top 5 products by total revenue this month." Your task is to: 1. Generate a SQL query that accurately answers the user's question. 2. Ensure the SQL is compatible with SQLite syntax. 3. Provide a short comment explaining what the query does. Output Format: - SQL Query - Explanation
https://github.com/openai/openai-python

When creating your openAI account (PUT budget limits)