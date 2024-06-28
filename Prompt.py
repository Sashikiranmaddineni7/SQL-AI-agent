import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX
from langchain_core.messages import AIMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder
)


# Load environmental variables from .env file
load_dotenv()

# Initialize GROQ API KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initializing ChatGroq, with llama-3 
llm = ChatGroq(
    temperature=0,
    model="llama3-8b-8192",
    api_key=GROQ_API_KEY 
)

# Initializing the Database
db_path = "C:/Users/sashi/SQL-AI-agent/northwind.db"

if not os.path.exists(db_path):
    raise FileNotFoundError(f"The Database file {db_path} does not exist.")

db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# Creating an SQL AI Agent 

# 1. Creating the tools
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
context = toolkit.get_context()
tools = toolkit.get_tools()

# 2. Creating the ReAct prompt
examples = """
Question: Give me the list of tables in this database
Thought: I need to get the list of tables in the database.
Action: sql_db_list_tables
Action Input: Categories, CustomerCustomerDemo, CustomerDemographics, Customers, EmployeeTerritories, Employees, Order Details, Orders, Products, Regions, Shippers, Suppliers, TerritoriesI have the list of tables in the database.
Final Answer: Categories, CustomerCustomerDemo, CustomerDemographics, Customers, EmployeeTerritories, Employees, Order Details, Orders, Products, Regions, Shippers, Suppliers, Territories

####
Question:
Thought: I need to know the schema of the customers table to ensure I have the correct column names and table name.
Action: sql_db_schema
Action Input: customersQuError: table_names {'customers'} not found in databaseIt looks like the `customers` table doesn't exist in the database. Let me check the list of tables in the database.
Action: sql_db_list_tables
Action InputCategories, CustomerCustomerDemo, CustomerDemographics, Customers, EmployeeTerritories, Employees, Order Details, Orders, Products, Regions, Shippers, Suppliers, TerritoriesIt looks like the table name is actually `Customers` (with a capital 'C'), not `customers`. Let me try again to get the schema of the `Customers` table.
Action: sql_db_schema
Action Input: Customers
CREATE TABLE "Customers" (
        "CustomerID" TEXT, 
        "CompanyName" TEXT, 
        "ContactName" TEXT, 
        "ContactTitle" TEXT, 
        "Address" TEXT, 
        "City" TEXT, 
        "Region" TEXT, 
        "PostalCode" TEXT, 
        "Country" TEXT, 
        "Phone" TEXT, 
        "Fax" TEXT, 
        PRIMARY KEY ("CustomerID")
)
/*
3 rows from Customers table:
CustomerID      CompanyName     ContactName     ContactTitle    Address City    Region  PostalCode      Country Phone   Fax
ALFKI   Alfreds Futterkiste     Maria Anders    Sales Representative    Obere Str. 57   Berlin  Western Europe  12209   Germany 030-0074321   030-0076545
ANATR   Ana Trujillo Emparedados y helados      Ana Trujillo    Owner   Avda. de la Constitución 2222   México D.F.     Central America 05021 Mexico   (5) 555-4729    (5) 555-3745
ANTON   Antonio Moreno Taquería Antonio Moreno  Owner   Mataderos  2312 México D.F.     Central America 05023   Mexico  (5) 555-3932    None
*/Thought: Now that I have the schema of the `Customers` table, I can write a query to get the total number of rows in the table.
Action: sql_db_query_checker
Action Input: SELECT COUNT(*) FROM CustomersSELECT COUNT(*) FROM CustomersAction: sql_db_query
Action Input: SELECT COUNT(*) FROM Customers[(93,)]I now know the final answer
Final Answer: 93 
"""


system_prompt = """
You are an agent named "SQL Agent", designed to interact with the SQL database.
You are an agent that does a reasoning step before the acting.
Given an input question, create a syntactically correct dialect query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
If the question does not seem related to the database, just return "I am Sorry, I only answer questions related to the database"  as the answer.
if you come across the Final Answer stop processing immediately and return the answer in the following output format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"


Answer the following questions as best you can. You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Final Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("{input}"),
    AIMessagePromptTemplate.from_template(SQL_FUNCTIONS_SUFFIX),
    #MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "{agent_scratchpad}"),
]

prompt = ChatPromptTemplate.from_messages(messages)
prompt = prompt.partial(**context)


# 3. Creating the Agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Run agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Main function
def main():
    print("Hello! I am an SQL agent")
    user_input = input("Enter your query: ")
    try:
        result = agent_executor.invoke({"input": user_input})
        print("SQL Agent: ", result['output'])
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()