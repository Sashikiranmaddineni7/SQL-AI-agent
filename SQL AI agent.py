import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType

# Load environment variables from .env file
load_dotenv()

# Initializing GROQ API KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize ChatGroq
llm = ChatGroq(
    temperature=0,
    model="llama3-8b-8192",
    api_key=GROQ_API_KEY
)

# Initialize the database
db_path = "C:/Users/sashi/SQL-AI-agent/northwind.db"
if not os.path.exists(db_path):
    raise FileNotFoundError(f"The database file {db_path} does not exist.")
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# Creating an SQL Agent
SQL_agent = create_sql_agent(
    llm=llm,
    db=db, 
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Function to process user query
def process_query(query):
    try:
        result = SQL_agent.invoke(query)
        print("\nResult: ", result["output"])
    except Exception as e:
        print("An error occured: ", e)

# Main function
def main():
    print("Hello! I am SQL AI Agent")
    while True:
        user_input = input("\nAsk your question (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        process_query(user_input)

if __name__ == "__main__":
    main()