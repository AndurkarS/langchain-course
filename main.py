from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()

tavily = TavilyClient()

@tool
def search(query:str) -> str:
    """
    Tool that searches the web.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    print(f"searching for: {query}")
    return tavily.search(query)


llm = ChatOpenAI(model="gpt-5")
#tools =[search]
tools = [TavilySearch()]
agent = create_agent(llm, tools)

def main():
    print("Hello from langchain-course!")
    response = agent.invoke({"messages":[HumanMessage(content="Search for 3 Job postings on linkedin for QA Lead/SDET with Langchain in Hyderabad, India")]})
    print(response)


if __name__ == "__main__":
    main()
