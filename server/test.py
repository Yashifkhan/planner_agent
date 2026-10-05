from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

tool = TavilySearchResults(max_results=2)

result = tool.invoke({
    "query": "What is the newest version of ChatGPT?"
})

print(result)