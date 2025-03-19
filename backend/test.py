from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Browser, BrowserConfig
from pydantic import SecretStr
import os
from dotenv import load_dotenv
load_dotenv()
import asyncio

api_key = os.getenv("GEMINI_API_KEY")
print(api_key)


browser = Browser(
    config=BrowserConfig(
        # Specify the path to your Chrome executable
        chrome_instance_path='/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',  # macOS path
        # For Windows, typically: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
        # For Linux, typically: '/usr/bin/google-chrome'
    )
)

# Initialize the model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=SecretStr(os.getenv('GEMINI_API_KEY')))

# Create agent with the model
async def main():
    agent = Agent(
        task="",
        llm=llm,
        browser=browser,
    )

    result = await agent.run()
    print(result)


asyncio.run(main())