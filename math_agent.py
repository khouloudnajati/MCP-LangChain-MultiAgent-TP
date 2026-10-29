import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate


async def main():

    project_dir = Path(__file__).resolve().parent
    load_dotenv(project_dir / ".env")

    # ======================
    # LLM GROQ
    # ======================
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-8b-8192",
        temperature=0
    )

    # ======================
    # MCP SERVER
    # ======================
    python_exe = project_dir / ".venv" / "Scripts" / "python.exe"
    math_server = project_dir / "math_server.py"

    client = MultiServerMCPClient({
        "math": {
            "command": str(python_exe),
            "args": [str(math_server)],
            "transport": "stdio"
        }
    })

    tools = await client.get_tools()
    print("Outils :", [t.name for t in tools])

    # ======================
    # PROMPT AGENT
    # ======================
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant qui utilise des outils pour calculer."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # ======================
    # AGENT (MODERNE)
    # ======================
    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    question = "Calcule (17 + 23) * 4"
    print("\nQuestion:", question)

    result = await agent_executor.ainvoke({"input": question})

    print("\nRéponse:", result)


if __name__ == "__main__":
    asyncio.run(main())