#PYTHON 3.12
# ============================
# ENVIRONMENT & SECURITY SETUP
# ============================

# dotenv is used to load secret values (API keys, passwords)
# from a .env file into environment variables.
# This prevents hardcoding secrets in source code.
from dotenv import load_dotenv

# os allows Python to interact with the operating system.
# Here it is used to READ environment variables securely.
import os


# ============================
# LANGCHAIN CORE COMPONENTS
# ============================

# These represent structured chat messages.
# HumanMessage -> user message
# AIMessage -> assistant message
# Used for maintaining conversation history (memory)
from langchain_core.messages import HumanMessage, AIMessage

# Converts the LLM output into a plain string.
# (Not strictly required here, but useful in chains.)
from langchain_core.output_parsers import StrOutputParser

# Used to build structured prompts for the LLM.
# ChatPromptTemplate -> template for messages
# MessagesPlaceholder -> placeholder for dynamic content like history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ============================
# LLM (GEMINI) INTEGRATION
# ============================

# This is LangChain's wrapper around Google Gemini chat models.
# It lets us talk to Gemini using LangChain abstractions.
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================
# TOOL & AGENT SYSTEM
# ============================

# @tool decorator allows us to expose Python functions
# so that the AI agent can CALL them.
from langchain.tools import tool

# AgentExecutor runs the agent loop (think → act → observe → respond)
# create_tool_calling_agent creates an agent that knows how to
# decide when to call tools.
from langchain.agents import AgentExecutor, create_tool_calling_agent


# ============================
# TODOIST API CLIENT
# ============================

# Official Python SDK for Todoist.
# Allows Python to communicate with Todoist servers.
from todoist_api_python.api import TodoistAPI


# ============================
# LOAD ENVIRONMENT VARIABLES
# ============================

# Reads the .env file and loads values into OS environment
load_dotenv()

# Read API keys safely from environment variables
todoist_API_KEY = os.getenv("TODOIST_API_KEY")
gemini_API_KEY = os.getenv("GEMINI_API_KEY")

# Create a Todoist client using the API key
# This object will be used to add / fetch tasks
todoist = TodoistAPI(todoist_API_KEY)


# ============================
# TOOL DEFINITIONS (AI ACTIONS)
# ============================

@tool
def add_task(task, desc=None):
    """
    Adds a new task to the user's Todoist list.

    This tool should be used when the user wants to:
    - add a task
    - create a task
    - remember something
    """

    # Calls Todoist API to create a task
    # 'content' is the task title
    # 'description' is optional
    todoist.add_task(content=task, description=desc)


@tool
def show_task():
    """
    Shows all tasks from the user's Todoist list.

    This tool should be used when the user wants to:
    - see tasks
    - list tasks
    - show reminders
    """

    # Todoist returns tasks in PAGINATED form.
    # Pagination means:
    # Instead of returning ALL tasks at once,
    # the API returns them in smaller chunks (pages)
    # to improve performance and reduce server load.
    #
    # Example:
    # Page 1 -> 50 tasks
    # Page 2 -> next 50 tasks
    # Page 3 -> next 50 tasks

    results_paginator = todoist.get_tasks()

    tasks = []

    # Outer loop: each page of results
    for task_list in results_paginator:

        # Inner loop: each task inside that page
        for task in task_list:

            # Extract only the task name/content
            tasks.append(task.content)

    # Return list of task names to the AI
    return tasks


# List of tools that the agent is allowed to use
tools = [add_task, show_task]


# ============================
# LLM INITIALIZATION
# ============================

# Create a Gemini chat model instance
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",        # Fast + cheap Gemini model
    google_api_key=gemini_API_KEY,   # Authentication
    temperature=0.3                  # Low randomness = stable answers
)


# ============================
# SYSTEM PROMPT (AI BEHAVIOR)
# ============================

# System prompt defines the role, rules, and behavior of the AI.
# This is NOT user input.
system_prompt = """
You are a helpful assistant.
You help the user add tasks.
You help the user show existing tasks.
"""


# ============================
# PROMPT STRUCTURE
# ============================

# This defines how messages are sent to the LLM
prompt = ChatPromptTemplate([
    ("system", system_prompt),             # Rules & behavior
    MessagesPlaceholder("history"),         # Conversation memory
    ("user", "{input}"),                    # Current user input
    MessagesPlaceholder("agent_scratchpad") # Agent reasoning + tool calls
])


# ============================
# AGENT CREATION
# ============================

# Creates an agent that can:
# - understand user input
# - decide whether to call a tool
# - call the tool
# - use the tool result
# - generate a final response
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor runs the agent loop
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False   # Set True to see internal reasoning steps
)


# ============================
# CONVERSATION MEMORY
# ============================

# Stores past messages so the AI can remember context
history = []


# ============================
# MAIN CHAT LOOP
# ============================

while True:
    # Take user input from terminal
    user_input = input("you : ")

    # Send input + history to agent
    response = agent_executor.invoke({
        "input": user_input,
        "history": history
    })

    # Print AI's final response
    print(response["output"])

    # Save conversation to memory
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response["output"]))
