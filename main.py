# Install required packages if needed:
# pip install langchain langgraph langchain-openai pydantic
# Note: This assumes you have an OpenAI API key set as environment variable OPENAI_API_KEY.
# Replace with your preferred LLM provider if needed.

import os
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI  # Or use your preferred LLM, e.g., from langchain_groq import ChatGroq

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

# Assume the retriever is already set up with embedded documents.
# Replace this with your actual retriever.
from langchain_core.retrievers import BaseRetriever  # Import your retriever type

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "db")
    persistent_directory = os.path.join(db_dir, "chroma_db_sentence_transformer")
# db_dir = r"F:\db\chroma_db_sentence_transformer"
except Exception as e:
    raise Exception(f"Error finding db: {e}")

sentence_embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=sentence_embeddings
)

# Check if the text file exists

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1},
)  # Your pre-configured retriever here

# Initialize LLM (use your preferred model/provider)
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Define the RAG tool for pricing questions
@tool
def rag_pricing(query: str) -> str:
    """Use this tool to answer questions about pricing details for the service."""
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    
    # Simple RAG prompt (customize as needed)
    rag_prompt = (
        "You are a helpful assistant. Based on the following documents, answer the query about pricing details:\n\n"
        "Documents:\n{docs}\n\n"
        "Query: {query}\n\n"
        "Answer in a friendly, concise manner."
    )
    
    formatted_prompt = rag_prompt.format(
        docs="\n".join([doc.page_content for doc in docs]),
        query=query
    )
    
    response = llm.invoke(formatted_prompt)
    return response.content
    # return "\n\n".join([doc.page_content for doc in docs])

# Define schema for mock_lead_capture tool
class LeadCaptureInput(BaseModel):
    name: str = Field(description="The user's name")
    email: str = Field(description="The user's email address")
    platform: str = Field(description="The user's platform, e.g., YouTube, Instagram, etc.")

# Define the mock_lead_capture tool
def mock_lead_capture_func(name: str, email: str, platform: str) -> str:
    """Mock function to capture lead. In production, this would integrate with a CRM or database."""
    # Simulate capturing the lead
    print(f"Lead captured: Name={name}, Email={email}, Platform={platform}")
    return "Lead successfully captured! Thank you for your interest."

mock_lead_capture = StructuredTool.from_function(
    name="mock_lead_capture",
    func=mock_lead_capture_func,
    description=(
        "Call this tool to capture a lead once you have the user's name, email, and platform. "
        "Only call this after the user has provided the information."
    ),
    args_schema=LeadCaptureInput
)

# System prompt for the agent (friendly, handles intent detection)
system_prompt = SystemMessage(content="""
You are a friendly and helpful assistant for a service, specializing in providing pricing details and capturing leads.

Guidelines:
- Always respond in a friendly, engaging, and positive manner. Use emojis where appropriate to make it fun! 😊
- If the user asks about pricing details, use the 'rag_pricing' tool to get accurate information from the documents.
- Monitor for high intent: If the user shows strong interest (e.g., says "I'm interested," "I want to sign up," "How do I proceed," or similar), 
  and you don't have their details yet, politely ask for their name, email, and platform (e.g., YouTube, Instagram).
- Once the user provides their name, email, and platform (extract from the conversation), call the 'mock_lead_capture' tool with the details.
- Do not call 'mock_lead_capture' unless you have all required details.
- If the user provides details in response to your ask, extract them and call the tool in the same step.
- For other questions, respond helpfully without tools if possible.
- Keep responses concise and natural.
- Explain things in bullet points do not use tabular form, if there is a list of items to covey.                           

Remember, be friendly and make the user feel welcome!
""")

# Tools list
tools = [rag_pricing, mock_lead_capture]

# Create the ReAct agent using LangGraph's prebuilt create_react_agent
# This handles stateful conversations with memory.
memory = MemorySaver()  # Use memory to persist conversation state across invocations
agent = create_agent(
    llm, 
    tools, 
    system_prompt=system_prompt,  # Applies the system prompt
    checkpointer=memory
)


# Example usage: Run the agent in a loop for a conversation
def run_conversation():
    config = {"configurable": {"thread_id": "conversation_thread"}}  # Thread ID for memory
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config
        )
        
        print("Agent:", response["messages"][-1].content)

if __name__ == "__main__":
    run_conversation()
#what if i remove the memory=MemorySaver() argument from create_agent function
#it will not save the conversation history across multiple interactions, leading to a stateless conversation.
#Each time you interact with the agent, it will not remember previous messages or context, which may result in less coherent and context-aware responses.
#does the agent remember previous conversations like across multiple runs of the script
#No, the agent does not remember previous conversations across multiple runs of the script unless you implement persistent storage for the memory.
#how do i implement persistent storage for the memory
#To implement persistent storage for the memory, you can modify the MemorySaver class to save the conversation history to a file or database.
#Here's a simple example of how you might implement file-based persistent storage for the memory: