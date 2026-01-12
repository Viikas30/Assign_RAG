**🤖 AI RAG Agent with Memory (LangChain + Chroma)**
**This project implements a stateful AI agent using LangChain, ChromaDB, and Sentence Transformers.**
**The agent supports RAG (Retrieval-Augmented Generation), tool calling, and conversation memory, and can be integrated with chat platforms like Telegram and WhatsApp.**

**📌 Features**
RAG-based question answering using ChromaDB

Tool calling (e.g., lead capture)

Conversation memory using MemorySaver

Modular ingestion and runtime separation

Easily integratable with messaging platforms

**🚀 Getting Started**
1️⃣ Clone the Repository
```bash
git clone https://github.com/Viikas30/Assign_RAG.git
cd your-repo-name
```
2️⃣ Create a Virtual Environment (Recommended)
**It is highly recommended to use a virtual environment to manage dependencies.**

```bash
# Create the environment
python -m venv .venv
```
**Activate on Linux / macOS**
source .venv/bin/activate
```
# Activate on Windows
.venv\Scripts\activate
```
**3️⃣ Install Requirements**
**Install the necessary libraries using pip:**

```bash
pip install -r requirements.txt
```

**4️⃣ Run the Embedding Ingestion Pipeline**
This step reads the knowledge base text file, splits it into chunks, embeds them, and stores the vectors in ChromaDB.

```bash
python ingestion.py
```
**What this does:**

-    Loads kb.txt

-    Splits text into semantic chunks

-    Generates embeddings using all-MiniLM-L6-v2

-    Persists embeddings in the db/ directory

-    Note: This step only needs to be run once, unless the knowledge base changes.

-5️⃣ Run the Agent
    Once the database is ready, start the agent:

```bash
python main.py
```
-    You can now interact with the agent via the terminal.
-    Type exit to quit.

**Architecture Overview**
-    High-Level Flow

**-User Input → Agent (LangChain create_agent) → Conversation State (Messages + Custom State) → RAG Tool (ChromaDB) → LLM Response
 Why LangChain?**
-    LangChain is utilized for its robust ecosystem, providing:

-    Agent abstraction: Seamlessly combines LLMs, tools, and reasoning.

-    First-class tool calling: Simplified integration with external data sources.

- **State & Memory Management**
1.   Message State (Automatic)
      Conversation history is maintained through the messages key, providing short-term context:

```python
{"messages": [HumanMessage(content=user_input)]}
```
2. Persistent Memory using MemorySaver
      We use a checkpointer to enable long-term persistence across different sessions.
```
python
memory = MemorySaver()

agent = create_agent(
    llm,
    tools,
    system_prompt=system_prompt,
    checkpointer=memory
)
```
-    Persistence: MemorySaver enables conversation state to survive restarts.
     Isolation: Conversations are scoped using a thread_id, ensuring users/sessions remain isolated.

- **WhatsApp Integration (Overview)**
-    WhatsApp does not provide a free native bot API. Integration is typically handled via official Business API providers.

-    Meta WhatsApp Cloud API: The direct hosting solution from Meta.
