import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from dotenv import load_dotenv
load_dotenv()
# Import agent and HumanMessage from main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import agent
from langchain_core.messages import HumanMessage


token = os.getenv("Telegram_bot_token")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Send me a message and I'll reply as your agent.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    # Use Telegram user id as thread id for memory separation
    thread_id = f"telegram_{update.effective_user.id}"
    config = {"configurable": {"thread_id": thread_id}}
    try:
        agent_response = agent.invoke({"messages": [HumanMessage(content=user_message)]}, config)
        # agent_response is a dict with a 'messages' list, get the last message's content
        reply = agent_response["messages"][-1].content
    except Exception as e:
        reply = f"Sorry, there was an error: {e}"
    await update.message.reply_text(reply)

if __name__ == "__main__":
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot is running...")
    app.run_polling()
