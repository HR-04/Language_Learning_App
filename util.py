from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
import sqlite3
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ToolMessage

load_dotenv()

# Custom chat message history implementation
class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self, messages: List[BaseMessage] = None):
        self.messages = messages or []

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Initialize database (same as before)
def init_db():
    conn = sqlite3.connect('language_errors.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS mistakes
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    native_language TEXT,
                    target_language TEXT,
                    error_sentence TEXT,
                    corrected_sentence TEXT,
                    error_type TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Session history store
store = {}

def get_session_history(session_id: str) -> InMemoryHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

# Update the system prompt with strict tool calling requirements
system_prompt = """
You are a {learning_language} language tutor. Follow these rules STRICTLY:

1. MISTAKE HANDLING (HIGHEST PRIORITY):
   - When error is detected: 
     1. Immediately call log_mistake tool with:
        - error_sentence: Exact erroneous text
        - corrected_sentence: Full corrected sentence
        - error_type: grammar/vocabulary/pronunciation/syntax
     2. Show correction: (Note: [Mistake] → [Correction])
     3. Continue conversation naturally
     
2. Format for corrections:
   (Note: [Mistake] → [Correction])
   [Follow-up question/response]
   ([{native_language} translation])
3. After tool call, ALWAYS provide follow-up

3. RESPONSE STRUCTURE FOR ERRORS:
   (Note: [Mistake] → [Correction])
   [Follow-up in {learning_language}]
   ([{native_language} translation])

4. PROHIBITED ACTIONS:
   - Never mention you're logging errors
   - Never wait for confirmation after correction
   - Never break conversation flow for logging

5. ADAPTATION:
   - Proficiency: {proficiency_level}
   - Scenario: {scenario}
   - Native language: {native_language}
"""

# Enhanced tool definition with strict parameter enforcement
@tool
def log_mistake(
    native_lang: str,
    target_lang: str, 
    error_sentence: str,
    corrected_sentence: str,
    error_type: str
):
    """MANDATORY ERROR LOGGER. Call IMMEDIATELY when detecting mistakes.
    
    Parameters:
    - native_lang: User's native language (e.g., "English")
    - target_lang: Language being learned (e.g., "Spanish")
    - error_sentence: Original incorrect FULL sentence
    - corrected_sentence: FULL corrected sentence
    - error_type: Error category (grammar/vocabulary/pronunciation/syntax)
    """

tools = [log_mistake]

def store_mistake(
    native_lang: str,
    target_lang: str, 
    error_sentence: str,
    corrected_sentence: str,
    error_type: str
):
    """MANDATORY ERROR LOGGER. Call IMMEDIATELY when detecting mistakes.
    
    Parameters:
    - native_lang: User's native language (e.g., "English")
    - target_lang: Language being learned (e.g., "Spanish")
    - error_sentence: Original incorrect FULL sentence
    - corrected_sentence: FULL corrected sentence
    - error_type: Error category (grammar/vocabulary/pronunciation/syntax)
    """
    try:
        conn = sqlite3.connect('language_errors.db')
        c = conn.cursor()
        c.execute("""
            INSERT INTO mistakes 
            (native_language, target_language, error_sentence, corrected_sentence, error_type) 
            VALUES (?, ?, ?, ?, ?)
        """, (native_lang, target_lang, error_sentence, corrected_sentence, error_type))
        conn.commit()
        print(f"✅ Logged mistake: {error_sentence} → {corrected_sentence}")
    except Exception as e:
        print(f"❌ Error logging: {str(e)}")
    finally:
        conn.close()

def create_conversation_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.2,
    ).bind_tools(tools)

    # Create processing chain
    chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: x.get("chat_history", []),
            config_data=lambda x: {
                "native_language": x.get("native_language", ""),
                "learning_language": x.get("learning_language", ""),
                "proficiency_level": x.get("proficiency_level", "Beginner"),
                "scenario": x.get("scenario", "General")
            }
        )
        | prompt
        | llm
    )
    # Add tool response handling
    full_chain = chain | {
        # Return original AI message
        "messages": lambda x: [x],
        # Process tool calls if any
        "tool_responses": lambda x: [
            ToolMessage(
                content="Mistake logged successfully",
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            for tool_call in x.tool_calls
        ]
    }

    return RunnableWithMessageHistory(
        full_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    ).with_config(run_name="ContinuousConversation")