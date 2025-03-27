from typing import List, Tuple, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from dotenv import load_dotenv
import sqlite3
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
import pandas as pd
import plotly.express as px

load_dotenv()

# -------------------------------
# In-memory history for conversation
# -------------------------------
class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self, messages: List[BaseMessage] = None):
        self.messages = messages or []

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# -------------------------------
# Initialize SQLite DB for storing mistakes
# -------------------------------

def init_db():
    with sqlite3.connect('language_errors.db') as conn:
        c = conn.cursor()
        c.execute('DROP TABLE IF EXISTS mistakes')
        c.execute('''
            CREATE TABLE mistakes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                native_language TEXT,
                target_language TEXT,
                error_sentence TEXT,
                corrected_sentence TEXT,
                error_type TEXT
            )
        ''')
        conn.commit()

init_db()


# -------------------------------
# Session history store for chain internal history
# -------------------------------
store = {}

def get_session_history(session_id: str) -> InMemoryHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

# -------------------------------
# System prompt for conversation chain
# -------------------------------
system_prompt = """
You are a {learning_language} language tutor. Follow these rules STRICTLY:

1. MISTAKE HANDLING (HIGHEST PRIORITY):
   - When an error is detected:
     1. Immediately call the log_mistake tool with:
        - error_sentence: Exact erroneous text
        - corrected_sentence: Full corrected sentence
        - error_type: grammar/vocabulary/pronunciation/syntax
     2. Show correction: (Note: [Mistaken word] → [Corrected word])
     3. Continue conversation naturally

2. RESPONSE STRUCTURE FOR ERRORS:
   (Note: [Mistaken word] → [Corrected word])\n\n
   [Follow-up in {learning_language}]\n
   ([{native_language} translation])\n

3. ADAPT TO LEARNER'S LEVEL:
   - For beginners, use simple sentences and basic vocabulary.
   - For intermediate learners, use richer vocabulary, varied tenses, and include one idiom.
   - For advanced learners, use complex grammar with cultural references.

4. SCENARIO-BASED TEACHING:
   - Focus the conversation on the {scenario} context.
   - Engage with relevant follow-up questions to sustain a natural dialogue.

5. INITIATE THE CONVERSATION:
   - Start with an engaging opening that suits a {proficiency_level} learner.
"""


# -------------------------------
# Tool to log mistakes into the database
# -------------------------------
@tool
def log_mistake(
    native_lang: str,
    target_lang: str, 
    error_sentence: str,
    corrected_sentence: str,
    error_type: str
):
    """MANDATORY ERROR LOGGER. Call immediately when detecting mistakes."""
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

tools = [log_mistake]

def store_mistake(
    native_lang: str,
    target_lang: str, 
    error_sentence: str,
    corrected_sentence: str,
    error_type: str
):
    """Call the log_mistake tool by passing arguments as a dictionary."""
    log_mistake.run({
        "native_lang": native_lang,
        "target_lang": target_lang,
        "error_sentence": error_sentence,
        "corrected_sentence": corrected_sentence,
        "error_type": error_type
    })

# -------------------------------
# Function to fetch errors and generate interactive feedback with Plotly
# -------------------------------
def get_feedback_with_graph() -> Tuple[str, Optional[object]]:
    conn = sqlite3.connect('language_errors.db')
    c = conn.cursor()
    c.execute("SELECT error_sentence, corrected_sentence, error_type FROM mistakes ORDER BY timestamp DESC LIMIT 50")
    mistakes = c.fetchall()
    conn.close()
    if not mistakes:
        return ("No errors logged yet.", None)
    
    # Create DataFrame from mistakes
    df = pd.DataFrame(mistakes, columns=["error_sentence", "corrected_sentence", "error_type"])
    mistakes_str = "\n".join([
        f"Error: {row['error_sentence']} -> Correction: {row['corrected_sentence']}, Type: {row['error_type']}"
        for _, row in df.iterrows()
    ])
    # Generate feedback prompt and invoke LLM for text feedback
    feedback_prompt = f"""Based on the following list of mistakes made by the user: {mistakes_str}

                            Generate a detailed feedback message only from user errors:
                            - A performance score out of 100\n
                            - A list of errors with their corrections and brief explanations\n
                            - Recommended best practices\n
                            - Practical suggestions to prevent these mistakes in the future\n
                            
                            Keep the response short, concise, and professional. Conclude with a motivational quote, and avoid using a letter format (no salutations or closing remarks).
                        """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    feedback = llm.invoke(feedback_prompt)
    feedback_text = feedback.content

    # Use Plotly to create an interactive donut chart showing error distribution
    error_counts = df['error_type'].value_counts().reset_index()
    error_counts.columns = ['error_type', 'count']
    fig = px.pie(error_counts, names='error_type', values='count',
                 title="Error Distribution", hole=0.4,
                 color_discrete_sequence=px.colors.qualitative.Vivid)
    return (feedback_text, fig)

# -------------------------------
# Create conversation chain
# -------------------------------
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
    chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: x["chat_history"],
            config_data=lambda x: {
                "native_language": x["native_language"],
                "learning_language": x["learning_language"],
                "proficiency_level": x["proficiency_level"],
                "scenario": x["scenario"]
            }
        )
        | prompt
        | llm
    )
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    ).with_config(run_name="StrictErrorHandlingChat")
