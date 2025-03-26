import streamlit as st
from util import create_conversation_chain, store_mistake
import sqlite3
import uuid

def init_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = create_conversation_chain()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "config" not in st.session_state:
        st.session_state.config = {}
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

init_session_state()

# UI Setup
st.set_page_config(page_title="Language Tutor", page_icon="🗣️")
st.title("🗣️ Language Learning Chatbot")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Lesson Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        native_lang = st.text_input("Native Language", placeholder="English")
    with col2:
        target_lang = st.text_input("Target Language", placeholder="Spanish")
    
    proficiency = st.selectbox("Proficiency", ["Beginner", "Intermediate", "Advanced"])
    scenario = st.selectbox("Scenario", [
        "Restaurant", "Hotel", "Shopping", 
        "Directions", "Social", "Work"
    ])
    
    if st.button("🚀 Start Lesson", use_container_width=True):
        if all([native_lang, target_lang]):
            st.session_state.config = {
                "native_language": native_lang,
                "learning_language": target_lang,
                "proficiency_level": proficiency,
                "scenario": scenario
            }
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
        else:
            st.error("Please specify both languages")

# Chat Display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Update the response processing section
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        # Get AI response
        response = st.session_state.conversation.invoke(
            {"input": prompt, **st.session_state.config},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        print(f"response : {response}")
        
        # Process tool calls first
        if hasattr(response, 'tool_calls'):
            print(f"response Tool calls : {response.tool_calls}")
            for tool_call in response.tool_calls:
                print(f"tool_call : {tool_call}")
                if tool_call['name'] == 'log_mistake':
                    args = tool_call['args']
                    print(f"args : {args}")
                    store_mistake(
                        native_lang=args['native_lang'],
                        target_lang=args['target_lang'],
                        error_sentence=args['error_sentence'],
                        corrected_sentence=args['corrected_sentence'],
                        error_type=args['error_type']
                    )
        
        # Get final response text
        response_content = response.content
        
        # If no content but tool calls, get follow-up
        if not response_content:
            follow_up = st.session_state.conversation.invoke(
                {"input": "Continue the conversation naturally", **st.session_state.config},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            response_content = follow_up.content

        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_content
        })
        st.rerun()

# Update the initial message handling
if not st.session_state.messages and st.session_state.config:
    with st.spinner("Preparing first lesson..."):
        first_msg = st.session_state.conversation.invoke(
            {"input": f"Begin {st.session_state.config['scenario']} scenario", **st.session_state.config},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        print(first_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": first_msg.content
        })
        st.rerun()

# Empty state
if not st.session_state.config:
    st.info("Configure your lesson in the sidebar to begin")

# Error Log Sidebar
with st.sidebar:
    st.header("📝 Error Log")
    if st.button("View Mistakes"):
        try:
            conn = sqlite3.connect('language_errors.db')
            c = conn.cursor()
            c.execute("""
                SELECT timestamp, error_sentence, corrected_sentence, error_type 
                FROM mistakes 
                ORDER BY timestamp DESC 
                LIMIT 10
            """)
            mistakes = c.fetchall()
            
            if mistakes:
                st.subheader("Recent Mistakes")
                for m in mistakes:
                    st.markdown(f"""
                    **Error**: `{m[1]}`  
                    **Correction**: `{m[2]}`  
                    **Type**: {m[3]}  
                    *{m[0]}*
                    """)
            else:
                st.info("No mistakes logged yet")
        except Exception as e:
            st.error(f"Error accessing database: {str(e)}")
        finally:
            conn.close()