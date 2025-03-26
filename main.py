import streamlit as st
from util import create_conversation_chain, store_mistake
import sqlite3
import uuid
from langchain_core.messages import AIMessage, HumanMessage

def init_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = create_conversation_chain()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "config" not in st.session_state:
        st.session_state.config = {}
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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
            st.session_state.chat_history = []
            st.rerun()
        else:
            st.error("Please specify both languages")

# Chat Display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def process_response(response):
    """Helper function to handle different response types"""
    if hasattr(response, 'content'):
        return response.content, getattr(response, 'tool_calls', [])
    elif isinstance(response, dict):
        ai_message = response.get('messages', [{}])[0]
        content = ai_message.get('content', '') if isinstance(ai_message, dict) else getattr(ai_message, 'content', '')
        tool_calls = ai_message.get('tool_calls', []) if isinstance(ai_message, dict) else getattr(ai_message, 'tool_calls', [])
        return content, tool_calls
    return str(response), []

if prompt := st.chat_input("Type your message..."):
    # Add user message to both display and history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    
    with st.spinner("Thinking..."):
        try:
            # Prepare input with all required parameters
            input_data = {
                "input": prompt,
                "native_language": st.session_state.config['native_language'],
                "learning_language": st.session_state.config['learning_language'],
                "proficiency_level": st.session_state.config['proficiency_level'],
                "scenario": st.session_state.config['scenario'],
                "chat_history": st.session_state.chat_history
            }
            
            # Get initial response
            response = st.session_state.conversation.invoke(
                input_data,
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            
            # Process the response
            response_content, tool_calls = process_response(response)
            
            # Process tool calls if any
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call['name'] == 'log_mistake':
                        args = tool_call['args']
                        store_mistake(
                            native_lang=args['native_lang'],
                            target_lang=args['target_lang'],
                            error_sentence=args['error_sentence'],
                            corrected_sentence=args['corrected_sentence'],
                            error_type=args['error_type']
                        )
                
                # Get follow-up response after tool execution
                follow_up = st.session_state.conversation.invoke(
                    {
                        **input_data,
                        "input": "[CONTINUE] Please provide natural follow-up to the corrected sentence"
                    },
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )
                follow_up_content, _ = process_response(follow_up)
                
                # Combine responses
                full_response = f"{response_content}\n{follow_up_content}"
                st.session_state.chat_history.append(AIMessage(content=full_response))
            else:
                full_response = response_content
                st.session_state.chat_history.append(AIMessage(content=full_response))
            
            # Add to display messages
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response
            })
            
        except Exception as e:
            st.error(f"Error in conversation: {str(e)}")
        
        st.rerun()

# Initial lesson message
if not st.session_state.messages and st.session_state.config:
    with st.spinner("Preparing first lesson..."):
        try:
            input_data = {
                "input": f"Begin {st.session_state.config['scenario']} scenario in {st.session_state.config['learning_language']}",
                "native_language": st.session_state.config['native_language'],
                "learning_language": st.session_state.config['learning_language'],
                "proficiency_level": st.session_state.config['proficiency_level'],
                "scenario": st.session_state.config['scenario'],
                "chat_history": []
            }
            
            response = st.session_state.conversation.invoke(
                input_data,
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            
            response_content, _ = process_response(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_content
            })
            st.session_state.chat_history.append(AIMessage(content=response_content))
            st.rerun()
        except Exception as e:
            st.error(f"Error starting conversation: {str(e)}")

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