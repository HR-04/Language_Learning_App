import streamlit as st
from util import create_conversation_chain, store_mistake, get_session_history, get_feedback_with_graph
import uuid
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

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

# Page configuration and header
st.set_page_config(page_title="Language Tutor", page_icon="🗣️", layout="wide")
st.title("🗣️ Language Learning Chatbot")
st.markdown("Welcome! Use this chat to practice your language skills.")

# Sidebar: Lesson Settings and Buttons side by side
with st.sidebar:
    st.header("Lesson Settings")
    col1, col2 = st.columns(2)
    with col1:
        native_lang = st.text_input("Native Language", placeholder="English")
    with col2:
        target_lang = st.text_input("Target Language", placeholder="French")
    proficiency = st.selectbox("Proficiency", ["Beginner", "Intermediate", "Advanced"])
    scenario = st.selectbox("Scenario", ["Restaurant", "Hotel", "Shopping", "Directions", "Social", "Work"])
    
    st.markdown("---")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
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
    with btn_col2:
        if st.button("Generate Feedback", use_container_width=True):
            feedback_text, feedback_fig = get_feedback_with_graph()
            # Clear previous conversation and show only feedback in chat space
            st.session_state.messages = []
            feedback_msg = AIMessage(content=feedback_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": feedback_text,
                "figure": feedback_fig
            })
            get_session_history(st.session_state.session_id).clear()
            get_session_history(st.session_state.session_id).add_messages([feedback_msg])
            st.rerun()

# Main chat container
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "figure" in message and message["figure"] is not None:
                st.plotly_chart(message["figure"], use_container_width=True)

# Chat input area
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    user_msg = HumanMessage(content=user_input)
    get_session_history(st.session_state.session_id).add_messages([user_msg])
    
    with st.spinner("Thinking..."):
        response = st.session_state.conversation.invoke(
            {"input": user_input, **st.session_state.config},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        print(f"response: {response}")
        
        # Track unique error keys for DB logging (only log once per error)
        logged_errors = set()
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                args = tool_call['args']
                error_key = (args['error_sentence'], args['corrected_sentence'])
                if error_key not in logged_errors:
                    store_mistake(
                        native_lang=args['native_lang'],
                        target_lang=args['target_lang'],
                        error_sentence=args['error_sentence'],
                        corrected_sentence=args['corrected_sentence'],
                        error_type=args['error_type']
                    )
                    logged_errors.add(error_key)
                    # Always create a tool message for each tool_call id
                    tool_response_text = f"Logged mistake: {args.get('error_sentence')} → {args.get('corrected_sentence')}"
                    st.session_state.messages.append({
                    "role": "tool",
                    "content": tool_response_text,
                    "tool_call_id": tool_call['id']
                    })
                tool_msg = ToolMessage(content=tool_response_text, tool_call_id=tool_call['id'])
                get_session_history(st.session_state.session_id).add_messages([tool_msg])
            
            follow_up = st.session_state.conversation.invoke(
                {"input": "Continue the conversation naturally", **st.session_state.config},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            response_content = follow_up.content
        else:
            response_content = response.content

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_content
        })
        assistant_msg = AIMessage(content=response_content)
        get_session_history(st.session_state.session_id).add_messages([assistant_msg])
        st.rerun()

# Initialize conversation if no messages and config is set
if not st.session_state.messages and st.session_state.config:
    with st.spinner("Preparing first lesson..."):
        first_msg = st.session_state.conversation.invoke(
            {"input": f"Begin {st.session_state.config['scenario']} scenario", **st.session_state.config},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": first_msg.content
        })
        initial_msg = AIMessage(content=first_msg.content)
        get_session_history(st.session_state.session_id).add_messages([initial_msg])
        st.rerun()

if not st.session_state.config:
    st.info("Configure your lesson in the sidebar to begin")