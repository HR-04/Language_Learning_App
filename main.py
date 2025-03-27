import streamlit as st
import uuid
from util import create_conversation_chain, store_mistake, get_session_history, get_feedback_with_graph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# Streamlit Page Configuration
st.set_page_config(page_title="Language Tutor", page_icon="💡", layout="wide")

# Initialize Session State Variables
def initialize_session():
    """Initialize Streamlit session state variables."""
    st.session_state.setdefault("conversation", create_conversation_chain())
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("config", {})
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("feedback_rendered", False)
    st.session_state.setdefault("lesson_initialized", False)

initialize_session()


# Generate AI Feedback and Disable Chat
def generate_feedback():
    """Fetch user mistakes, generate feedback, and render it in the chat."""
    feedback_text, feedback_fig = get_feedback_with_graph()
    
    # Reset messages and display feedback only
    st.session_state.messages = [{"role": "assistant", "content": feedback_text, "figure": feedback_fig}]
    
    # Clear session history for a fresh start
    session_history = get_session_history(st.session_state.session_id)
    session_history.clear()
    session_history.add_messages([AIMessage(content=feedback_text)])

    # Disable further input after feedback generation
    st.session_state.feedback_rendered = True
    st.rerun()


# Sidebar: Lesson Settings and Controls

with st.sidebar:
    st.header("⚙️ Lesson Settings")

    col1, col2 = st.columns(2)
    with col1:
        native_lang = st.text_input("Native Language", placeholder="English")
    with col2:
        target_lang = st.text_input("Target Language", placeholder="French")

    proficiency = st.selectbox("Proficiency", ["Beginner", "Intermediate", "Advanced"])
    scenario = st.selectbox("Scenario", [
        "Restaurant Ordering", "Hotel Booking", "Retail Shopping",
        "Getting Directions", "Social Greetings", "Workplace Communication",
        "Travel Planning", "Event Coordination"
    ])
    
    st.markdown("---")

    # Start Lesson Button
    if st.button("Start Lesson 🚀", use_container_width=True):
        if native_lang and target_lang:
            st.session_state.config.update({
                "native_language": native_lang,
                "learning_language": target_lang,
                "proficiency_level": proficiency,
                "scenario": scenario
            })
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.feedback_rendered = False
            st.session_state.lesson_initialized = False
            st.rerun()
        else:
            st.error("Please specify both languages")

    # Generate Feedback Button
    if st.button("Generate Feedback ⚡", use_container_width=True):
        generate_feedback()


# Display Chat Messages

st.title("🌐 Global Language Tutor Chat")
st.markdown("Welcome! Use this chat to practice, learn, and improve with engaging conversations 🌟")

chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "figure" in message and message["figure"]:
                st.plotly_chart(message["figure"], use_container_width=True)


# Initialize Conversation on First Lesson

if not st.session_state.messages and st.session_state.config:
    with st.spinner("Preparing first lesson..."):
        first_message = st.session_state.conversation.invoke(
            {"input": f"Begin {st.session_state.config['scenario']} scenario", **st.session_state.config},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        
        st.session_state.messages.append({"role": "assistant", "content": first_message.content})
        get_session_history(st.session_state.session_id).add_messages([AIMessage(content=first_message.content)])
        st.session_state.lesson_initialized = True
        st.rerun()


# Handle User Chat Input
if st.session_state.get("lesson_initialized", False) and not st.session_state.get("feedback_rendered", False):
    user_input = st.chat_input("Type your message...")

    if user_input:
        # Append user message to session
        st.session_state.messages.append({"role": "user", "content": user_input})
        get_session_history(st.session_state.session_id).add_messages([HumanMessage(content=user_input)])

        with st.spinner("Creating your lesson..."):
            response = st.session_state.conversation.invoke(
                {"input": user_input, **st.session_state.config},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )

            # Handle detected mistakes and store them
            logged_errors = set()
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    args = tool_call['args']
                    error_key = (args['error_sentence'], args['corrected_sentence'])
                    
                    # Log the mistake only once
                    if error_key not in logged_errors:
                        store_mistake(
                            native_lang=args['native_lang'],
                            target_lang=args['target_lang'],
                            error_sentence=args['error_sentence'],
                            corrected_sentence=args['corrected_sentence'],
                            error_type=args['error_type']
                        )
                        logged_errors.add(error_key)
                    
                        # Always append a tool message for each tool call, regardless of duplicates
                        tool_response_text = f"Logged mistake: {args['error_sentence']} → {args['corrected_sentence']}"
                        st.session_state.messages.append({
                            "role": "tool",
                            "content": tool_response_text,
                            "tool_call_id": tool_call["id"]
                        })
                    tool_msg = ToolMessage(content=tool_response_text, tool_call_id=tool_call["id"])
                    get_session_history(st.session_state.session_id).add_messages([tool_msg])
                
                # Continue conversation after processing tool calls
                follow_up_response = st.session_state.conversation.invoke(
                    {"input": "Continue the conversation naturally", **st.session_state.config},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )
                response_content = follow_up_response.content
            else:
                response_content = response.content
            # Append AI response to chat
            st.session_state.messages.append({"role": "assistant", "content": response_content})
            get_session_history(st.session_state.session_id).add_messages([AIMessage(content=response_content)])
            st.rerun()

if not st.session_state.config:
    st.info("Configure your lesson in the sidebar to begin")
