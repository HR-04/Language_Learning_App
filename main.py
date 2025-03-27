import streamlit as st
import os
import uuid
from util import create_conversation_chain, store_mistake, get_session_history, get_feedback_with_graph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Set page configuration as the very first Streamlit command
st.set_page_config(page_title="Language Tutor", page_icon="💡", layout="wide")

# -------------------------------
# API Key Input in Sidebar
# -------------------------------
# Ensure an API key entry exists in session_state
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""

with st.sidebar:
    st.header("🔑 API Key")
    # Use the current key as the default value
    api_key = st.text_input("Enter your OpenAI API key:", type="password", value=st.session_state["OPENAI_API_KEY"])
    if api_key:
        st.session_state["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_KEY"] = api_key  # Make it available to the OpenAI library

# If no API key is provided, show a warning and halt further execution.
if not st.session_state["OPENAI_API_KEY"]:
    st.sidebar.error("An OpenAI API key is required to run the app.")
    st.stop()

# -------------------------------
# Existing initialization code
# -------------------------------
def init_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = create_conversation_chain()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "config" not in st.session_state:
        st.session_state.config = {}
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "feedback_rendered" not in st.session_state:
        st.session_state.feedback_rendered = False
    if "lesson_initialized" not in st.session_state:
        st.session_state.lesson_initialized = False

init_session_state()

def generate_feedback():
    """Generate feedback and disable chat input."""
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
    st.session_state.feedback_rendered = True  # Disable further chat input
    st.rerun()

# -------------------------------
# Page header
# -------------------------------
st.title("🌐 Global Language Tutor Chat")
st.markdown("Welcome! Use this chat to practice, learn, and improve with engaging conversations 🌟")

# -------------------------------
# Sidebar: Lesson Settings and Buttons side by side
# -------------------------------
with st.sidebar:
    st.header("⚙️ Lesson Settings")
    col1, col2 = st.columns(2)
    with col1:
        native_lang = st.text_input("Native Language", placeholder="English")
    with col2:
        target_lang = st.text_input("Target Language", placeholder="French")
    proficiency = st.selectbox("Proficiency", ["Beginner", "Intermediate", "Advanced"])
    scenario = st.selectbox("Scenario", [
        "Restaurant Ordering",
        "Hotel Booking",
        "Retail Shopping",
        "Getting Directions",
        "Social Greetings",
        "Workplace Communication",
        "Travel Planning",
        "Event Coordination"
    ])
    st.markdown("---")

    if st.button("Start Lesson 🚀", use_container_width=True):
        if all([native_lang, target_lang]):
            st.session_state.config = {
                "native_language": native_lang,
                "learning_language": target_lang,
                "proficiency_level": proficiency,
                "scenario": scenario
            }
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.feedback_rendered = False  # Reset feedback flag when starting a new lesson
            st.session_state.lesson_initialized = False   # Disable chat input until lesson is initialized
            st.rerun()
        else:
            st.error("Please specify both languages")

    if st.button("Generate Feedback ⚡", use_container_width=True):
        generate_feedback()

# -------------------------------
# Main chat container
# -------------------------------
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "figure" in message and message["figure"] is not None:
                st.plotly_chart(message["figure"], use_container_width=True)

# -------------------------------
# Initialize conversation if no messages and config is set.
# Once the first lesson message is generated, enable chat input.
# -------------------------------
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
        st.session_state.lesson_initialized = True  # Lesson initialized; enable chat input
        st.rerun()

# -------------------------------
# Chat input area: Render only if lesson is initialized and feedback is not rendered
# -------------------------------
if st.session_state.get("lesson_initialized", False) and not st.session_state.get("feedback_rendered", False):
    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        user_msg = HumanMessage(content=user_input)
        get_session_history(st.session_state.session_id).add_messages([user_msg])
        
        with st.spinner("Creating your lesson..."):
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
                print("history messagess: ", get_session_history(st.session_state.session_id).messages)
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

if not st.session_state.config:
    st.info("Configure your lesson in the sidebar to begin")
