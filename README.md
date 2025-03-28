# Global Language Tutor

## Overview

The *Global Language Tutor* is a Streamlit-based web application that enables users to practice and improve their language skills through an interactive chat interface. The project integrates multiple components, including a conversation chain powered by LangChain, error logging via SQLite, and feedback generation using Plotly and an LLM (Chat OpenAI). It also includes specialized tool calls for mistake handling, ensuring that errors are logged in the database, corrected, and used to provide targeted feedback.

## Tech Stack

- *Language*: Python
- *Web Framework*: Streamlit
- *Conversation Orchestration*: LangChain
- *AI Services*: OpenAI API
- *Database*: SQLite
- *Data Visualization*: Plotly

## Architecture

1. *User Interface*: Entry point for users to input messages and receive responses.
2. *Streamlit Frontend*: Renders the UI, manages interactions, and bridges the UI with the backend.
3. *LangChain Backend*: Orchestrates conversation logic, detects errors, and executes correction tools.
4. *OpenAI GPT-4*: Generates responses and detects language errors via API.
5. *Mistake Logging Module*: Logs errors detected by GPT-4 into an SQLite database.
6. *SQLite Database*: Stores user language errors and GPT corrections.
7. *Feedback Generator*: Analyzes user errors from the database and generates text-based feedback with visual insights.
8. *Plotly Visualizations*: Displays interactive charts showcasing error trends and patterns.

## Installation and Setup

### Prerequisites

Ensure you have the following installed:

- Python (>=3.8)
- pip (Python package manager)
- OpenAI API key

### Step-by-Step Guide

1. *Clone the Repository*
   sh
   git clone https://github.com/HR-04/Language_Learning_App.git
   cd Language_Learning_App
   
2. *Create a Virtual Environment* (Recommended)
   sh
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   
3. *Install Dependencies*
   sh
   pip install -r requirements.txt
   
4. *Set Up API Key*
   - Create a .env file in the project root directory.
   - Add the following line with your OpenAI API key:
     sh
     OPENAI_API_KEY=your_api_key_here
     
5. *Run the Application*
   sh
   streamlit run app.py
   
6. *Access the Application*
   - Open your browser and go to http://localhost:8501