import streamlit as st
from handle_gemini import ask_gemini

st.set_page_config(page_title="RAAG - ASHA", page_icon="🤖")
st.title("RAG bot for ASHA workers💬")
st.text("RAAG-ASHA is essentially a Retrieval Augmented Generated bot trained on govt. issued training material discussing the day to day responsibilities of ASHA workers for quick reference, in any language they want")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me something...")

if user_input:
    # Add user message
    st.session_state.chat_history.append(("user", user_input))
    
    # Get bot response
    answer = ask_gemini(user_input)
    st.session_state.chat_history.append(("bot", answer))

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)
