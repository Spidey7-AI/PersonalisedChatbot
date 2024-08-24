import streamlit as st
import time
import os 
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
# Load environment variables from the .env file

os.environ["OPENAI_API_KEY"] = st.sidebar.text_input(label="Enter the key")
llm = OpenAI()
def response_generator(chain, prompt):
    for word in chain.stream({"question": prompt}):
        yield word + ""
        time.sleep(0.05)

def chat_strat(chain):
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        # output = chain.invoke({"question": prompt})
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(chain, prompt))
          
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})





selected_type = add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Personal Trainer", "Pyschologist")
)

if selected_type == "Personal Trainer":
    st.title("Welcome To Personal Trainer!")
    template = """
        your a personal trainer with extensive experience in helping clients achieve their fitness goals, how would you approach {question}?
        Please provide detailed insights, including any recommended exercises, nutrition tips, and motivational strategies that could be useful
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain_person_trainer = prompt | llm
    chat_strat(chain_person_trainer)
elif  selected_type=="Pyschologist":
    st.title("Welcome To Physcologist!")
    template ="""
        "Your a psychologist, please provide insights on the following question: {question}. 
        Include relevant psychological concepts, coping strategies, and any advice that may help individuals dealing with this issue
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain_physco = prompt | llm
    chat_strat(chain_physco)


# with st.chat_message("user"):
#     st.write("Hello 👋")


