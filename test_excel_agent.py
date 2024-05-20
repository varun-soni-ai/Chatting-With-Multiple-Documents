import logging
from logging.handlers import RotatingFileHandler
import sys
import streamlit as st
from langchain.agents import create_xml_agent
from langchain.llms import openai
from langchain import hub
from langchain_community.chat_models import ChatAnthropic
from langchain.agents import AgentExecutor, create_xml_agent
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
log_file = 'chat-csv-log-1.log'
def setup_logger(log_file):
    logger = logging.getLogger(__name__)# Create a logger
    logger.setLevel(logging.DEBUG)  # Set the logging level
    # Create a rotating file handler
    rotating_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)  # Max file size 1MB, keep up to 5 backup files
    rotating_handler.setLevel(logging.DEBUG)  # Set the logging level for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')# Create a formatter
    rotating_handler.setFormatter(formatter)# Add the formatter to the handler
    logger.addHandler(rotating_handler)# Add the rotating file handler to the logger
    return logger
logger = setup_logger(log_file)
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
def get_conversational_chain(user_question,csv_file):
    prompt_template = """
    You are working with a pandas dataframe in Python. The name of the dataframe is df. You should use the tools below to answer the question posed of you:\n
    {python_repl_ast}: A Python shell. Use this to execute python commands. Input should be a valid python command. When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer.\n
    Use the following format:\n
    Question: the input question you must answer Thought: you should always think about what to do Action: the action to take, should be one of [python_repl_ast] Action Input: the input to the action Observation: the result of the action ... (this Thought/Action/Action Input/Observation can repeat N times) Thought: I now know the final answer Final Answer: the final answer to the original input question\n
    This is the result of print(df.head()): {df}\n
    Begin! Question: {input} {agent_scratchpad}\n
    """
    try:
        agent = create_xml_agent(openai(),prompt=prompt_template)
        agent_executor = AgentExecutor(agent=agent)
        agent_executor.invoke({"input": {user_question}})
        # create_xml_agent(openai(temperature=0),csv_file,verbose=True)
        st.dataframe(agent)
    except Exception as e:
        logger.error(f"Error at get_conversational_chain function : {e}")
        logging.error(f"Error at get_conversational_chain function : {e}")
    return agent.run(user_question)

def main():
    st.set_page_config("Chat CSV")
    st.header("Chat with CSV 💁")
    user_question = st.text_input("Ask a Question from the CSV Files")
    with st.sidebar:
        st.title("Menu:")
        csv_file = st.file_uploader("Upload your excel files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # df = pd.read_excel(csv_file)
                # st.data_editor(df)
                response = get_conversational_chain(user_question,csv_file=csv_file)
                st.write(response)
                st.success("Done")
if __name__ == "__main__":
    main()