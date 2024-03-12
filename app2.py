import json
import logging
import getpass
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import langchain.embeddings
# from langchain_community.llms import openai,huggingface_hub
# from langchain_community.llms import openai as llms_openai , huggingface_hub
# from langchain_community.llms.openai
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import openai, huggingface
# from langchain.embeddings import openai as embeddings_openai, huggingface
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import openai            # import issue to reslove
from langchain_community.vectorstores import faiss
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import openai
# from langchain.llms import openai ,huggingface_hub
# from langchain.chat_models import openai
from htmltemplate import css, bot_template,user_template

import logging
from logging.handlers import RotatingFileHandler
def setup_logger(log_file):
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the logging level
    
    # Create a rotating file handler
    rotating_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)  # Max file size 1MB, keep up to 5 backup files
    rotating_handler.setLevel(logging.DEBUG)  # Set the logging level for the handler
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Add the formatter to the handler
    rotating_handler.setFormatter(formatter)
    
    # Add the rotating file handler to the logger
    logger.addHandler(rotating_handler)
    
    return logger
log_file = 'chatPDF.log'
logger = setup_logger(log_file)
def setup_logger(log_file):
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the logging level
    
    # Create a rotating file handler
    rotating_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)  # Max file size 1MB, keep up to 5 backup files
    rotating_handler.setLevel(logging.DEBUG)  # Set the logging level for the handler
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Add the formatter to the handler
    rotating_handler.setFormatter(formatter)
    
    # Add the rotating file handler to the logger
    logger.addHandler(rotating_handler)
    
    return logger

def get_docs_text(docs):
    text = ""            # used to store all the text from the pdf
    for doc in docs:     
        reader = PdfReader(doc)   
        for page in reader.pages:             
            text += page.extract_text() # concat the text 
    return text       # Single String with all the content in the docs
        
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',         
        chunk_size=1000,
        chunk_overlap=200,     # handling Knowledge loss with chunk overlapping the text 
        length_function=len
        )
    logger.debug(f"Get text chuck from get_text_chunks function \n {text_splitter.split_text(text=text)} \n DUNE")
    # chunk = text_splitter.split_text(text=text)
    # return chunk
    return text_splitter.split_text(text=text)  # returning list of chunks
    
def get_vector_store(text_chunk):
    # embeddings = huggingface.HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large')
    embeddings =  OpenAIEmbeddings()
    vectorstore = faiss.FAISS.from_texts(texts=text_chunk,embedding=embeddings)
    logger.debug(f"Vector store from get_vector_store function : {vectorstore} ")
    return vectorstore


def get_conversation_chain(vectorestore):
    llm = ChatOpenAI()
    # The line `optional_llm =
    # huggingface_hub.HuggingFaceHub(repo_id="",model_kwargs={"temperature":0.5,"max_length":512})` is
    # creating an instance of the `HuggingFaceHub` class from the `huggingface_hub` module.
    # optional_llm = huggingface_hub.HuggingFaceHub(repo_id="",model_kwargs={"temperature":0.5,"max_length":512})
    memory = ConversationBufferMemory(memory_key="Chat_history",return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectorestore.as_retriever(),memory=memory
    )
    logger.info(f"get_conversation_chain Function : DONE ")
    return conversation_chain

def handle_user_input(input_data):
    """
    The function `handle_user_input` processes user input, generates a response using a conversation
    model, and displays the conversation history in an alternating user-bot format.
    
    :param input_data: The `input_data` parameter is a dictionary containing the user input data. It
    should have the following keys:
    """
    logger.info(f"DIDI : {input_data}")
    logger.info(f"DIDI type: {type(input_data)}")
    try:
        if isinstance(input_data, str):
            # Check if the input string is empty
            if not input_data.strip():
                raise ValueError("Input data is empty.")
            
            # Parse the input_data string into a dictionary
            input_data = json.loads(input_data)

        question = input_data.get('question')
        chat_history = input_data.get('chat_history', [])
        
        if not question:
            raise ValueError("Question is missing in the input data.")
        
        response = st.session_state.conversation({"question": question, "chat_history": chat_history})
        logger.info(f"Response : {response}")
        st.write(response)
        st.session_state.chat_history = response.get('chat_history', [])
        for i, message in enumerate(st.session_state.chat_history):  # Corrected typo in variable name
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        print(f"Exception at handle_user_input: {e}")



def main():
    # st.set_page_config(page_title="Chat with multiple documents", page_icon=":books:")
    # log_file = 'chatPDF.log'
    # logger = setup_logger(log_file)
    # load_dotenv()
    # st.write(css,unsafe_allow_html=True)
    # if "conversation" not in st.session_state :   
    #     st.session_state.conversation = None
    # if "chat_history" not in st.session_state :
    #     st.session_state.chat_history = None
    # logger.info(f"type check  : {type(st.session_state.conversation)}")
    # logger.error(f"kiki : {type(st.session_state.chat_history)}")
    # logger.info(f"hi : {st.session_state.chat_history}")
    # logger.error(f"ji ji : {st.session_state.conversation}")
    # st.header("Chat with multiple documents :books:")
    # userquestion = st.text_input("Ask question about your documents")
    # logger.info(f"UserQuestion and its type : {userquestion}  and \n Type : {type(userquestion)}")
    # if userquestion:
    #     try :
    #         handle_user_input(userquestion)
    #     except Exception as e:
    #         print(f"Exception in userquestion : {userquestion}")
    #     # if userquestion:
    # #     try :
    # #         handle_user_input(userquestion)
    # #     except Exception as e:
    # #         print(f"Exception in userquestion : {userquestion}")
    # with st.sidebar:
    #     st.subheader("Your Document")
    #     docs = st.file_uploader(
    #         "upload your docs and click on 'Process'",
    #         accept_multiple_files=True
    #         )
    #     if st.button("Process"):
    #         with st.spinner():
    #             # get the docs text
    #             raw_text = get_docs_text(docs)
    #             # st.write(raw_text)    # printing the raw_text
                
    #             # # get text chunk
    #             text_chunks = get_text_chunks(raw_text)
    #             st.write(text_chunks)  # printing list of chunk
                
    #             # # create vector store or vector representation of the text
    #             vectorstore = get_vector_store(text_chunks)
    #             # st.write(vectorstore)
    #             # # crate conversation chain : conversation object
    #             st.session_state.conversation = get_conversation_chain(vectorstore)
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                    page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question) 
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_docs_text(pdf_docs)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # create vector store
                vectorstore = get_vector_store(text_chunks)
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                    
        
if __name__ == "__main__":
    main()