import os
import getpass
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import langchain.embeddings
from langchain_community.llms import openai,huggingface_hub
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import openai, huggingface
from langchain_community.chat_models import openai            # import issue to reslove
from langchain_community.vectorstores import faiss
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import openai
# from langchain.llms import openai ,huggingface_hub
# from langchain.chat_models import openai
from htmltemplate import css, bot_template,user_template

def get_docs_text(docs):
    text = ""            # used to store all the text from the pdf
    for doc in docs:     
        reader = PdfReader(doc)   
        '''Object of PdfReader class and initialise with doc object.
        It creates PDF objects that has pages and we will have to read from the pages in the docs.
        And we will be looping through the pages and add text 
        '''
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
    chunk = text_splitter.split_text(text=text)
    return chunk  # returning list of chunks
    
def get_vector_store(text_chunk):
    # embeddings = huggingface.HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large')
    embeddings =  openai.OpenAIEmbeddings()
    vectorstore = faiss.FAISS.from_texts(text=text_chunk,embedding=embeddings)
    return vectorstore
    
def get_conversation_chain(vectorestore):
    llm = openai.ChatOpenAI()
    optional_llm = huggingface_hub.HuggingFaceHub(repo_id="",model_kwargs={"temperature":0.5,"max_length":512})
    memory = ConversationBufferMemory(memory_key="Chat_history",return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorestore.as_retriever() ,
        memory=memory
    )
    return conversation_chain

def handle_user_input(userquestion):
    response = st.session_state.conversation({"question":userquestion})
    # st.write(response)
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.cha_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}"),message.content ,unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}"),message.content ,unsafe_allow_html=True)
    
def main():
    load_dotenv()
    password = os.environ.get("MY_PASSWORD")
    os.environ["OPENAI_API_KEY"] = getpass.getpass()
    st.set_page_config(
        page_title="Chat with multiple documents",
        page_icon=":books:"
        )
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state :
        st.session_state.conversation = None 
    '''
    This only allow us to generate new messages of the conversation.
    It takes history of conversation and return you the next element in the conversation.
    st.session_state makes variable persistance when used.
    st.session_state allow us to not reinitalises the variable in the case where some element in the streamlit events.
    It allow user to protect the variable to reinitalises, as in streamlit wherever any event happens 
    like click a button, upload etc. it reinitalise/reload entire program element or we can say all its varible as st.session_state allow user to protect from this.
    And using this the streamlit know that this variable do not need to reinitalise.
    '''
    if "chat_history" not in st.session_state :
        st.session_state.chat_history = None 
        
    st.header("Chat with multiple documents :books:")
    userquestion = st.text_input("Ask question about your documents")
    username = st.text_input("Add User Name: ")
    botname = st.text_input("Add Bot Name : ")
    if not username:
        username = "Anonymous User"
    if not botname:
        botname = "Julie"
    if userquestion:
        handle_user_input(userquestion)
    st.write(user_template.replace("{{MSG}}",f"Hello {botname}"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}",f"Hello {username}"),unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Your Docs:")
        docs = st.file_uploader(
            "upload your docs and click on 'Process'",
            accept_multiple_files=True
            )
        if st.button("Process"):
            with st.spinner("On the Way / Processing"):
                # get the docs text
                raw_text = get_docs_text(docs)
                # st.write(raw_text)    # printing the raw_text
                
                # get text chunk
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)  # printing list of chunk
                
                # create vector store or vector representation of the text
                vectorstore = get_vector_store(text_chunks)
                
                # crate conversation chain : conversation object
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
        
if __name__ == "__main__":
    main()