import logging
from logging.handlers import RotatingFileHandler
import sys
from transformers import pipeline
import os
from llama_index.core.postprocessor import NERPIINodePostprocessor
from llama_index.core.schema import TextNode
import getpass
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
log_file = 'CHAT@2.log'
'''LangChain: An Open-Source Framework for Building LLM Applications
> LangChain is an open-source orchestration framework for the development of applications that use Large Language Models (LLMs).
> It's essentially a generic interface for nearly any LLM.
> LangChain streamlines the programming of LLM applications through abstraction.
> LangChain's abstractions represent common steps and concepts necessary to work with language models. These abstractions can be chained together to create applications, minimizing the amount of code required to execute complex Natural Language Processing (NLP) tasks.

Prompts: Instructions for LLMs
> Prompts are the instructions given to an LLM. LangChain's prompt template class formalizes the composition of prompts without the need to manually hardcode context and queries. A prompt template can contain instructions like "Do not use technical terms in your response" or it could include a set of examples to guide its responses or specify an output format.
> Chains: The Core of LangChain Workflows : Chains are the core of LangChain's workflows. They combine LLMs with other components, creating applications by executing a sequence of functions. For example: An application might need to first retrieve data from a website, then summarize the retrieved text, and finally use that summary to answer user-submitted questions. This is a sequential chain where the output of one function acts as the input to the next. Each function in the chain could use different prompts, different parameters, and even different models.
> External Data Sources: Indexes : LLMs may need to access specific external data sources that are not included in their training dataset. LangChain collectively refers to these data sources as indexes. One of the indexes is the Document Loader, used for importing data sources from sources like file storage services. LangChain also supports vector databases and text splitters.
> LangChain Agents (Optional) : LangChain agents are reasoning engines that can be used to determine which actions to take within an application.

Applications of LangChain

LangChain can be used for various NLP applications, including:

1. Enhanced Chatbots: LangChain can provide proper context for chatbots, integrate them with existing communication channels, and leverage APIs for a more streamlined user experience.
2. Efficient Summarization: Build applications that extract key points from lengthy documents or articles using LLM capabilities.
3. Accurate Question Answering: Develop systems that can search for and deliver clear answers to user queries.
4. Data Augmentation: Generate new training data for your LLMs based on existing information, leading to improved performance.
5. Intelligent Virtual Assistants: Create virtual assistants capable of answering questions, searching for information, and even completing tasks online.'''
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
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
ner_pipeline = pipeline(
    'token-classification', 
    model=r'djagatiya/ner-roberta-base-ontonotesv5-englishv4',
    aggregation_strategy='simple'
)
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
    except Exception as e:
        logger.error(f"Error at get_text_chunks function : {e}")
        logging.error(f"Error at get_text_chunks function : {e}")
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        # embeddings =  OpenAIEmbeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        logger.error(f"Error at get_vector_store function : {e}")
        logging.error(f"Error at get_vector_store function : {e}")
        logger.debug(f"Vector store from get_vector_store function : {vector_store} ")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
        prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        logger.error(f"Error at get_conversational_chain function : {e}")
        logging.error(f"Error at get_conversational_chain function : {e}")
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])

def get_entities_from_text(raw_text, ner_pipeline):
    """Extracts entities from text using the provided NER pipeline and displays them in a Streamlit table with color-coded entity groups."""
    try:
        response_NER = ner_pipeline(inputs=raw_text)
        total_entities = len(response_NER)
        st.write(f"Total entities recognized in the document: {total_entities}")
        logger.info(f"Response : {response_NER}")
        # Create a dictionary to map entity_group to its color
        entity_group_colors = {
            "CARDINAL": "red",
            "DATE": "yellow",
            "EVENT": "lightblue",
            "FAC": "red",
            "GPE": "lightpink",
            "LANGUAGE": "darkyellow",
            "LAW": "black",
            "LOC": "lightbrown",
            "MONEY": "lightgreen",
            "NORP": "grey",
            "ORDINAL": "magenta",
            "ORG": "orange",
            "PERCENT": "darkpink",
            "PERSON": "mintcream",  # Using MintCream for better visibility
            "PRODUCT": "cyan",
            "QUANTITY": "lavender",
            "TIME": "apricot",
            "WORK_OF_ART": "teal",
            "micro avg": "gold",
            "macro avg": "maroon",
            "weighted avg": "beige"
        }
        entities = []# Create an empty list to store entities
        for entity in response_NER:
            entity_group = entity['entity_group']
            word = entity['word']
            score = entity['score']
            color = entity_group_colors.get(entity_group, "lightgray")  # Default color for unknown groups
            # HTML to create the colored box
            colored_box_html = f'<div style="background-color:{color}; padding:2px; border-radius:3px;">{entity_group}</div>'
            # Append entity information with colored box HTML
            entities.append([word, colored_box_html, score])
    except Exception as e:
        logger.info(f"Error at get_entities_from_text function : {e}")
        logging.info(f"Error at get_entities_from_text function : {e}")
    # Display table only if there are entities
    return entities if entities else "No"

    
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
    clicked = st.button("Recognise Entity")  # Add a button for entity recognition
    if clicked:  # Execute entity recognition only if the "Recognise Entity" button was clicked
        raw_text = get_pdf_text(pdf_docs)
        entities_table = get_entities_from_text(raw_text, ner_pipeline)  # Replace ner_pipeline with your actual instance
        st.data_editor(entities_table,use_container_width=True,num_rows="fixed")
    # text_entities = st.text_area("Enter Text Here:", height=100)
    # # Call the function to process text and display results
    # if raw_text:
        # get_entities_from_text(text_entities, ner_pipeline)  # Replace ner_pipeline with your actual instance

if __name__ == "__main__":
    main()