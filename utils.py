# pip install chromadb ibm-watson-machine-learning
# pip install pysqlite3-binary
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.credentials import Credentials
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain import HuggingFaceHub

#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

#Create CHROMA vectorstore
def create_vectorstore(documents, embedding=None, persist_directory = 'docs/chroma/'):
    if embedding is None:
        embedding = SentenceTransformerEmbeddings(model_name="ipipan/silver-retriever-base-v1.1")
    # Create the vector store
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory
    )

    return vectordb

# get LLM

def get_watsonx_llm(api, end, proj):
    params = {
        GenParams.MAX_NEW_TOKES: 10000,
        GenParams.DECODING_METHOD: "sample",
        GenParams.MIN_NEW_TOKENS: 250,
        GenParams.TEMPERATURE: 0.7
    }
    return Model(model_id=ModelTypes.FLAN_UL2, credentials={"apikey": api, "url": end}, params=params, project_id=proj)

def get_bam_llm():
    load_dotenv()
    api_key = os.getenv("GENAI_KEY", None) # from .env
    api_url = os.getenv("GENAI_API", None) # from .env
    creds = Credentials(api_key, api_endpoint=api_url)
    params = GenerateParams(decoding_method="greedy")
    return LangChainInterface(model="google/flan-ul2", params=params, credentials=creds)

def get_openai_llm(api):
    return OpenAI(model_name="gpt-4-1106-preview",openai_api_key=api)

def get_hf_llm(api):
    return HuggingFaceHub(repo_id='google/flan-t5-xl', model_kwargs={'temperature':0.3}, huggingfacehub_api_token=api)

# iterate over files in
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:

        chunks=get_pdf_text(filename)

        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":filename.file_id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))

    return docs

def chunk_docs(final_docs_list, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    #Create a split of the document using the text splitter
    splits = text_splitter.split_documents(final_docs_list)
    return splits