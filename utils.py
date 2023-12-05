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


#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

#Create CHROMA vectorstore
def create_vectorstore(documents, embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"), persist_directory = 'docs/chroma/'):
    # Create the vector store
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory
    )

    return vectordb

# get LLM
def get_bam_llm():
    load_dotenv()
    api_key = os.getenv("GENAI_KEY", None) # from .env
    api_url = os.getenv("GENAI_API", None) # from .env
    creds = Credentials(api_key, api_endpoint=api_url)
    params = GenerateParams(decoding_method="greedy")
    return LangChainInterface(model="google/flan-ul2", params=params, credentials=creds)

def get_openai_llm():
    return OpenAI(model_name="gpt-4-1106-preview")

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