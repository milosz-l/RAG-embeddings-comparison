import streamlit as st
from langchain.chains import RetrievalQA
from utils import *
import uuid
import sys

# #Update SQL-Lite
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#Creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] =''
if 'OpenAI_Key' not in st.session_state:
    st.session_state['OpenAI_Key'] =''
if 'HF_Key' not in st.session_state:
    st.session_state['HF_Key'] =''
if 'WX_Key' not in st.session_state:
    st.session_state['WX_Key'] =''
if 'WX_End' not in st.session_state:
    st.session_state['WX_End'] =''
if 'WX_Project' not in st.session_state:
    st.session_state['WX_Project'] =''

st.set_page_config(page_title="Knowledge Screening Assistance")
st.title("Knowledge Screening Assistance...")
st.subheader("I can help you in screening process")

st.sidebar.title("API Managment")

st.session_state["OpenAI_Key"] = st.sidebar.text_input("What is your OpenAI API Key?", type="password", key="OpenAI")
st.sidebar.image("./assets/oAIlogo.png", use_column_width=True)

st.session_state['HF_Key'] = st.sidebar.text_input("What is your HuggingFace API Key?", type="password", key="HuggingFace")
st.sidebar.image("./assets/HFlogo.png", use_column_width=True)

st.session_state["WX_Key"] = st.sidebar.text_input("What is your WatsonX API Key?", type="password", key="WatsonX")
st.session_state["WX_End"] = st.sidebar.text_input("What is your WatsonX API Endpoint?", value="https://us-south.ml.cloud.ibm.com")
st.session_state["WX_Project"] = st.sidebar.text_input("What is your WatsonX Project ID?", type="default")
st.sidebar.image("./assets/WXlogo.jpg", use_column_width=True)

def execute_query(pdf, query, option):
    with st.spinner('Wait for it...'):
        #Creating a unique ID, so that we can use to query and potentialy get only the user uploaded documents from vector store
        st.session_state['unique_id']=uuid.uuid4().hex

        #Get LLM
        llm = None
        if option == "OpenAI":
            llm = get_openai_llm(st.session_state['OpenAI_Key'])
        elif option == "HuggingFace":
            llm = get_hf_llm(st.session_state['HF_Key'])
        elif option == "WatsonX":
            llm = get_watsonx_llm(st.session_state['WX_Key'], st.session_state['WX_End'], st.session_state['WX_Project'])
        else:
            st.error("No model found")

        #Create a documents list out of all the user uploaded pdf files
        final_docs_list=create_docs(pdf,st.session_state['unique_id'])

        #Displaying the count of files that have been uploaded
        st.write("*Files uploaded* :"+str(len(final_docs_list)))

        #Chunk the knowledge in documents into managable pieces
        final_docs_chunks=chunk_docs(final_docs_list, chunk_size=500, chunk_overlap=100)

        #Create vectorstore
        vectordb = create_vectorstore(final_docs_chunks)

        #QA Chain
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever())

        #Answer
        st.subheader("Answer:")
        st.write(qa.run(query))
        #Fetch relavant documents from CHROMA
        relavant_docs=vectordb.max_marginal_relevance_search(query, k=3, fetch_k=10)

        #Introducing a line separator
        st.write(":heavy_minus_sign:" * 30)

        #For each item in relavant docs - we are displaying some info of it on the UI
        for item in range(len(relavant_docs)):

            st.subheader("👉 "+str(item+1))

            #Displaying Filepath
            st.write("**File** : "+relavant_docs[item].metadata['name'])

            #Introducing Expander feature
            with st.expander('Show me 👀'):
                #st.info("**Match Score** : "+str(relavant_docs[item][1]))
                summary = relavant_docs[item].page_content
                st.write("**Chunk** : "+summary)

    st.success("Hope I was able to save your time❤️")
    return


def main():
    query = st.text_area("Please paste the question here...",key="1")

    pdf = st.file_uploader("Upload knowledge sources here, only PDF files allowed", type=["pdf"],accept_multiple_files=True)

    option = st.selectbox("Which LLM provider would you like to use?", ("OpenAI", "HuggingFace", "WatsonX"))

    submit=st.button("Help me with the analysis")

    if submit:
        if (option == "OpenAI" and st.session_state["OpenAI_Key"]) or (option == "HuggingFace" and st.session_state["HF_Key"]) or (option == "WatsonX" and st.session_state["WX_Key"] and st.session_state["WX_End"] and st.session_state["WX_Project"]):
            if pdf == []:
                st.error("Please upload the PDF files for knowledge base")
            else:
                try:
                    execute_query(pdf, query, option)
                except Exception as e:
                    st.error(f"Exception occured: {e}")
        else:
            st.error("It looks as though you haven't provided valid API data")



#Invoking main function
if __name__ == '__main__':
    main()