import streamlit as st
from langchain.chains import RetrievalQA
from utils import *
import uuid

#Creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] =''

def main():
    st.set_page_config(page_title="Knowledge Screening Assistance")
    st.title("Knowledge Screening Assistance...")
    st.subheader("I can help you in screening process")

    query = st.text_area("Please paste the question here...",key="1")

    pdf = st.file_uploader("Upload knowledge sources here, only PDF files allowed", type=["pdf"],accept_multiple_files=True)

    submit=st.button("Help me with the analysis")

    if submit:
        with st.spinner('Wait for it...'):

            #Creating a unique ID, so that we can use to query and potentialy get only the user uploaded documents from vector store
            st.session_state['unique_id']=uuid.uuid4().hex

            #Create a documents list out of all the user uploaded pdf files
            final_docs_list=create_docs(pdf,st.session_state['unique_id'])

            #Displaying the count of files that have been uploaded
            st.write("*Files uploaded* :"+str(len(final_docs_list)))

            #Chunk the knowledge in documents into managable pieces
            final_docs_chunks=chunk_docs(final_docs_list, chunk_size=500, chunk_overlap=100)

            #Create vectorstore
            vectordb = create_vectorstore(final_docs_chunks)

            #QA Chain
            qa = RetrievalQA.from_chain_type(llm=get_bam_llm(), chain_type="stuff", retriever=vectordb.as_retriever())

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


#Invoking main function
if __name__ == '__main__':
    main()