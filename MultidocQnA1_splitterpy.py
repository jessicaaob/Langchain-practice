import streamlit as st
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
import PyPDF2

#Function goes through pdfs and extracts and returns a list of all combined text and a list of combined sources 
def read_and_textify(files):
    text = ""
    # sources_list = []
    for file in files:
        pdfReader = PyPDF2.PdfReader(file)
        #print("Page Number:", len(pdfReader.pages))
        # for i in range(len(pdfReader.pages)):
        #   pageObj = pdfReader.pages[i]
        #   text += pageObj.extract_text()
        #   pageObj.clear()
        for page in pdfReader.pages:
           text += page.extract_text()
          # text_extend += text
          # sources_list.extend(file.name + "_page_"+str(i))
    return text

# Some streamlit app set up/ configuration
st.set_page_config(layout="centered", page_title="ESB_Help")
st.header("ESB Help Centre Q&A")
st.write("---")
  
#streamlit file uploader
uploaded_files = st.file_uploader("Upload documents",accept_multiple_files=True, type=["txt","pdf"])
st.write("---")

# outcome after files are uploaded or not
if uploaded_files is None:
  st.info(f"""Upload files to analyse""")
elif uploaded_files:
  st.write(str(len(uploaded_files)) + " document(s) loaded..")

# use pdf function from above to read th euploaded pdfs and output the text and sources lists
  textify_output = read_and_textify(uploaded_files)
# textify_output = read_and_textify(files)

  documents = textify_output
  # sources = textify_output[1]

  # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=[" ", ",", "\n"])
  # docs = text_splitter.split_documents(documents)
    # split into chunks
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
      )
  chunks = text_splitter.split_text(documents)

  # model set up  
  #extract embeddings
  embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["openai_api_key"])
  #vectore with metadata. Here we will store page numbers.

  vStore = Chroma.from_texts(chunks, embeddings) #, metadatas=[{"source": s} for s in sources])
  #pick a model
  model_name = "gpt-3.5-turbo"
  # retriver and number of docs to be used 
    ### could improve this part, character output limits erroring here) ###
  retriever = vStore.as_retriever()
  retriever.search_kwargs = {'k':4}

  # initiate model
  llm = OpenAI(model_name=model_name, openai_api_key = st.secrets["openai_api_key"], streaming=True)
  #model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
  model = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

  # more app config
  st.header("Ask your data")
  user_q = st.text_area("Enter your questions here")
  
  # output printing in the app
  if st.button("Get Response"):
    try:
      with st.spinner("Model is working on it..."):
        result = model({"question":user_q}, return_only_outputs=True)
        st.subheader('Your response:')
        st.write(result['answer'])
        #st.subheader('Source pages:')
        #st.write(result['sources'])
    except Exception as e:
      st.error(f"An error occurred: {e}")
      st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
