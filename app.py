from dotenv import load_dotenv
import google.generativeai as generativeai
import os
from pinecone.grpc import PineconeGRPC as Pinecone
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Constants
uploads_location = "user_uploads"
os.makedirs(uploads_location, exist_ok=True)

# Initialize session state
if 'filename_dict' not in st.session_state:
    st.session_state.filename_dict = {}

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Functions
def load_document(doc):
    file_path = os.path.join(uploads_location, f"user_doc{st.session_state.filename_dict[doc.name]}.pdf")
    with open(file_path, "wb") as f:
        f.write(doc.getbuffer())
    loader = PyPDFLoader(file_path)
    pdf_doc = loader.load()
    return pdf_doc

def create_chunks(pdf_doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    pdf_chunks = text_splitter.split_documents(pdf_doc)
    return [chunk.page_content for chunk in pdf_chunks]

def get_embeddings(chunk_data):
    embeddings = []
    for i, chunk in enumerate(chunk_data):
        result = generativeai.embed_content(model="models/text-embedding-004", content=chunk)
        embeddings.append({'text': chunk, 'vector': result['embedding']})
    return embeddings

def pinecone_store(embeddings, i):
    records = []
    for id, embedding in enumerate(embeddings):
        records.append({
            "id": str(id),
            "values": embedding['vector'],
            "metadata": {'text': embedding['text']}
        })
    stats = st.session_state.index.describe_index_stats()
    if "namespaces" in stats and f"ProjectVectorStore{i}" in stats["namespaces"]:
        st.session_state.index.delete(delete_all=True, namespace=f"ProjectVectorStore{i}")
    st.session_state.index.upsert(vectors=records, namespace=f"ProjectVectorStore{i}")
    return st.session_state.index

def get_answer(query, selected_document):
    query_embedding = generativeai.embed_content(model="models/text-embedding-004", content=query)['embedding']
    query_result = st.session_state.index.query(
        vector=query_embedding,
        top_k=5,
        namespace=f"ProjectVectorStore{st.session_state.filename_dict[selected_document]}",
        include_metadata=True
    )
    context = "\n\n".join(match['metadata']['text'] for match in query_result['matches'])
    prompt = (
        "The user will ask a question. The context of information will be provided. "
        "Answer from the given context only. DO NOT ANSWER FROM YOUR KNOWLEDGE OR TRY TO MAKE UP SOME ANSWER. "
        "IF THE ANSWER IS NOT PRESENT IN THE CONTEXT, REPLY 'I DO NOT KNOW'."
    )
    final_query = f"{prompt}\n\nQuestion: {query}\n\nContext: {context}"
    response = st.session_state.model.generate_content(final_query)
    return response.text

# Main function
def main():
    load_dotenv()
    os.environ['Gemini_api_key'] = os.getenv('Gemini_api_key')
    os.environ['pinecone_api_key'] = os.getenv('pinecone_api_key')
    generativeai.configure(api_key=os.environ['Gemini_api_key'])
    st.session_state.model = generativeai.GenerativeModel("gemini-1.5-flash")
    st.session_state.pc = Pinecone(api_key=os.environ['pinecone_api_key'])
    st.session_state.index = st.session_state.pc.Index("project-index")

    # Sidebar for document upload
    with st.sidebar:
        st.subheader("Your document")
        pdf_docs = st.file_uploader("Upload your PDFs here", type=['pdf'], accept_multiple_files=True)
        if pdf_docs is not None and len(pdf_docs) <= 5:
            if st.button("Process document"):
                i = 0
                progress_bar = st.progress(0)
                status_text = st.empty()
                for doc in pdf_docs:
                    status_text.text(f"Creating embeddings for {doc.name} {i + 1}/{len(pdf_docs)}")
                    st.session_state.filename_dict[doc.name] = i
                    pdf_doc = load_document(doc)
                    chunk_data = create_chunks(pdf_doc)
                    embeddings = get_embeddings(chunk_data)
                    st.session_state.index = pinecone_store(embeddings, i)
                    progress = (i + 1) / len(pdf_docs)
                    progress_bar.progress(progress)
                    i += 1
                status_text.text("Embeddings created successfully.")
        else:
            st.warning("Please upload a maximum of 5 PDFs.")

    # Main app
    st.title("Document Query App")

    # Chat interface
    if st.session_state.filename_dict:
        selected_document = st.selectbox("Choose a file to query", list(st.session_state.filename_dict.keys()))

        # Display chat history
        st.subheader("Chat History")
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(f"**You:** {chat['question']}")
            with st.chat_message("assistant"):
                st.write(f"**Assistant:** {chat['answer']}")

        # Chat input
        query = st.chat_input("Enter your query")
        if query:
            # Get answer
            answer = get_answer(query, selected_document)
            
            # Add to chat history
            st.session_state.chat_history.append({"question": query, "answer": answer})
            
            # Display the latest question and answer
            with st.chat_message("user"):
                st.write(f"**You:** {query}")
            with st.chat_message("assistant"):
                st.write(f"**Assistant:** {answer}")
    else:
        st.info("Please upload and process documents to begin querying.")

if __name__ == '__main__':
    main()