from abc import ABC, abstractmethod
import bs4, os, re, json
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.load import dumps, loads
import tiktoken
import argparse
from operator import itemgetter

def set_env(project_name:str):
    def os_param_not_set(param:str)->bool:
        return os.environ.get(param) is None or os.environ.get(param) == ""
    
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ["LANGCHAIN_PROJECT"] = project_name
    if os_param_not_set("LANGCHAIN_API_KEY"):
        os.environ['LANGCHAIN_TRACING_V2'] = None
        os.environ['LANGCHAIN_ENDPOINT'] = None
        os.environ["LANGCHAIN_PROJECT"] = None
    if os_param_not_set("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not set.")
    
    if os_param_not_set("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY not set.")
    
def get_index_name(prefix:str, data_source:str):
    index_name = prefix + ":" + ' '.join(re.split(r'[.]', os.path.basename(data_source))[:-1][0].split(' ')[-5:]).lower()
    index_name = re.sub(r'-{1,}', '-', re.sub(r'[^a-z0-9]', '-', index_name))[:45] # the maximum allowed length: 45
    return index_name

def get_llm(temperature:float=.7, model:str="llama3-8b-8192"):
    groq = ChatGroq(api_key=os.environ["GROQ_API_KEY"], temperature=temperature, model=model)
    return groq

def get_embedder(model:str="mofanke/acge_text_embedding:latest"):
    embeddings = OllamaEmbeddings(model=model)
    return embeddings

def split_pdf(pdf_source:str, chunk_size:int=512, chunk_overlap:int=128, encoding:str="cl100k_base"):
    def _load_pdf(data_source:str):
        pdf_loader = PyPDFLoader(file_path=data_source, extract_images=True)
        pdf_docs = pdf_loader.load()
        print(f"Loaded {len(pdf_docs)} chunks from {data_source}")
        return pdf_docs
    
    def _num_tokens(doc:str):
        encoder = tiktoken.get_encoding(encoding)
        tokens = encoder.encode(doc)
        return len(tokens)

    pdf_docs = _load_pdf(pdf_source)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=_num_tokens)
    return splitter.split_documents(pdf_docs)

def get_retriever(docs, k:int=3, index_name:str="rag-fusion-6"):

    def _create_pinecone_index(index_name:str=index_name, dim:int=1024):
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone()
        if sum([index.name==index_name for index in pc.list_indexes().get('indexes')]) == 1:
            return False
        else:
            pc.create_index(name=index_name, dimension=dim, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
            return True

    embedder = get_embedder()
    if _create_pinecone_index(index_name=index_name, dim=len(embedder.embed_documents(['a'])[0])): # new index
        vectorstore = PineconeVectorStore.from_documents(docs, embedding=embedder, index_name=index_name)
    else: # existing index
        vectorstore = PineconeVectorStore.from_existing_index(index_name, embedder)

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

def get_query_enhance_chain():
    llm = get_llm(temperature=1., model="llama3-8b-8192")
    template = """You are an expert having sound world knowledge. Your task is to write a scientific paper passage that can answer the following question.
     Question: {question}
     Passage:"""
    prompt_hyde = ChatPromptTemplate.from_template(template)
    hyde_chain = prompt_hyde | llm | StrOutputParser()
    return hyde_chain

def get_final_rag_chain(query_enhance_chain, retriever):
    template = """Please answer the following question, using the provided contexts.
    Context 1: ```{HyDE Context}```
    
    Context 2: ```{Normal Context}```
    
    Question: {question}
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm(temperature=.7, model="llama3-70b-8192")

    rag_chain = {"HyDE Context": query_enhance_chain | retriever,
     "Normal Context": itemgetter("question") | retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    
    return rag_chain

def main(pdf_path:str, question:str):
    set_env(project_name="rag_hyde_9_python")
    splits = split_pdf(pdf_source=pdf_path)
    index_name = get_index_name(prefix="rag-hyde-9", data_source=pdf_path)
    retriever = get_retriever(docs=splits, k=5, index_name=index_name)
    query_enhance_chain = get_query_enhance_chain()
    # enhanced_question = query_enhance_chain.invoke({"question": question})
    # print(f"Enhanced question: {enhanced_question}")
    final_rag_chain = get_final_rag_chain(query_enhance_chain, retriever)
    answer = final_rag_chain.invoke({"question": question})
    print({"question": question, "answer": answer})


if __name__ == "__main__":
    # Add argument parser to accept input parameter pdf_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--question", type=str, help="The question you want to ask your PDF file")
    args = parser.parse_args()

    main(pdf_path=args.pdf_path, question=args.question)
