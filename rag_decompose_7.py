from abc import ABC, abstractmethod
import bs4, os
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.load import dumps, loads
import tiktoken
import argparse
from operator import itemgetter

def set_env():
    def os_param_not_set(param:str)->bool:
        return os.environ.get(param) is None or os.environ.get(param) == ""
    
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ["LANGCHAIN_PROJECT"] = "rag_decompose_7"
    if os_param_not_set("LANGCHAIN_API_KEY"):
        os.environ['LANGCHAIN_TRACING_V2'] = None
        os.environ['LANGCHAIN_ENDPOINT'] = None
        os.environ["LANGCHAIN_PROJECT"] = None
    if os_param_not_set("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not set.")
    
    if os_param_not_set("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY not set.")
    
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

def get_retriever(docs, index_name:str, k:int=3):

    def _create_pinecone_index(dim:int, index_name:str=index_name):
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone()
        if len([index.name for index in pc.list_indexes().get('indexes') if index.name==index_name]) == 1:
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

def gen_sub_questions(question:str):
    # Decomposition
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. 
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation.
    Generate multiple search queries related to: ```{question}```
    You should provide the THREE your generated sub-questions only (with sequence number), nothing else. Separate the sub-questions without new line."""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    llm = get_llm(temperature=0., model="llama3-8b-8192")
    chain = prompt_decomposition | llm | StrOutputParser() | RunnablePassthrough()
    sub_questions = chain.invoke({"question": question})
    print(sub_questions)
    return sub_questions.strip().split("\n")

def decompose_prompt():
    # Prompt
    template = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """

    decomposition_prompt = ChatPromptTemplate.from_template(template)
    return decomposition_prompt

def get_decompose_chain(docs, k:int=1, index_name:str="rag-decompose-7"):
    from operator import itemgetter
    from langchain_core.output_parsers import StrOutputParser

    llm = get_llm(temperature=0., model="llama3-8b-8192")
    retriever = get_retriever(docs, k=k, index_name=index_name)
    decomposition_prompt = decompose_prompt()
    rag_chain = (
    {"context": itemgetter("question") | retriever, 
    "question": itemgetter("question"),
    "q_a_pairs": itemgetter("q_a_pairs")} 
    | decomposition_prompt
    | llm
    | StrOutputParser())
    return rag_chain

def get_answers(original_question, sub_questions, rag_chain):
    def format_qa_pair(question, answer):
        """Format Q and A pair"""
        
        formatted_string = ""
        formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
        return formatted_string.strip()

    answers = []
    q_a_pairs = ""
    for q in sub_questions:
        answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
        q_a_pair = format_qa_pair(q,answer)
        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
        answers.append(answer)
    
    answer = rag_chain.invoke({"question":original_question,"q_a_pairs":q_a_pairs})
    answers.append(answer)

    return answers

def main(pdf_path:str, question:str):
    set_env()
    splits = split_pdf(pdf_source=pdf_path)
    decompose_chain = get_decompose_chain(docs=splits, k=1, index_name="rag-decompose-7")
    sub_questions = gen_sub_questions(question=question)
    answers = get_answers(question, sub_questions, decompose_chain)
    print({"original_question": question, "sub_questions": sub_questions, "answers": answers[:-1]})
    print({"original_question": question, "answer": answers[-1]})
    # retrieval_chain = 


if __name__ == "__main__":
    # Add argument parser to accept input parameter pdf_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--question", type=str, help="The question you want to ask your PDF file")
    args = parser.parse_args()

    main(pdf_path=args.pdf_path, question=args.question)