# Ref: https://github.com/langchain-ai/langchain/blob/master/cookbook/rag_fusion.ipynb?ref=blog.langchain.dev

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
    os.environ["LANGCHAIN_PROJECT"] = "rag_fusion_6_python"
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

def get_retriever(docs, k:int=3, index_name:str="rag-fusion-6"):

    def _create_pinecone_index(index_name:str=index_name, dim:int=1024):
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone()
        if len([index.name==index_name for index in pc.list_indexes().get('indexes')]) == 1:
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

def get_fusion_chain(retriever, n_multiply:int=4, k:int=60):

    def reciprocal_rank_fusion(results: list[list], k:int=k):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
            and an optional parameter k used in the RRF formula """
        
        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}

        # Iterate through each list of ranked documents
        for docs in results:
            # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(docs):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(doc)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)


        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return reranked_results[:k]

    multiply_prompt = hub.pull("langchain-ai/rag-fusion-query-generation")
    multiply_prompt.messages[2].prompt.template = f'OUTPUT ({n_multiply} queries)'
    llm = get_llm(temperature=0, model="llama3-8b-8192")
    multiply_chain = (multiply_prompt | llm | StrOutputParser() | (lambda x: x.split('\n')))
    fusion_chain = multiply_chain | retriever.map() | reciprocal_rank_fusion
    return fusion_chain

def get_rag_chain(retrieval_chain):
    llm = get_llm(temperature=1, model="llama3-8b-8192")
    template = """Answer the following question based on this context:

```
{context}
```

Question: {question}
"""
    rag_prompt = ChatPromptTemplate.from_template(template)
    final_rag_chain = (
        {"context": retrieval_chain, 
        "question": itemgetter("question")} 
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return final_rag_chain

def main(pdf_path:str, question:str):
    set_env()
    splits = split_pdf(pdf_source=pdf_path)
    retrieval_chain = get_fusion_chain(retriever=get_retriever(splits), n_multiply=3, k=5)
    final_rag_chain = get_rag_chain(retrieval_chain)
    answer = final_rag_chain.invoke({"original_query": question, "question": question})
    print({"question": question, "answer": answer})


if __name__ == "__main__":
    # Add argument parser to accept input parameter pdf_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--question", type=str, help="The question you want to ask your PDF file")
    args = parser.parse_args()

    main(pdf_path=args.pdf_path, question=args.question)
