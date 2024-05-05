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
import tiktoken
import argparse

class BasicRAG():
    def __init__(self):
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
        os.environ['LANGCHAIN_API_KEY'] = # "<Your Langchain API key so that you can see the trace information.>"
        os.environ["LANGCHAIN_PROJECT"] = "basic_rag_python"
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _split_doc(self, docs, chunk_size:int=512, chunk_overlap:int=128):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=self._num_tokens)
        return splitter.split_documents(docs)
    
    # A function to calculate the number of tokens in a document, using tiktoken encoder
    def _num_tokens(self, doc:str):
        tokens = self.encoder.encode(doc)
        return len(tokens)

    @abstractmethod
    def gen_embedding(self):
        pass

    @abstractmethod
    def gen_llm(self):
        pass

    @abstractmethod
    def gen_prompt(self):
        pass

    @abstractmethod
    def gen_chain(self):
        pass

    @abstractmethod
    def query(self, question:str):
        pass

class PDFRAG(BasicRAG):
    
    def __init__(self, pdf_path:str, chunk_size:int=512, chunk_overlap:int=128):
        super().__init__()
        self._load_pdf(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def _load_pdf(self, pdf_path:str, chunk_size:int=512, chunk_overlap:int=128):
        pdf_loader = PyPDFLoader(file_path=pdf_path, extract_images=True)
        pdf_docs = pdf_loader.load()
        self.splits = self._split_doc(pdf_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"Loaded {len(self.splits)} chunks from {pdf_path}")

class GroqPDFRAG(PDFRAG):
    def __init__(self, pdf_path: str, groq_api_key:str, chunk_size:int=512, chunk_overlap:int=128):
        super().__init__(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.groq_api_key = groq_api_key
        self.gen_llm()
        self.gen_embedding()
        self.gen_prompt()
        self.gen_chain()

    def gen_llm(self):
        super().gen_llm()
        self.llm = ChatGroq(api_key=self.groq_api_key, temperature=0.7, model_name="llama3-8b-8192")

    def gen_embedding(self):
        super().gen_embedding()
        self.vectorstore = Chroma.from_documents(documents=self.splits, 
                            embedding=OllamaEmbeddings(model="mofanke/acge_text_embedding:latest"))
        n_retrieval = 3
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": n_retrieval})

    def gen_prompt(self):
        super().gen_prompt()
        self.prompt = hub.pull("rlm/rag-prompt")
        
    def gen_chain(self):
        super().gen_chain()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        def output_prompt(prompt):
            with open("/tmp/prompt.txt", "a") as f:
                prompt_contents = dict(dict(prompt).get('messages')[0])['content']
                f.write(prompt_contents + '\n\n')
            return prompt
        
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt | output_prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question:str):
        answer = self.rag_chain.invoke(question)
        return answer



def main(pdf_path:str):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key is None:
        raise ValueError("GROQ_API_KEY is not set")
    rag = GroqPDFRAG(pdf_path, groq_api_key=groq_api_key, chunk_size=512, chunk_overlap=128)
    question = "What is jailbreak?"
    answer = rag.query(question)
    print({"question": question, "answer": answer})

    question = "Please summarize the whole paper?"
    answer = rag.query(question)
    print({"question": question, "answer": answer})

    question = "What's the major point in the \"Failure Modes\" chapter?"
    answer = rag.query(question)
    print({"question": question, "answer": answer})

    question = "Why protected LLM can be jailbroken? Anything we can do better?"
    answer = rag.query(question)
    print({"question": question, "answer": answer})

if __name__ == "__main__":
    # Add argument parser to accept input parameter pdf_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    main(pdf_path=args.pdf_path)