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
    def __init__(self, langchain_api_key:str=None, langchain_project:str="rag_multi_query_5_python", encoding:str="cl100k_base"):
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
        if langchain_api_key:
            os.environ['LANGCHAIN_API_KEY'] = langchain_api_key # "<Your Langchain API key so that you can see the trace information.>"
        if langchain_project:
            os.environ["LANGCHAIN_PROJECT"] = langchain_project # 
        self.encoder = tiktoken.get_encoding(encoding)

    def _get_splitter(self, chunk_size:int=512, chunk_overlap:int=128):
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=self._num_tokens)
    
    def _split_doc(self, splitter, docs):
        return splitter.split_documents(docs)
    
    # A function to calculate the number of tokens in a document, using tiktoken encoder
    def _num_tokens(self, doc:str):
        tokens = self.encoder.encode(doc)
        return len(tokens)
    
    def _multiply_question_retrieval_chain(self, llm, n:int=5):
        from langchain.prompts import ChatPromptTemplate

        def _digit_to_words(n:int):
            if n == 1:
                return "one"
            elif n == 2:
                return "two"
            elif n == 3:
                return "three"
            elif n == 4:
                return "four"
            elif n == 5:
                return "five"
            elif n == 6:
                return "six"
            elif n == 7:
                return "seven"
            elif n == 8:
                return "eight"
            elif n == 9:
                return "nine"
            elif n == 10:
                return "ten"
            else:
                return str(n)

        # Multi Query: Different Perspectives
        template = f"""You are an AI language model assistant. Your task is to generate {_digit_to_words(n)} """ + \
        """different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide only these alternative questions, separated by newlines, without any explanation or leading introduction.
        Original question: {question}"""
        prompt_perspectives = ChatPromptTemplate.from_template(template)

        from langchain_core.output_parsers import StrOutputParser

        generate_queries = (
            prompt_perspectives 
            | llm
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )
        from langchain.load import dumps, loads

        def get_unique_union(documents: list[list]):
            """ Unique union of retrieved docs """
            # Flatten list of lists, and convert each Document to string
            flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
            # Get unique documents
            unique_docs = list(set(flattened_docs))
            # Return
            return [loads(doc).page_content for doc in unique_docs]

        # Retrieval chain
        retrieval_chain = generate_queries | self.retriever.map() | get_unique_union
        return retrieval_chain

    @abstractmethod
    def _load_data(self, data_source:str):
        pass

    @abstractmethod
    def _set_retriever(self, docs):
        pass

    @abstractmethod
    def _gen_llm(self):
        pass

    @abstractmethod
    def _gen_prompt(self):
        pass

    @abstractmethod
    def _gen_chain(self):
        pass

    @abstractmethod
    def query(self, question:str):
        pass

class GroqPDFRAG(BasicRAG):
    def __init__(self, pdf_path:str, groq_api_key:str, langchain_api_key:str=None, langchain_project:str="rag_multi_query_5_python", encoding:str="cl100k_base", chunk_size:int=512, chunk_overlap:int=128):
        super().__init__(langchain_api_key=langchain_api_key, langchain_project=langchain_project, encoding=encoding)
        self.docs = self._load_data(data_source=pdf_path)
        self.spliter = self._get_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.splits = self._split_doc(self.spliter, self.docs)
        self.groq_api_key = groq_api_key
        self.llm = self._gen_llm()
        self.embed = OllamaEmbeddings(model="mofanke/acge_text_embedding:latest", num_thread=4)
        self._set_retriever(self.splits)
        self.final_prompt = self._gen_prompt()
        self.final_chain = self._gen_chain()

    def _load_data(self, data_source:str):
        pdf_loader = PyPDFLoader(file_path=data_source, extract_images=True)
        pdf_docs = pdf_loader.load()
        print(f"Loaded {len(pdf_docs)} chunks from {data_source}")
        return pdf_docs
    
    def _gen_llm(self):
        llm = ChatGroq(model="llama3-8b-8192", temperature=0.7, api_key=self.groq_api_key)
        return llm
    
    def _set_retriever(self, docs):
        vectorstore = Chroma.from_documents(documents=docs, embedding=self.embed)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    def _gen_prompt(self):
        # RAG
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        return prompt
    
    def _gen_chain(self):
        from operator import itemgetter
        final_rag_chain = (
            {"context": self._multiply_question_retrieval_chain(self.llm, n=5), 
            "question": itemgetter("question")} 
            | self.final_prompt
            | self.llm
            | StrOutputParser()
        )
        return final_rag_chain
    
    def query(self, question:str):
        return self.final_chain.invoke({"question": question})
    
def main(pdf_path:str):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key is None:
        raise ValueError("GROQ_API_KEY is not set")
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    if langchain_api_key is None:
        raise ValueError("LANGCHAIN_API_KEY is not set")
    rag = GroqPDFRAG(pdf_path, groq_api_key=groq_api_key, langchain_api_key=langchain_api_key, chunk_size=512, chunk_overlap=128)
    question = "What is jailbreak?"
    answer = rag.query(question)
    print({"question": question, "answer": answer})

if __name__ == "__main__":
    # Add argument parser to accept input parameter pdf_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    main(pdf_path=args.pdf_path)