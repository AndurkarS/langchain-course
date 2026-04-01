import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def main():
    print("Hello from langchain-course!")


if __name__ == "__main__":
    loader = TextLoader("C:\\Users\\shrir\\LLM-Ollama\\projects\\langchain\\langchain-course\\mediumblog.txt",encoding='utf-8')
    documents = loader.load()

    print('Splitting the document into chunks...')
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f'Number of chunks created: {len(texts)}')

    print('Creating embeddings...')
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    print('Storing embeddings in Pinecone...')
    PineconeVectorStore.from_documents(texts,embeddings,index_name=os.getenv("INDEX_NAME"),pinecone_api_key=os.getenv("PINECONE_API_KEY"))
    print('Embeddings stored successfully in Pinecone!')

