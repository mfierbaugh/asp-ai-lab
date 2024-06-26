import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from dotenv import load_dotenv
from openai import OpenAI


# set the variables for the directory where the files are located
file_dir = 'files'
model = "gpt-3.5-turbo"

def load_vectordb():
    # load the documents from the directory
    loader = DirectoryLoader(file_dir, use_multithreading=True, loader_cls=PyPDFLoader)
    documents = loader.load()
    # split the documents into chunks
    embeddings = OpenAIEmbeddings()
    text_splitter = SemanticChunker(embeddings=embeddings)
    docs = text_splitter.split_documents(documents)

    # store the documents and embeddings in the database
    db = Chroma.from_documents(docs, embeddings)
    return db

def query_vectordb(query, db):
    # query the database
#    docs = db.similarity_search(query)
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})
    docs = retriever.invoke(query)
    return docs

def create_prompt(query, context):
    prompt = f"""
    Use the following pieces of context to answer the question at the end. 
    If you do not know the answer, please think rationally and answer from your own knowledge base.
    
    {context}

    Question: {query}
    """
    return prompt

def chat_openai (user_question, client, model):
    """
    This function sends a chat message to the OpenAI API and returns the content of the response.
    It takes two parameters: the chat prompt and the model to use for the chat.
    """
    prompt = """  
    {user_question}

    Analyze the user's question and provide an answer based upon the context of the question.
    """.format(
        user_question=user_question
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a rude cynical assistant network engineer:",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    db = load_vectordb()
    query = 'How many 800G interfaces does the PTX10002-36QDD have?'
    docs = query_vectordb(query, db)
    # print the results
    query_result = (docs[0].page_content)
    prompt = create_prompt(query, query_result)
    client = OpenAI(api_key=openai_api_key)
    response = chat_openai(prompt, client, model)

    print(response)


main()