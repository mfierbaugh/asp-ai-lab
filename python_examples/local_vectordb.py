
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma


# In this basic example, we take the documents in the files directory, split them into chunks, embed them using an open-source embedding model, load it into Chroma, and then query it.

# set the variables for the directory where the files are located
file_dir = 'files'
embedding_model = 'all-MiniLM-L6-v2'

# load the documents from the directory
loader = DirectoryLoader(file_dir, use_multithreading=True, loader_cls=PyPDFLoader)
documents = loader.load()

# split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=5)
docs = text_splitter.split_documents(documents)

# create the embeddings - using local model all-MiniLM-L6-v2
embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)

# store the documents and embeddings in the database
db = Chroma.from_documents(docs, embeddings)

# query the database
query = 'How many 800G interfaces does the PTX10002-36QDD have?'
docs = db.similarity_search(query)

# print the results
print (docs[0].page_content)
