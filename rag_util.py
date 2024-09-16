import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer
import torch





CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),"")
)


class Encoder:
    def __init__(
        self, tokenizer,embeddings: str ,model1:str,model2:str,gpt_model:str
    ):
        self.dist_old = DistanceStrategy.COSINE
        self.db=[]
        self.doc=[]
        self.filepath=[]
        self.radio_b = 1
        self.split = False
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.embedding_function = SentenceTransformer(embeddings)
        # self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embeddings,
            
            # cache_folder=CACHE_DIR,
            model_kwargs={"device": device}
        )
        # model_id = "../llm-chatbot-rag/LaBSE"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer
                                                        # gguf_file= "../llm-chatbot-rag/dictalm2.0-instruct-GGUF//dictalm2.0-instruct.F16.gguf"
                                                        )
        self.model_id2 = model1
        self.model_id3 = model2
        self.gpt_model = gpt_model
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     tokenizer,
        #     local_files_only=True, 
        #     add_prefix_space=False,
        #     use_fast=False
        # )
        
       
class FaissDb:
    def __init__(self,docs, embedding_function,distance):


        self.db = FAISS.from_documents(
        #    docs, embedding_function, distance_strategy=DistanceStrategy.DOT_PRODUCT
            docs,  embedding_function, distance_strategy=distance
        )
       
    def similarity_search(self, question: str, k: int = 3):
        retrieved_docs = self.db.similarity_search(question, k=k)
        return retrieved_docs


def load_and_split_pdfs(encoder,file_paths: list, chunk_size: int = 512):
    loaders = [PyPDFLoader(file_path) for file_path in file_paths]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
  

    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        ""
    ]
    # text_splitter = SemanticChunker(encoder.embedding_function)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=encoder.tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_size/10,
        strip_whitespace=True,
        separators=separators
    )
    docs = text_splitter.split_documents(pages)
        # encoder.split = True

    return docs
