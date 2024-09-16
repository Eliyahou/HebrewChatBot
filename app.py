import os
from dotenv import load_dotenv



from model import ChatModel
import rag_util
import streamlit as st
from streamlit_modal import Modal

from langchain_community.vectorstores.utils import DistanceStrategy

modal = Modal(
    "The Context", 
    key="demo-modal",
    padding=10,    
    max_width=900  
)

FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)
load_dotenv()
@st.cache_resource
def handler():

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    embedding = os.getenv("EMBEDDING_MODEL")
    model1    = os.getenv("MODEL_1")
    model2    = os.getenv("MODEL_2")
    gpt_model = os.getenv("MODEL_GPT")
    return embedding,model1,model2,gpt_model

radio_b=1


st.title("Hebrew ChatBot")




@st.cache_resource
def load_encoder():
    embedding,model1,model2,gpt_model = handler()
    encoder = rag_util.Encoder(
        tokenizer= embedding,
        embeddings=embedding,
        model1 = model1,
        model2 = model2,
        gpt_model = gpt_model
        
    )
    return encoder

@st.cache_resource     
def load_model():
    model = ChatModel(encoder)
    return model

encoder = load_encoder()
model = load_model()  # load our models once and then cache it



def save_file(uploaded_file):
    """helper function to save documents to disk"""
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


with st.sidebar:
    
    option = st.selectbox(
    "***DistanceStrategy***",
    ("Cosine", "Dot Product", "Euclidan Distance","JACCARD","Max Inner Product"),
    )

    match option:
        case "Cosine":
            dist = DistanceStrategy.COSINE
        case "Dot Product":
            dist = DistanceStrategy.DOT_PRODUCT
        case "Euclidan Distance":
            dist = DistanceStrategy.EUCLIDEAN_DISTANCE
        case "JACCARD":
            dist = DistanceStrategy.JACCARD
        case "Max Inner Product":
            dist = DistanceStrategy.MAX_INNER_PRODUCT    
              
    if encoder.gpt_model != None:
        gpt=":cyclone:[ChatGpt]"
    else: 
        gpt =None   
    if  encoder.model_id2!= None:
        model_1 =  "***Dicta-quantization Q4_K_M***"  
    else:
        model_1 = None

    if  encoder.model_id3!= None:  
        model_2= "***Premise Aya Model Q5_K_M***"  
    else:    
        model_2=None
    genre = st.radio(
        "***Which Kind Of Model Do You Prefer***",
        [gpt, model_1,model_2],
        index=0
        )
    match genre:
        case ":cyclone:[ChatGpt]":
            encoder.radio_b=0
        case  "***Dicta-quantization F16***":  
            encoder.radio_b=2
        case "***Premise Aya Model***":  
             encoder.radio_b=3  
   
    max_new_tokens = st.number_input("max_new_tokens", 128, 4096,512) 
    k = st.number_input("k", 1,10,6)
    uploaded_files = st.file_uploader(
        "Upload PDFs for context", type=["PDF", "pdf"], accept_multiple_files=True
)
    file_paths = []
    for uploaded_file in uploaded_files:
        file_paths.append(save_file(uploaded_file))
    if uploaded_files != []:
        if encoder.filepath != file_paths or dist !=  encoder.dist_old :
            ChatModel.prompt=""
            ChatModel.answer=""
            encoder.filepath=file_paths
            encoder.doc = rag_util.load_and_split_pdfs(encoder,file_paths,chunk_size=max_new_tokens)
            encoder.db = rag_util.FaissDb(encoder.doc,embedding_function=encoder.embedding_function,distance=dist)
            encoder.dist_old = dist
        

def reply_again_cb():
    st.session_state.repeat = True
    with modal.container():
            st.write(context)

           

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Please Ask Your Question!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        user_prompt = st.session_state.messages[-1]["content"]
        retrieved_docs = (
            None if uploaded_files == [] else  encoder.db.similarity_search(user_prompt, k=k)
        )
       
        context = "\n".join("#" + str(i+1) + "--" + "\n" + retrieved_docs[i].page_content + "\n"   for i in range(len(retrieved_docs)))
       
        answer = model.generate(
            user_prompt, retrieved_docs=retrieved_docs, max_new_tokens=max_new_tokens,type_model=encoder.radio_b
        )
        
    response = st.write(answer)
    open_modal = st.button("Show The Context",on_click=reply_again_cb)
    st.session_state.messages.append({"role": "assistant", "content": answer})
   
  
