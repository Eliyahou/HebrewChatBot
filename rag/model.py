import torch
from llama_cpp import Llama
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

class ChatModel:
    def __init__(self,encoder):
        self.conversation_history=""
        self.answer=""
        self.prompt=""
        self.tokenizer = encoder.tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#-------------------------------------------------------------------------------------------    
        if encoder.gpt_model !=None:
            model_name = encoder.gpt_model
            self.modelchatGpt = ChatOpenAI(model_name=model_name)

        n_gpu_layers=-1  # Metal set to 1 is enough.
        n_gpu_layers=50  # Metal set to 1 is enough.
        n_batch = 256  
        if encoder.model_id2 !=None:
            self.model2 = Llama(
                temperature=0.7
                ,
                device='auto',
                n_threads=10,
                max_tokens=2000,
                top_p=0.5,
                # use_mlock = True,
                # top_k=150,
                # tokenizer=self.tokenizer,
                # model_path='./model_files/llama-2-13b-chat.Q4_K_S.gguf',
                model_path=encoder.model_id2,
                # /dictalm2.0-instruct-GGUF/dictalm2.0-instruct.Q4_K_M.gguf",
                repeat_penalty=1.7,
                # model_path = "./dictalm2.0-instruct-GGUF/dictalm2.0-instruct.Q4_K_M.gguf",
                n_gpu_layers=0,
                n_batch=n_batch,
                n_ctx=4000, # context window. By default 512
                f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
                # callback_manager=callback_manager1,
                echo = True,
                verbose=True, # Verbose is required to pass to the callback manager
            )
        
    
        if encoder.model_id3 !=None:
            self.model3 = Llama(
                temperature=0.7,
                n_threads=10,
                max_tokens=2000,
                top_p=0.5,
                device='cpu',
                # tokenizer=self.tokenizer,
                # top_k=150,
                # model_path='./model_files/llama-2-13b-chat.Q4_K_S.gguf',
                model_path=encoder.model_id3,
                # /dictalm2.0-instruct-GGUF/dictalm2.0-instruct.Q4_K_M.gguf",
                repeat_penalty=1.7,
                # model_path = "./dictalm2.0-instruct-GGUF/dictalm2.0-instruct.Q4_K_M.gguf",
                n_gpu_layers=33,
                n_batch=n_batch,
                stop=["<|im_end|>","question:","### question:","###<user>","</user>"],
                n_ctx=4000, # context window. By default 512
                f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
                # callback_manager=callback_manager2,
                echo = True,
                verbose=True, # Verbose is required to pass to the callback manager
            )






    def generate(self, question: str, retrieved_docs: str = None, max_new_tokens: int = 1000,type_model:int=0):
#-------------------------------------------------------------------------------------------           
        context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
        template = f""""Answer the question based only on the following context:
                [context]
                {context}
                [end of context]

                Question: {question}
                נא לענות על השאלה בעברית
                """
        
        messages = [ {"role": "user", "content":"שלום לך!"},
              {"role": "assistant", "content": "You are a Language Model trained to answer questions based solely on the provided text. \
              you are answering only in hebrew language. please provide precise answer.\
              Please explain your answer",},
              {"role": "user", "content": template},
              ]
  
        match type_model:
            case 0:
                 chain = load_qa_chain(self.modelchatGpt, chain_type="stuff",verbose=True)
                 answer =  chain.run(input_documents=retrieved_docs, question=question)
            case 2:  
                 with torch.no_grad():
                     answer = self.model2.create_chat_completion(
                                    messages=messages)
                     answer = answer["choices"][0]["message"]["content"].strip()
            case 3:
                 with torch.no_grad():
                    answer = self.model3.create_chat_completion(
                                    messages=messages)
                    answer = answer["choices"][0]["message"]["content"].strip()
                
               
        return answer

