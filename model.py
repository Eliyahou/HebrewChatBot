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
        system_message = """Using the information contained in the context 
                            give a detailed answer to the question Please explain your answer
                            You have to think very well before answering
                            the answer must be in hebrew
                           """
        # Please Display the quotes from the context on which the answer was given
       
        prompt_template = f"""[INST] <<SYS>>\n{system_message} context:{context}\n<</SYS>>\n\n{question} [/INST]
 """
        messages = [ {"role": "user", "content": "Hi there I am the User"},
                     {"role": "system","content":  "You are a friendly assistant. You answer questions from users."},
                     {"role": "user", "content": f"""
                    Answer the following question using only the context below. Only include information specifically discussed.
                    question: {question}
                    context: {context}
                    """}
                    # {"role": "assistant", "content": question}   
                            ]   
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
  #structure the prompt into a chat template
#   messages = [
#               {"role": "system", "content": "You are a Language Model trained to answer questions based solely on the provided text.",},
#               {"role": "user", "content": template},
#               ]
                











        self.prompt =  prompt_template
       
        match type_model:
            case 0:
                 chain = load_qa_chain(self.modelchatGpt, chain_type="stuff",verbose=True)
                 answer =  chain.run(input_documents=retrieved_docs, question=question)
            case 1:
                # answer = self.pipe(prompt_template, max_new_tokens=512)
                # answer = self.model1(prompt_template,max_new_tokens=512)
                # answer = self.model1.invoke(prompt_template) 

                answer = self.model2(
                                    messages=messages,
                                    # max_tokens=2000,
                                    # stop=["</s>","[/INST]","/INST",'<|eot_id|>','<|end|>'],
                                    # temperature = 0.1,
                                    # repeat_penalty = 1.4
                                    )



                # with torch.no_grad():
                #     answer = self.model2(messages)
            case 2:  
                 with torch.no_grad():
                     answer = self.model2.create_chat_completion(
                                    messages=messages,
                                    # max_tokens=2000,
                                    # stop=["</s>","[/INST]","/INST",'<|eot_id|>','<|end|>'],
                                    # temperature = 0.75,
                                    # repeat_penalty = 1.4
                                    )
                     answer = answer["choices"][0]["message"]["content"].strip()
            case 3:
                 with torch.no_grad():
                    answer = self.model3.create_chat_completion(
                                    messages=messages,
                                    max_tokens=2000,
                                    stop=["</s>","[/INST]","/INST",'<|eot_id|>','<|end|>'],
                                    temperature = 0.75,
                                    repeat_penalty = 1.4)
                    answer = answer["choices"][0]["message"]["content"].strip()
                
               
 
                # # prompt_template = f"""<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>\n{system_message} context:{context}\n question{question}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"""






                # input_ids = self.tokenizer3.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

                # model_inputs = input_ids.to(self.device)
                # ## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
                # # inputs = self.tokenizer.encode(
                # # prompt_template, add_special_tokens=False, return_tensors="pt"
                # # ).to(self.device)
                # with torch.no_grad():#Specifying no_grad() to my model tells PyTorch that I don't want to store any previous computations, thus freeing my GPU space.
                #     outputs = self.model3.generate(
                #         input_ids=model_inputs,
                #         max_new_tokens=1000,
                #         do_sample=True, 
                #         temperature=0.3
                #     )
                # answer = str(self.tokenizer3.decode(outputs[0]))
                # str1 ='<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'
                # start = answer.find(str1, answer.find(str1)+1)
                # str1length = len('<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>')
                # start = start +  str1length
                # str2length = len(answer[start: ]) - 21
                # answer= answer[start:start + str2length]
               
                # torch.cuda.empty_cache()
                # answer = response[len(input_ids) :]  # remove input prompt from reponse
                # response = response.replace("\n")
                # response = response.replace("<eos>", "")  # remove eos token
                # answer = self.tokenizer.decode(gen_tokens[0])
               
       
       
       
       
        return answer



     
       
           
    #     prompt = PromptTemplate(
    #     input_variables=["system_message", "context", "question"],
    #     template=prompt_template
    # )
    #     formatted_prompt = prompt.format(system_message={system_message}, context={context}, question={question})
       
        # else:
        #     prompt_template=f"""{self.prompt}{self.answer}</s><s>[INST] {question} [/INST]"""
        #     prompt = PromptTemplate(
        #     input_variables=["prompt_cont","answer", "question"],
        #     template=prompt_template
        # )
        #     self.prompt = prompt.format(prompt_cont={self.prompt},answer={self.answer}, question={question})
     
        
        
      