import os
import pandas as pd
import random

idk = ["I am not sure...", "I am so sorry, I don't know.", "I am afraid I do not know how to answer this question.", "I don't know. Even though I am an old man, my knowledge is limited. Can you ask me something else?", "I am sorry, I do not know the answer to that. I died over 500 years ago, so there are many things that I don't know. Can you ask me something else?"]

# Read docs
f = open("questions.txt", "r")
docs = f.readlines()
f.close()

g = open("themes.txt", "r")
answerdoc = g.readlines()
g.close()

# Load LLaMa
from llama_cpp import Llama
llm = Llama(model_path="../llama.cpp/models/alpaca.13b.ggmlv3.q8_0.bin", n_ctx=2048)


## CLASS ##
from sentence_transformers import SentenceTransformer, util, models
import torch

class Character:

  def __init__(self, name, image_driver=None):

    self.s_prompt = ""
    self.name = name
    self.image_driver = image_driver
    self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    self.corpus_embeddings = self.embedder.encode(docs, convert_to_tensor=True)


  def search_docs(self, query):
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity

    top_k = min(5, len(docs))
    # print("These are the closest 2 sentences to the given query: \n\n", top_k)

    query_embedding = self.embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    context = [docs[idx] for idx in top_results[1]]
    themes = [answerdoc[idx] for idx in top_results[1]]

    context_n = 2
    final_context = ""

    while len(final_context) < 2000 and len(context) >= context_n:
      context_n += 1
      final_context = ''.join(context[:context_n])

    final_context = [final_context, float(top_results[0][0].cpu().detach().numpy())]

    return final_context[0], themes

  def style_transfer(self, context, themes, question, qa_pairs):
    self.s_prompt = ""
    for q, a in qa_pairs:
        self.s_prompt += f'\n\nPatient: {q}\nCounselor: {a}'
    theme = themes[0]
    self.s_prompt += f'\n\nPatient question: {question}\nLLM prompt: '
    self.f_prompt = f"Below is an instruction that describes a task, paired with some inputs that provide further context. Write a response that appropriately completes the request.\n\n### Instruction:\nDetermine if any parts of the below context are relevant to the below question. If so, concisely summarize the relevant information from the context in bullet points. If the context is not relevant to the question, output 'not relevant.'\n\n### Context:\n{context}\n\n### Question:\n{question}\n\n### Response:\n"


    try :
        print(self.f_prompt)
        self.completion = llm.create_completion(
                                            prompt=self.f_prompt,
                                            temperature=0.3,
                                            max_tokens=512,
                                            top_p=.2,
                                            frequency_penalty=0,
                                            presence_penalty=0,
                                            stop=['Patient:']
                                            )
        # print(self.completion)
        self.completion = self.completion["choices"][0]["text"]
    except:
        print("in except")
        return random.choice(idk)
    # self.completion = self.completion[:self.completion.rfind('.')]+'.'
    self.s_prompt += self.completion +'\n'
    print("Completion: \n\n", self.completion)
    # print(self.f_prompt)
    return self.completion

c = Character(name='xyz')


if __name__ == '__main__':
    while True:
        question = input("What question would you like to ask GenChat? \n Press q to quit\n\n")
        if question == 'q':
           break
        context, themes = c.search_docs(question)
        c.style_transfer(context, themes, question, [])
