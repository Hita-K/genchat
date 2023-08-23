import os
import pandas as pd
import random
import torch

from sentence_transformers import SentenceTransformer, util, models
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import LlamaCPP

idk = ["I am not sure...", "I am so sorry, I don't know.", "I am afraid I do not know how to answer this question.", "I don't know. Even though I am an old man, my knowledge is limited. Can you ask me something else?", "I am sorry, I do not know the answer to that. I died over 500 years ago, so there are many things that I don't know. Can you ask me something else?"]

# Load LLaMa
from llama_cpp import Llama
# llm = Llama(model_path="../llama.cpp/models/alpaca.13b.ggmlv3.q8_0.bin", n_ctx=2048)

## CLASS ##

class Character:

  def __init__(self, name, docs_fp='questions.txt', answerdoc_fp='themes.txt'):
    self.s_prompt = ""
    self.name = name

    # Load docs
    self.docs_fp = docs_fp
    with open(docs_fp) as f:
      self.docs = f.readlines()

    with open(answerdoc_fp) as g:
      self.answerdoc = g.readlines()

    # LlamaIndex stuff
    self.llm = LlamaCPP(
        model_path="../llama.cpp/models/alpaca.13b.ggmlv3.q8_0.bin",
        context_window=2048,
        )
    self.service_context = ServiceContext.from_defaults(llm=self.llm)
    self.build_vector_index()

  def search_docs(self, query):
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    self.corpus_embeddings = self.embedder.encode(self.docs, convert_to_tensor=True)

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


  def build_vector_index(self):
    documents = SimpleDirectoryReader(input_files=[self.docs_fp]).load_data()
    self.index = VectorStoreIndex.from_documents(documents, service_context=self.service_context)


  def query_vector_index(self, query):
    retriever = self.index.as_retriever(similarity_top_k=5)
    top_results = retriever.retrieve(query)
    combined_context = ''
    i = 0

    while len(combined_context) < 1800:
      combined_context += top_results[i].node.text
      i += 1
    ## TODO retrieve real themes
    dummy_themes = ['foo' for foo in top_results]

    return combined_context, dummy_themes

  def style_transfer(self, context, themes, question, qa_pairs):
    self.s_prompt = ""
    for q, a in qa_pairs:
        self.s_prompt += f'\n\nPatient: {q}\nCounselor: {a}'
    theme = themes[0]
    self.s_prompt += f'\n\nPatient question: {question}\nLLM prompt: '
    self.f_prompt = f"Below is an instruction that describes a task, paired with some inputs that provide further context. Write a response that appropriately completes the request.\n\n### Instruction:\nDetermine if any parts of the below context are relevant to the below question. If so, concisely summarize the relevant information from the context in bullet points. If the context is not relevant to the question, output 'not relevant.'\n\n### Context:\n{context}\n\n### Question:\n{question}\n\n### Response:\n"

    print(self.f_prompt)
    self.completion = self.llm.complete(
       self.f_prompt,
        temperature=0.3,
        max_tokens=512,
        top_p=.2,
        frequency_penalty=0,
        presence_penalty=0,
        )
    self.completion = self.completion.text
    self.s_prompt += self.completion +'\n'
    print("Completion: \n\n", self.completion)
    return self.completion

c = Character(name='xyz')


if __name__ == '__main__':
    while True:
        question = input("What question would you like to ask GenChat? \n Press q to quit\n\n")
        if question == 'q':
           break
        context, themes = c.query_vector_index(question)
        c.style_transfer(context, themes, question, [])
