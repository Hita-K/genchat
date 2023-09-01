import os
import pandas as pd
import random
import torch

from sentence_transformers import SentenceTransformer, util, models
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings import OpenAIEmbedding


idk = ["I am not sure...", "I am so sorry, I don't know.", "I am afraid I do not know how to answer this question.", "I don't know. Even though I am an old man, my knowledge is limited. Can you ask me something else?", "I am sorry, I do not know the answer to that. I died over 500 years ago, so there are many things that I don't know. Can you ask me something else?"]

# TODO: Set up OpenAI auth tokens 

## CLASS ##

class Character:

  def __init__(self, name, docs_fp='summaries.txt', answerdoc_fp='themes.txt'):
    self.s_prompt = ""
    self.name = name

    # Load docs
    self.docs_fp = docs_fp
    with open(docs_fp) as f:
      self.docs = f.readlines()

    with open(answerdoc_fp) as g:
      self.answerdoc = g.readlines()

    # LlamaIndex stuff
    self.embedding_model = OpenAIEmbedding()
    self.service_context = ServiceContext.from_defaults(embed_model=embed_model)
    self.build_vector_index()


  def build_vector_index(self):
    documents = SimpleDirectoryReader(input_files=[self.docs_fp]).load_data()
    self.index = VectorStoreIndex.from_documents(documents, service_context=self.service_context)


  def query_vector_index(self, query):
    retriever = self.index.as_retriever(similarity_top_k=3)
    top_results = retriever.retrieve(query)

    ## TODO retrieve real themes
    dummy_themes = ['foo' for foo in top_results]

    return top_results, dummy_themes


  def style_transfer(self, context, themes, question, qa_pairs):
    self.s_prompt = ""
    for q, a in qa_pairs:
        self.s_prompt += f'\n\nPatient: {q}\nCounselor: {a}'
    theme = themes[0]
    self.s_prompt += f'\n\nPatient question: {question}\nLLM prompt: '
    self.f_prompt = f"Below is an instruction that describes a task, paired with some inputs that provide further context. Write a response that appropriately completes the request.\n\n### Instruction:\nDetermine if any parts of the below context are relevant to the below question. If so, concisely summarize the relevant information from the context in bullet points. If the context is not relevant to the question, output 'not relevant.'\n\n### Context:\n{context}\n\n### Question:\n{question}\n\n### Response:\n"

    # print(self.f_prompt)
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
  with open('faqs.txt') as f:
    questions = f.readlines()

  for i, question in enumerate(questions):
    print(f"Question: {question}")
    context, themes = c.query_vector_index(question)
    os.makedirs(f'retrieved_contexts/q{i}', exist_ok=True) 
    with open(f'retrieved_contexts/q{i}/contexts.txt', 'w') as f:
        f.write('\n\n---\n'.join(context))
    # c.style_transfer(context, themes, question, [])
