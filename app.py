import os
import pandas as pd
import random
import torch
import openai
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()

from sentence_transformers import SentenceTransformer, util, models
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings import OpenAIEmbedding


idk = ["I am not sure...", "I am so sorry, I don't know.", "I am afraid I do not know how to answer this question.", "I don't know. Even though I am an old man, my knowledge is limited. Can you ask me something else?", "I am sorry, I do not know the answer to that. I died over 500 years ago, so there are many things that I don't know. Can you ask me something else?"]

OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

def process_string(docs):
    joinstring = ''.join(docs)
    sections = joinstring.split('---')
    sections = [s.strip() for s in sections if s.strip()]
    processed_sections = []
    for section in sections:
        lines = section.split('\n')
        lines = [line.strip() for line in lines if line.strip().startswith('*')]
        processed_section = ''.join(lines)
        processed_sections.append(processed_section)
    return processed_sections

def safe_index(processed_sections, value):
    try:
        return processed_sections.index(value)
    except ValueError:
        return None

## CLASS ##

class Character:

  def __init__(self, name, docs_fp='summaries.txt', answerdoc_fp='themes.txt'):
    self.s_prompt = ""
    self.name = name

    # Load docs
    self.docs_fp = docs_fp
    with open(docs_fp) as f:
      # s = f.read()
      # self.docs = s.split('\n\n---\n')
      self.docs = f.readlines()
    

    with open(answerdoc_fp) as g:
      self.answerdoc = g.readlines()

    # LlamaIndex stuff
    self.embedding_model = OpenAIEmbedding()
    self.service_context = ServiceContext.from_defaults(embed_model=self.embedding_model)
    self.build_vector_index()

  def build_vector_index(self):
    documents = SimpleDirectoryReader(input_files=[self.docs_fp]).load_data()
    self.index = VectorStoreIndex.from_documents(documents, service_context=self.service_context)


  def query_vector_index(self, query):
    processed_sections = process_string(self.docs)
    # print(len(processed_sections))
    retriever = self.index.as_retriever(similarity_top_k=3)
    top_results = retriever.retrieve(query)
    reslist = [top_results[i].node.get_text() for i in range(len(top_results))]
    indices = [safe_index(processed_sections, process_string([reslist[0]])[i]) for i in range(len(reslist))]
    intindices = [x for x in indices if isinstance(x, int)]
    themes = [self.answerdoc[i] for i in intindices]


    return reslist, themes


  def style_transfer(self, context, themes, question, qa_pairs):
    self.s_prompt = ""
    for q, a in qa_pairs:
        self.s_prompt += f'\n\nPatient: {q}\nCounselor: {a}'
    theme = themes[0]
    self.s_prompt += f'\n\nPatient question: {question}\nCounselor: '
    self.f_prompt = f"You are a genetic counselor talking to a patient who has 2 APoE E4 genes. The patient asks the following question:\n{question}\n\n Answer this question as a genetic counselor. Make sure to use the following information in your asnwer when appropriate: \n{context}\n\n When answering the question, make sure you use the following theme(s) in your response: {theme}. Keep the response short and do not use lists."
    # print(self.f_prompt)

    try:
        self.completion = openai.Completion.create(
                                            model = "text-davinci-003",
                                            prompt=self.f_prompt,
                                            temperature=0.3,
                                            max_tokens=256,
                                            top_p=.2,
                                            frequency_penalty=0,
                                            presence_penalty=0, 
                                            )
        self.completion = self.completion["choices"][0]["text"]
        # print("Completion: \n\n", self.completion)
    except:
       return random.choice(idk)
    self.s_prompt += self.completion +'\n'
    # print(self.completion)
    return self.completion

c = Character(name='xyz')


if __name__ == '__main__':
  with open('faqs.txt') as f:
    questions = f.readlines()

  for i, question in enumerate(questions):
    print(f"Question: {question}")
    context, themes = c.query_vector_index(question)
    completion = c.style_transfer(context, themes, question, [])
    os.makedirs(f'retrieved_contexts/q{i}', exist_ok=True) 
    with open(f'retrieved_contexts/q{i}/contexts.txt', 'w') as f:
        f.write('\n\n---\n'.join(context))
    with open(f'retrieved_contexts/q{i}/completions.txt', 'w') as g:
       g.write(completion)
    
