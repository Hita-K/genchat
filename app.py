import openai
import os
import pandas as pd
import random
import logging
import sys

import nest_asyncio

nest_asyncio.apply()

from llama_index.retrievers import VectorIndexRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.postprocessor import LLMRerank
from llama_index.llms import OpenAI

from IPython.display import display, HTML, Markdown

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    LLMPredictor,
)

idk = ["I am not sure...", "I am so sorry, I don't know.", "I am afraid I do not know how to answer this question.", "I don't know. Even though I am an old man, my knowledge is limited. Can you ask me something else?", "I am sorry, I do not know the answer to that. I died over 500 years ago, so there are many things that I don't know. Can you ask me something else?"]

# Load your API key from an environment variable or secret management service
OPENAI_API_KEY = "blah"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# do we need to change this to llama?
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

# Read docs
f = open("questions.txt", "r")
# do we need to change this to a document object?
docs = f.readlines()
f.close()

g = open("themes.txt", "r")
answerdoc = g.readlines()
g.close()


## LLM Retrieval ##

# pd.set_option("display.max_colwidth", -1)

index = VectorStoreIndex.from_documents(docs, service_context=service_context)

def get_retrieved_nodes(
    query_str, vector_top_k=10, reranker_top_n=3, with_reranker=False
):
    query_bundle = QueryBundle(query_str)
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    )
    retrieved_nodes = retriever.retrieve(query_bundle)

    if with_reranker:
        # configure reranker
        reranker = LLMRerank(
            choice_batch_size=5, top_n=reranker_top_n, service_context=service_context
        )
        retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

    return retrieved_nodes


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


def visualize_retrieved_nodes(nodes) -> None:
    result_dicts = []
    for node in nodes:
        result_dict = {"Score": node.score, "Text": node.node.get_text()}
        result_dicts.append(result_dict)

    pretty_print(pd.DataFrame(result_dicts))



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

    top_k = min(3, len(docs))
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
    print("These are the closest 2 sentences to the given query: \n\n", final_context)

    return final_context, themes
  
  def style_transfer(self, context, themes, question, qa_pairs):
    self.s_prompt = ""
    for q, a in qa_pairs:
        self.s_prompt += f'\n\nPatient:{q}\nCounselor:{a}'
    theme = themes[0]
    self.s_prompt += f'\n\nPatient:{question}\nCounselor:'
    self.f_prompt = f"Modify this text as if you are a genetic counselor talking to a patient who has two copies of the apolipoprotein E (APOE4) allele, a major genetic risk factor of Alzheimer's disease. Make sure to use proper grammar and explain the term in multiple ways:\"{context}\"{self.s_prompt}"


    try :
        self.completion = openai.Completion.create(
                                            model = "text-davinci-003",
                                            prompt= "What is the capital of France?",
                                            temperature=0.3,
                                            max_tokens=256,
                                            top_p=.2,
                                            frequency_penalty=0,
                                            presence_penalty=0, 
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
    
    new_nodes = get_retrieved_nodes(
    "What is the definition of MCI?",
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=False,
)

    # while True:
    #     question = input("What question would you like to ask GenChat? \n Press q to quit\n\n")
    #     if question == 'q':
    #        break
    #     context, themes = c.search_docs(question)
    #     c.style_transfer(context, themes, question+" Please give me a long answer.", [])


