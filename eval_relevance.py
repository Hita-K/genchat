import whylogs as why
import glob

from langkit import llm_metrics # alternatively use 'light_metrics'
from whylogs.experimental.core.udf_schema import udf_schema


## Load docs
with open('qlist.txt') as f: 
    questions = f.readlines()
responses = []
for fp in glob.glob('retrieved_contexts/q*/llama_completion.txt'):
    with open(fp) as f:
        responses.append(f.read())
# with open('genanswers.txt') as f:
#     responses = [s.strip() for s in f.readlines()]
assert len(responses) == len(questions)

## Load langkit
why.init(session_type='whylabs_anonymous')
themes.init(theme_file='themes.json')
text_schema = udf_schema()

l = [{'prompt': q, 'response': r} for q, r in zip(questions, responses)]
profiles = [why.log(d, schema=text_schema, name=f'q{i}') for i, d in enumerate(l)]
relevance_scores = [p.view().to_pandas().loc['response.relevance_to_prompt', 'distribution/max'] for p in profiles]
print(relevance_scores)
