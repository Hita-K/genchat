import whylogs as why
import glob

from langkit.input_output import init, prompt_response_similarity

## Load docs
with open('qlist.txt') as f: 
    questions = f.readlines()
responses = []
for fp in glob.glob('retrieved_contexts/q*/llama_completion.txt'):
    with open(fp) as f:
        responses.append(f.read())
# responses = []
# for fp in glob.glob('retrieved_contexts/q*/completions.txt'):
#     with open(fp) as f:
#         responses.append(f.read())
# responses = []
# for fp in glob.glob('retrieved_contexts/q*/rawcomplete.txt'):
#     with open(fp) as f:
#         responses.append(f.read())
# with open('genanswers.txt') as f:
#     responses = [s.strip() for s in f.readlines()]
assert len(questions) == len(responses)
l = [{'prompt': [q], 'response': [r]} for q, r in zip(questions, responses)]

## Load langkit
init()

relevance_scores = [prompt_response_similarity(d)[0] for d in l]
print(relevance_scores)
