import whylogs as why
import glob
import json
import pandas as pd

from langkit.themes import init, group_similarity

## Load docs
llama_responses = []
for fp in glob.glob('retrieved_contexts/q*/llama_completion.txt'):
    with open(fp) as f:
        llama_responses.append(f.read())
gpt_constrained_responses = []
for fp in glob.glob('retrieved_contexts/q*/completions.txt'):
    with open(fp) as f:
        gpt_constrained_responses.append(f.read())
gpt_unconstrained_responses = []
for fp in glob.glob('retrieved_contexts/q*/rawcomplete.txt'):
    with open(fp) as f:
        gpt_unconstrained_responses.append(f.read())
with open('genanswers.txt') as f:
    human_responses = [s.strip() for s in f.readlines()]
assert len(llama_responses) == len(gpt_constrained_responses) == len(gpt_unconstrained_responses) == len(human_responses)

## Load langkit
init(theme_file_path='themes.json')
with open('themes.json') as f:
    themes = json.load(f).keys()

named_responses = {'llama': llama_responses, 'gpt_constrained': gpt_constrained_responses, 'gpt_unconstrained': gpt_unconstrained_responses, 'human': human_responses}
for response_name, responses in named_responses.items():  
    print(response_name)
    theme_similarities = [[group_similarity(resp, theme) for theme in themes] for resp in responses]
    output_df = pd.DataFrame(theme_similarities, columns=themes)
    print(output_df.to_csv())
