from tqdm.auto import tqdm 
from kani import Kani, chat_in_terminal
from kani.engines.huggingface.llama2 import LlamaEngine


if __name__ == '__main__':
    ## Load Kani
    engine = LlamaEngine(model_id='TheBloke/Llama-2-70B-chat-GPTQ', model_load_kwargs=dict(device_map='auto'))
    ai = Kani(engine)
    prompt_template = 'Summarize the important information in following transcription of a genetic counselor speaking with a patient who is learning about their ApoE test result: %s'
    ## Read input file
    with open('questions.txt') as f:
    snippets = f.readlines()
    snippets = [s.strip() for s in snippets]
    ## Call LLM
    for s in tqdm(snippets):
        response = await ai.chat_round_str(prompt_template % s)
        with open('summaries.txt', 'a') as f:
            f.write(response + '\n\n---\n')
