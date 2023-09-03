from rouge import Rouge
import textstat

reference = "from beth/cara"

rouge = Rouge()

with open("qlist.txt", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    line = line.strip()


    # ROUGE Scores
    scores = rouge.get_scores(line, reference)
    
    print(f"Line {i+1} ROUGE Scores:")
    for score_type, score_value in scores[0].items():
        print(f"  {score_type.upper()}:")
        for metric, value in score_value.items():
            print(f"    {metric}: {value}")


##########

# Readability

with open("gen.txt", "r") as f:
    ans = f.readlines()

def evaluate_readability(text):
    print(f"Text: {text}\n")
    
    fk = textstat.flesch_reading_ease(text)
    print(f"Flesch-Kincaid Reading Ease: {fk}")

    fk_grade = textstat.flesch_kincaid_grade(text)
    print(f"Flesch-Kincaid Grade Level: {fk_grade}")

    gf = textstat.gunning_fog(text)
    print(f"Gunning Fog Index: {gf}")

    cl = textstat.coleman_liau_index(text)
    print(f"Coleman-Liau Index: {cl}")


for i, line in enumerate(ans):
    ans = ans.strip()
    evaluate_readability(ans)


# ###########

from langkit import textstat
from whylogs.experimental.core.udf_schema import udf_schema
import whylogs as why
text_schema = udf_schema()

profile = why.log({"input":"I like you. I love you."}, schema=text_schema).profile()

print(profile)