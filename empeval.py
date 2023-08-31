from rouge import Rouge

reference = "from beth/cara"

rouge = Rouge()

with open("qlist.txt", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    line = line.strip()
    scores = rouge.get_scores(line, reference)
    
    print(f"Line {i+1} ROUGE Scores:")
    for score_type, score_value in scores[0].items():
        print(f"  {score_type.upper()}:")
        for metric, value in score_value.items():
            print(f"    {metric}: {value}")