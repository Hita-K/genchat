import json

from collections import defaultdict


with open('themes.txt') as f:
    l = [s.strip() for s in f.readlines()]
inv_map = defaultdict(list)
for i, row in enumerate(l):
    labels = [s.strip() for s in row.split(',')]
    for label in labels: 
        inv_map[label].append(i)

with open('questions.txt') as f:
    sents = [s.strip() for s in f.readlines()]

themes_map = {k: [sents[i] for i in v] for k, v in inv_map.items()}

## Remap some unicode key names
themes_map['Distilling definitions into "take home messages"'] = themes_map.pop('Distilling definitions into “take home messages”')
themes_map["Identifying when to defer to other specialists' opinion"] = themes_map.pop('Identifying when to defer to other specialists’ opinion')
themes_map["Recognizing patients' prior background knowledge"] = themes_map.pop('Recognizing patients’ prior background knowledge')

with open('themes.json', 'w') as f:
    f.write(json.dumps(themes_map))
