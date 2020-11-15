from flashtext import KeywordProcessor
import json

file = "/home/hieunm/TextMining/multidocsum/src/data/preprocess/test.txt"

kp = KeywordProcessor(case_sensitive=True)
rules = json.load(open("/home/hieunm/TextMining/multidocsum/src/data/preprocess/rules.json"))
kp.add_keywords_from_dict(rules)
for i, line in enumerate(open(file)):
    text = line.strip()
    text = text.replace("_"," ")
    founds = kp.extract_keywords(text, span_info=True)
    if len(founds)>0 : 
        print("\n",founds)
        print(text)
        print(kp.replace_keywords(text))