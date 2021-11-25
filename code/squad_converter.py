import pandas as pd
from string import punctuation
import re

def strip_data(df):
    df['question']  = df['question'].apply(lambda x: x.strip(punctuation).strip())
    df['context']  = df['context'].apply(lambda x: x.strip(punctuation).strip())
    # df['answer']  = df['answer'].apply(lambda x: x.strip(punctuation).strip())
    return df

def find_all_answers(sentence, answer):
    
    answers_list = []
    answer = answer.strip(punctuation)
    
    for index, match in enumerate(re.finditer(r'\b%s\b' % re.escape(answer), sentence, re.IGNORECASE)):
        answer_obj = {
            "answer_start": [match.start()],
            "text": [match.group(0)],
        }
        answers_list.append(answer_obj)         

    return answers_list[0]

def snli_to_squad(data_src,dest):

    SNLI_COLS_TO_SELECT = ['hypothesis','premise','label']
    SNLI_RENAME_MAPPING = {
        'premise': 'context',
        'hypothesis': 'question',
        'label': 'answer'
    }
    SNLI_CLASS_LABEL_MAP = {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }
    df = pd.read_csv(data_src)
    df = df[(df.label!=-1)]

    df = df[SNLI_COLS_TO_SELECT]
    df = df.rename(SNLI_RENAME_MAPPING, axis=1)
    df['question'] = df['question'].apply(lambda x: "Hypothesis: " + x)
    df['context'] = df['context'].apply(lambda x: "Premise: " + x + " entailment, neutral, or contradiction ?")
    strip_data(df)
    df['answer'] = df.apply(lambda x: find_all_answers(x['context'],SNLI_CLASS_LABEL_MAP[x['answer']].strip(punctuation).strip()),axis=1)
    df['id'] = "snli_" + df.index.astype(str)
    df = df[['id','question', 'context', 'answer']]
    df.to_json(dest,orient='records',lines=True)
    return df

def swag_to_squad(data_src,dest):
    df = pd.read_csv(data_src)
    df['id'] = df.index + 1
    df['question'] = df.apply(lambda x:"Sentence 1: " + x['sent1'] + " Sentence 2: " + x['sent2'],axis=1)
    df['context'] = df.apply(lambda x:"Endings: " + x['ending0'] + ", " + x['ending1'] + ", " + x["ending2"] + ", " + x["ending3"],axis=1)
    strip_data(df)
    df['answer'] = df.apply(lambda x: find_all_answers(x['context'],x["ending"+ str(x['label'])].strip(punctuation).strip()),axis=1)
    df['id'] = "swag_" + df.index.astype(str) 
    df = df[['id','question', 'context', 'answer']]
    df.to_json(dest,orient='records',lines=True)
    return df   

def comqa_to_squad(data_src,dest):
    df = pd.read_csv(data_src)
    df['id'] = df.index + 1
    df['choices'] = df['choices'].apply(lambda x: x.replace('array(', ''))
    df['choices'] = df['choices'].apply(lambda x: x.replace(' dtype=object),', ''))
    df['choices'] = df['choices'].apply(lambda x: x.replace(' dtype=object)', ''))
    df['choices'] = df['choices'].apply(lambda x: eval(x))
    df['context'] = df.apply(lambda x: "Options: " + ", ".join(x['choices']['text']),axis=1)
    strip_data(df)
    df['answer'] = df.apply(lambda x: find_all_answers(x['context'],x['choices']['text'][x['choices']['label'].index(x['answerKey'])].strip(punctuation).strip()),axis=1)
    df['id'] = "csqa_" + df.index.astype(str)
    df = df[['id','question', 'context', 'answer']]
    df.to_json(dest,orient="records",lines=True)
    return df

def art_to_squad(data_src,dest):
    df = pd.read_csv(data_src)
    df['id'] = df.index + 1
    df['question'] = df.apply(lambda x: "Observation 1: " +  x['observation_1'] + ' Observation 2: ' + x['observation_2'], axis=1)
    df['context'] = df.apply(lambda x: "Hypothesis 1: " + x['hypothesis_1'] + ' Hypothesis 2: ' + x['hypothesis_2'],axis=1)
    strip_data(df)
    df['answer'] = df.apply(lambda x: find_all_answers(x['context'],x["hypothesis_"+str(x['label'])].strip(punctuation).strip()),axis=1)
    df['id'] = "art_" + df.index.astype(str)
    df = df[['id','question', 'context', 'answer']]
    df.to_json(dest,orient="records",lines=True)
    return df

def siqa_to_squad(data_src,dest):
    SIQA_LABEL_MAP = {
        1: 'A',
        2: 'B',
        3: 'C'
    }
    df = pd.read_csv(data_src)
    df['id'] = "siqa_" + df.index.astype(str)
    df['context'] = df.apply(lambda x: x['context'] + ' Options: ' + x['answerA'] + ' ,' + x['answerB'] + ' ,' + x['answerC'],axis=1)
    strip_data(df)
    df['answer'] = df.apply(lambda x: find_all_answers(x['context'],x['answer' + SIQA_LABEL_MAP[x['label']]].strip(punctuation).strip()),axis=1)
    df = df[['id','question', 'context', 'answer']]
    df['id'] = "siqa_" + df.index.astype(str)
    df.to_json(dest,orient="records",lines=True)
    return df

if __name__ == "__main__":
    main_df = pd.concat([snli_to_squad("dataset/snli.csv", "dataset/snli_squad.json"),swag_to_squad("dataset/swag.csv", "dataset/swag_squad.json"),comqa_to_squad("dataset/csqa.csv", "dataset/csqa_squad.json"),art_to_squad("dataset/anli.csv", "dataset/anli_squad.json"),siqa_to_squad("dataset/siqa.csv","dataset/siqa_squad.json")])
    main_df = main_df.sample(frac=1,random_state=10)
    main_df.to_json("dataset/main.json",orient="records",lines=True)
    snli_to_squad("dataset/snli_eval.csv", "dataset/snli_squad_eval.json")
    swag_to_squad("dataset/swag_eval.csv", "dataset/swag_squad_eval.json")
    comqa_to_squad("dataset/csqa_eval.csv", "dataset/csqa_squad_eval.json")
    art_to_squad("dataset/anli_eval.csv", "dataset/anli_squad_eval.json")
    siqa_to_squad("dataset/siqa_eval.csv","dataset/siqa_squad_eval.json")

