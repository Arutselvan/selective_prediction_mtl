from datasets import load_dataset, Dataset
import pandas as pd

class HeterogenousSampling():
    def __init__(self):
        self.SNLI_PREFIX = ""
        self.CSQA_PREFIX = ""
        self.SWAG_PREFIX = ""
        self.ANLI_PREFIX = ""
        self.SIQA_PREFIX = ""
        self.total_dataset = []

    def create_snli_t5_df(self):
        class_label_map = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }

        sent1 = []
        sent2 = []

        dataset = load_dataset("snli", 'plain_text', split='validation')
        dataset = dataset.shuffle(seed=10)
        # dataset = dataset.select(range(SAMPLE_SIZE))
        # dataset = dataset.select(EVAL_SIZE)

        # dataset.map(lambda instance: self.total_dataset.append({"dataset": "snli", "instance": instance}))

        dataset.to_csv(path_or_buf="snli_eval.csv")

        dataset.map(lambda instance: sent1.append(self.SNLI_PREFIX + "premise: " + instance['premise'] + " hypothesis: " + instance['hypothesis']) if instance['label']!=-1 else None)
        dataset.map(lambda instance: sent2.append(class_label_map[instance['label']]) if instance['label']!=-1 else None)

        data = {"sentence1": sent1,"sentence2": sent2}

        return pd.DataFrame(data)
    
    def create_swag_t5_df(self):
        swag_dataset = load_dataset("swag", 'regular', split='validation')
        swag_dataset = swag_dataset.shuffle(seed=10)
        # swag_dataset = swag_dataset.select(range(SAMPLE_SIZE))
        # swag_dataset = swag_dataset.select(EVAL_SIZE)
        sent1 = []
        sent2 = []

        # swag_dataset.map(lambda instance: self.total_dataset.append({"dataset": "swag", "instance": instance}))

        swag_dataset.to_csv(path_or_buf="swag_eval.csv")

        swag_dataset.map(lambda instance: sent1.append(self.SWAG_PREFIX + "startphrase: " + instance['startphrase'] + " - ending0: " + instance['ending0'] + " ending1: " + instance['ending1'] + " ending2: " + instance['ending2'] + " ending3: " + instance['ending3']) if instance['label']!=-1 else None)
        swag_dataset.map(lambda instance: sent2.append(instance["ending"+str(instance["label"])])if instance['label']!=-1 else None)

        swag_data = {"sentence1": sent1,"sentence2": sent2}

        return pd.DataFrame(swag_data)
    
    def create_csqa_t5_df(self):

        csqa_dataset = load_dataset("commonsense_qa", 'regular', split='validation')
        csqa_dataset = csqa_dataset.shuffle(seed=10)
        # csqa_dataset = csqa_dataset.select(range(SAMPLE_SIZE))
        # csqa_dataset = csqa_dataset.select(EVAL_SIZE)

        # csqa_dataset.map(lambda instance: self.total_dataset.append({"dataset": "comqa", "instance": instance}))

        csqa_dataset.to_csv(path_or_buf="csqa_eval.csv")

        sent1 = []
        sent2 = []
        csqa_dataset.map(lambda instance: sent1.append(self.CSQA_PREFIX + "question: " + instance['question'] + ", choices : " + ",".join(instance['choices']['text'])) if instance['answerKey'] != '' else None)
        csqa_dataset.map(lambda instance: sent2.append(instance['choices']['text'][instance['choices']['label'].index(instance['answerKey'])]) if instance['answerKey'] !='' else None)

        csqa_data = {"sentence1": sent1,"sentence2": sent2}

        return pd.DataFrame(csqa_data)

    def create_anli_t5_df(self):
        hypothesis_label_map = {
            0: "",
            1: "hypothesis_1",
            2: "hypothesis_2"
        }

        sent1 = []
        sent2 = []

        dataset = load_dataset("art", split='validation')
        dataset = dataset.shuffle(seed=10)
        # dataset = dataset.select(range(SAMPLE_SIZE))
        # dataset = dataset.select(EVAL_SIZE)

        dataset.to_csv(path_or_buf="anli_eval.csv")

        # dataset.map(lambda instance: self.total_dataset.append({"dataset": "anli", "instance": instance}))

        dataset.map(lambda instance: sent1.append(self.ANLI_PREFIX + "observation 1: " + instance['observation_1'] + ", observation 2: " + instance['observation_2'] + ", hypothesis 1: " + instance['hypothesis_1'] + ", hypothesis 2: " + instance['hypothesis_2']))
        dataset.map(lambda instance: sent2.append(instance[hypothesis_label_map[instance['label']]] if 1 <= instance['label'] <= 2 else "Neither of the hypotheses infer the observations"))

        data = {"sentence1": sent1,"sentence2": sent2}

        return pd.DataFrame(data)

    def create_siqa_t5_df(self):

        siqa_dataset = load_dataset("social_i_qa", 'regular', split='validation')
        siqa_dataset = siqa_dataset.shuffle(seed=10)
        # siqa_dataset = siqa_dataset.select(range(SAMPLE_SIZE))
        # siqa_dataset = siqa_dataset.select(EVAL_SIZE)

        # siqa_dataset.map(lambda instance: self.total_dataset.append({"dataset": "siqa", "instance": instance}))

        siqa_dataset.to_csv(path_or_buf="siqa_eval.csv")

        answer_label_map = {
            "1": "answerA",
            "2": "answerB",
            "3": "answerC"
        }

        sent1 = []
        sent2 = []
        siqa_dataset.map(lambda instance: sent1.append(self.SIQA_PREFIX + "context: " + instance['context'] + ", question: " + instance['question'] + ", choices : A - " + instance['answerA'] + "; B - " + instance['answerB'] + "; C - " + instance['answerC']) if instance['label'] != '' else None)
        siqa_dataset.map(lambda instance: sent2.append(instance[answer_label_map[instance['label']]]) if instance['label'] !='' else None)

        siqa_data = {"sentence1": sent1,"sentence2": sent2}

        return pd.DataFrame(siqa_data)

if __name__=="__main__":
    sampler = HeterogenousSampling()
    snli_data = sampler.create_snli_t5_df()
    swag_data = sampler.create_swag_t5_df()
    csqa_data = sampler.create_csqa_t5_df()
    anli_data = sampler.create_anli_t5_df()
    siqa_data = sampler.create_siqa_t5_df()
    
    # frames=[snli_data, swag_data, csqa_data, anli_data, siqa_data]
    # data = pd.concat(frames)
    # data = data.sample(frac=1)
    # print(len(data))
    # data.to_csv("t5_mtl.csv")
    # print(pd.Series({c: data[c].map(lambda x: len(str(x).split(' '))).max() for c in data}).sort_values(ascending =False))