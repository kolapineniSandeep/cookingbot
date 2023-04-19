import pandas as pd

import json
from glob import glob



class dataset(object):

    def __init__(self):
        self.datasets = None
        self.data = None
        self.responses=None

    def load_dataset(self):
        files = glob('data/*', recursive=True)
        self.datasets = []
        for single_file in files:
            with open(single_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                self.datasets.append(dataset)

    def processing_json_dataset(self,dataset):
        tags = []
        inputs = []
        responses = {}
        for intent in dataset['intents']:
            responses[intent['intent']] = intent['responses']
            for lines in intent['text']:
                inputs.append(lines)
                tags.append(intent['intent'])
        return [tags, inputs, responses]



    def process_data(self):
        self.load_dataset()
        tags = []
        inputs = []
        self.responses = {}

        for dataset in self.datasets:
            [tags_list, inputs_list, responses_dict] = self.processing_json_dataset(dataset)
            tags = tags + tags_list
            inputs = inputs + inputs_list
            self.responses.update(responses_dict)

        self.data = pd.DataFrame({"inputs": inputs,
                                "tags": tags})