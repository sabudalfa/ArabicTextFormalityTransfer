import os
import time
from os.path import *
from pathlib import Path
import pandas as pd
import requests
import json
from typing import *
import urllib3
from abc import *
import sacrebleu
from evaluate import load
from statistics import mean

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

USER_PROMPT = 'Convert the following text to modern standard Arabic. Do not write comments or explanations. Just rewrite the text in modern standard Arabic.'

ACCESS_TOKEN = '' # TODO: Add the token

def make_parent_dirs(path: str):
    (Path(path)
     .parent
     .mkdir(parents=True, exist_ok=True))

def read_from_file(file_path: str, default_value=None):
    if not exists(file_path):
        return default_value
    with open(file_path, mode='r') as file:
        return file.read()


def read_from_json_file(file_path: str, default_value=None):
    if not exists(file_path):
        print(f'{file_path} does not exist')
        return default_value
    with open(file_path, mode='r') as json_file:
        return json.load(json_file)


def write_to_file(file_path: str, content: Union[str, List[str]], mode='w'):
    make_parent_dirs(file_path)
    with open(file_path, mode=mode) as file:
        if type(content) is str:
            file.write(content)
        else:
            for line in content:
                file.write(f'{line}\n')


def append_to_file(file_path: str, content: Union[str, List[str]]):
    write_to_file(file_path, content, mode='a')


def write_to_json_file(file_path: str, content):
    make_parent_dirs(file_path)
    with open(file_path, mode='w') as json_file:
        return json.dump(content, json_file, ensure_ascii=False)



def get_chat_completion(messages):
    payload = {
        "messages": messages,
        "temperature": 0.6,
        "stream": False,
        "model": "allam",
        "top_p": 0.98,
        "n": 1,
        "add_generation_prompt": True,
        "echo": False,
        "stop": " </s>",
    }
    response = requests.post(
        'https://vllm-v19.allam.ai/v1/chat/completions',
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ACCESS_TOKEN}"
        },
        data=json.dumps(payload),
        timeout=150,
        verify=False,
    )
    if response.status_code == 200:
        chat_response_data = response.json()
        return chat_response_data['choices'][0]['message']['content']
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
        return None


def create_prompt_messages(shots, dialect_sentence):
    return [
        *[
            shot_message
            for dialect_text_i, msa_text_i in shots
            for shot_message in [
                {
                    'role': 'user',
                    'content': f'{USER_PROMPT}\nText: {dialect_text_i}',
                },
                {
                    'role': 'assistant',
                    'content': f'{msa_text_i}',
                },
            ]
        ],
        {
            'role': 'user',
            'content': f'{USER_PROMPT}\nText: {dialect_sentence}',
        },
    ]


def convert_dialect(dialect_sentence, shots=None):
    if shots is None:
        shots = []
    prompt_messages = create_prompt_messages(
        shots=shots,
        dialect_sentence=dialect_sentence,
    )
    model_output = None
    sleep_period = 1.0
    while model_output is None:
        model_output = get_chat_completion(prompt_messages)
        if model_output is not None:
            break
        print(f"Sleep for {sleep_period} and then retry")
        time.sleep(sleep_period)
        sleep_period *= 2
    model_output = model_output.strip()
    if '\n' in model_output:
        model_output = model_output.split('\n')[0]
    return model_output

dataset_names = [
    'madar_v6.csv',
    'MDC_splitted.csv',
    'bible_splitted.csv',
    'PADIC_splitted.csv',
]

output_files = []
i = 0
for shots_count in [0, 5]:
    for dataset_name in dataset_names:
        dataframe = pd.read_csv(f'./data/{dataset_name}')
        train_dataframe = dataframe[dataframe['split'] == 'train']
        test_dataframe = dataframe[dataframe['split'] == 'test']
        dialects = train_dataframe['dialect'].unique().tolist()
        for dialect in dialects:
            shots = [
                (row['dialect_sentence'], row['msa_sentence'])
                for _, row in train_dataframe[train_dataframe['dialect'] == dialect][:shots_count].iterrows()
            ]
            dialect_dataframe = test_dataframe[test_dataframe['dialect'] == dialect]
            dialect_sentences = [
                row['dialect_sentence']
                for _, row in dialect_dataframe.iterrows()
            ]
            msa_sentences = [
                row['msa_sentence']
                for _, row in dialect_dataframe.iterrows()
            ]
            output_file = f'./output/{i}.json'
            i += 1
            output_files.append(output_file)
            # write_to_json_file(
            #     output_file,
            #     content={
            #         'dataset': dataset_name,
            #         'dialect': dialect,
            #         'shots_count': shots_count,
            #         'shots': shots,
            #         'dialect_sentences': dialect_sentences,
            #         'actual_msa_sentences': msa_sentences,
            #         'predicted_msa_sentences': [
            #             convert_dialect(
            #                 dialect_sentence=dialect_sentence,
            #                 shots=shots,
            #             )
            #             for dialect_sentence in dialect_sentences
            #         ],
            #     }
            # )


class TGMetric(ABC):

    @abstractmethod
    def __call__(
            self,
            ground_truth: List[str],
            predictions: List[str],
            sources: List[str],
    ) -> Dict[str, float]:
        pass


class BleuMetric(TGMetric):

    def __init__(self):
        self.bleu = sacrebleu.BLEU()

    def __call__(self, ground_truth: List[str], predictions: List[str], sources: List[str]):
        bleu_score = self.bleu.corpus_score(predictions, [ground_truth]).score
        return {
            'bleu': bleu_score,
        }
    
    
class ChrfMetric(TGMetric):

    def __call__(self, ground_truth: List[str], predictions: List[str], sources: List[str], word_order=2):
        chrf = sacrebleu.CHRF(word_order=word_order)
        chrf_score = chrf.corpus_score(predictions, [ground_truth]).score
        return {
            'chrf': chrf_score,
        }


class CometMetric(TGMetric):

    def __init__(self):
        self._comet_metric = load('comet')

    def __call__(
            self,
            ground_truth: List[str],
            predictions: List[str],
            sources: List[str],
    ) -> Dict[str, float]:
        compute_result = self._comet_metric.compute(
            predictions=predictions,
            references=ground_truth,
            sources=sources,
            progress_bar=True,
        )
        return {'comet': compute_result['mean_score']}


class BertScoreMetric(TGMetric):

    def __init__(self, lang):
        self._bertscore = load("bertscore")
        self._lang = lang

    def __call__(self, ground_truth: List[str], predictions: List[str], sources: List[str],) -> Dict[str, float]:
        assert len(ground_truth) == len(predictions)
        compute_result = self._bertscore.compute(
            predictions=predictions,
            references=ground_truth,
            lang=self._lang,
            verbose=True,
        )
        return {
            'bertscore_precision': mean(compute_result['precision']),
            'bertscore_recall': mean(compute_result['recall']),
            'bertscore_f1': mean(compute_result['f1']),
        }


metrics = [
    BleuMetric(),
    CometMetric(),
    BertScoreMetric(lang='ar'),
    ChrfMetric(),
]

evaluation_records = []
for output_file in output_files:
    data = read_from_json_file(output_file)
    evaluation_records.append(
        {
            'dataset': data['dataset'],
            'dialect': data['dialect'],
            'shots_count': data['shots_count'],
            **{
                k: v
                for metric in metrics
                for k, v in metric(
                    ground_truth=data['actual_msa_sentences'],
                    predictions=data['predicted_msa_sentences'],
                    sources=data['dialect_sentences'],
                ).items()
            },
        }
    )

evaluation_dataframe = pd.DataFrame.from_records(evaluation_records)
evaluation_dataframe.to_csv('./output/evaluation.csv', index=False)
