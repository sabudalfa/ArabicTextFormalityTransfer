import sys
sys.path.append('./corekit/')
from llm import *
from llm_jobs import *
from simpleio import *
import pandas as pd

LLM_PATH = '/hdd/shared_models/AceGPT-7B/' # TODO: update me

llm_loader = LLMLoader(
    model_path=LLM_PATH,
    llm_initializer=Llama2Initializer(
        max_new_tokens=100,
    ),
)

dataframe = pd.read_csv('./data/madar_v6.csv')
dataframe = dataframe[dataframe['dialect'] == 'Doha']


def get_samples(split):
    split_dataframe = dataframe[dataframe['split'] == split]
    return list(zip(split_dataframe['dialect_sentence'], split_dataframe['msa_sentence']))

test_samples = get_samples(split='test')

jobs = [
    ExperimentDirInitializationJob(
        job_information={
            'loader': llm_loader.name,
        },
        test_samples=test_samples,
        experiment_path=f'./output/{llm_loader.name}/'
    ),
    TextGenerationJob(
        llm_loader=LLMLoader(
            model_path=f'./output/models/{llm_loader.name}-fine-tuned-for-formality-transfer/',
            tokenizer_path=llm_loader.tokenizer_path,
            llm_initializer=llm_loader.llm_initializer,
        ),
        experiment_path=f'./output/{llm_loader.name}/',
        use_chat_format=False,
        gpus_count=1,
    ),
    LLMEvaluationJob(
        llm_evaluator=LLMEvaluator(
            evaluation_metrics_getter=lambda: [
                BleuMetric(),
                ChrfMetric(),
                BertScoreMetric(lang='ar'),
                CometMetric(),
            ],
        ),
        experiment_path=f'./output/{llm_loader.name}/',
    )
]

for job in jobs:
    job()

print(read_from_json_file(f'./output/{llm_loader.name}/evaluation.json'))
