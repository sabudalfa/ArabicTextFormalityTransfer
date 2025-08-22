import sys
sys.path.append('./corekit/')
from llm import *
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


train_samples = get_samples(split='train')
eval_samples = get_samples(split='dev')


llm_trainer = LLMTrainer(
    llm_loader=llm_loader,
    train_samples=train_samples,
    eval_samples=eval_samples,
    peft_config=LoRAConfigRepository.llama_3(),
    learning_rate=2.5e-4,
    epochs_count=10,
    train_batch_size=16,
    eval_batch_size=16,
    output_dir=f'./output/models/{llm_loader.name}-fine-tuned-for-formality-transfer/'
)
llm_trainer()
