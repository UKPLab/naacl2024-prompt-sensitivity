import os
import json
import time
import torch
import random
import pandas as pd

from tqdm import tqdm
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration


number_of_shots = {'cola': [4], 'mnli': [3], 'rte': [2], 'sst2': [4], 'csqa': [5], 'gsm8k': [4]}
seeds = [2266, 105, 86379]
device = 'cuda'

max_tokens = {'zero_sensitivity_a': 2, 'zero_sensitivity_b': 2, 'standard_a': 2, 'standard_b': 2, 'context faithful prompting': 2, 'CoT': 64, 'CoT_r': 64, 'CoT_standard_a':64, 'ape': 2, 'gkp': 2}

pad_tokens = {
    'gpt2': '<|endoftext|>',
    'gpt2-xl': '<|endoftext|>',
    'EleutherAI/gpt-j-6b': '<|endoftext|>',
    'togethercomputer/GPT-JT-6B-v1': '<|endoftext|>',
    't5-small': '<pad>',
    't5-large': '<pad>',
    't5-11b': '<pad>',
    'google/flan-t5-small': '<pad>',
    'google/flan-t5-large': '<pad>',
    'google/flan-t5-xxl': '<pad>',
    'meta-llama/Llama-2-7b-chat-hf': '<unk>',
    'meta-llama/Llama-2-13b-chat-hf': '<unk>'
}

pad_ids = {
    'gpt2': 50256,
    'gpt2-xl': 50256,
    'EleutherAI/gpt-j-6b': 50256,
    'togethercomputer/GPT-JT-6B-v1': 50256,
    't5-small': 0,
    't5-large': 0,
    't5-11b': 0,
    'google/flan-t5-small': 0,
    'google/flan-t5-large': 0,
    'google/flan-t5-xxl': 0,
    'meta-llama/Llama-2-7b-chat-hf': 0,
    'meta-llama/Llama-2-13b-chat-hf': 0
}

eos_ids = {
    'gpt2': 50256,
    'gpt2-xl': 50256,
    'EleutherAI/gpt-j-6b': 50256,
    'togethercomputer/GPT-JT-6B-v1': 50256,
    't5-small': 1,
    't5-large': 1,
    't5-11b': 1,
    'google/flan-t5-small': 1,
    'google/flan-t5-large': 1,
    'google/flan-t5-xxl': 1,
    'meta-llama/Llama-2-7b-chat-hf': 2,
    'meta-llama/Llama-2-13b-chat-hf': 2
}

prompts = {
    'cola': {
        'zero_sensitivity_a': '{sentence} The answer is {answer}.\n',
        'zero_sensitivity_b': 'SENTENCE: {sentence} The answer is {answer}.\nANSWER:',
        'standard_a': '{sentence}\n',
        'standard_b': 'SENTENCE: {sentence}\nQUESTION: Is this (0) unacceptable, or (1) acceptable?\nANSWER:',
        'CoT': 'SENTENCE: {sentence}\nQUESTION: Is this (0) unacceptable, or (1) acceptable?\nANSWER: Let\'s think step by step.',
        'CoT_r': 'SENTENCE: {sentence}\nQUESTION: Is this (0) unacceptable, or (1) acceptable?\nANSWER:',
        'CoT_standard_a': '{sentence}\nLet\'s think step by step.',
        'context faithful prompting': 'Bob said, \"{sentence}\"\nQUESTION: Is this (0) unacceptable, or (1) acceptable in Bob\'s opinion?\nANSWER:',
        'ape': 'Input: {sentence}\nOutput:'
    },
    'mnli': {
        'zero_sensitivity_a': '{sentence1}\n{sentence2}\nThe answer is {answer}.\n',
        'zero_sensitivity_b': 'SENTENCE1: {sentence1}\nSENTENCE2: {sentence2}\nThe answer is {answer}.\nANSWER:',
        'standard_a': '{sentence1}\n{sentence2}\n',
        'standard_b': 'SENTENCE1: {sentence1}\nSENTENCE2: {sentence2}\nQUESTION: Are the two sentences (0) contradiction, (1) entailment, or (2) neutral?\nANSWER:',
        'CoT': 'SENTENCE1: {sentence1}\nSENTENCE2: {sentence2}\nQUESTION: Are the two sentences (0) contradiction, (1) entailment, or (2) neutral?\nANSWER: Let\'s think step by step.',
        'context faithful prompting': 'Bob said, \"sentence 1 is \'{sentence1}\', and sentence 2 is \'{sentence2}\'\"\nQUESTION: Are the two sentences (0) contradiction, (1) entailment, or (2) neutral in Bob\'s opinion?\nANSWER:',
    },
    'rte': {
        'zero_sensitivity_a': '{sentence1}\n{sentence2}\nThe answer is {answer}.\n',
        'zero_sensitivity_b': 'SENTENCE1: {sentence1}\nSENTENCE2: {sentence2}\nThe answer is {answer}.\nANSWER:',
        'standard_a': '{sentence1}\n{sentence2}\n',
        'standard_b': 'SENTENCE1: {sentence1}\nSENTENCE2: {sentence2}\nQUESTION: Are the two sentences (0) entailment, or (1) not_entailment?\nANSWER:',
        'CoT': 'SENTENCE1: {sentence1}\nSENTENCE2: {sentence2}\nQUESTION: Are the two sentences (0) entailment, or (1) not_entailment?\nANSWER: Let\'s think step by step.',        
        'CoT_r': 'SENTENCE1: {sentence1}\nSENTENCE2: {sentence2}\nQUESTION: Are the two sentences (0) entailment, or (1) not_entailment?\nANSWER:',
        'CoT_standard_a': '{sentence1}\n{sentence2}\nLet\'s think step by step.',
        'context faithful prompting': 'Bob said, \"sentence 1 is \'{sentence1}\', and sentence 2 is \'{sentence2}\'\"\nQUESTION: Are the two sentences (0) entailment, or (1) not_entailment in Bob\'s opinion?\nANSWER:',
        'ape': 'Input: {sentence1}\n{sentence2}\nOutput:'
    },
    'sst2': {
        'zero_sensitivity_a': '{sentence} The answer is {answer}.\n',
        'zero_sensitivity_b': 'SENTENCE: {sentence} The answer is {answer}.\nANSWER:',
        'standard_a': '{sentence}\n',
        'standard_b': 'SENTENCE: {sentence}\nQUESTION: Is this (0) negative, or (1) positive?\nANSWER:',
        'CoT': 'SENTENCE: {sentence}\nQUESTION: Is this (0) negative, or (1) positive?\nANSWER: Let\'s think step by step.',
        'context faithful prompting': 'Bob said, \"{sentence}\"\nQUESTION: Is this (0) negative, or (1) positive in Bob\'s opinion?\nANSWER:',
    },
    'csqa': {
        'zero_sensitivity_a': '{sentence}\n{options}\nThe answer is {answer}.\n',
        'zero_sensitivity_b': 'SENTENCE: {sentence}\nOPTIONS: {options}\nThe answer is {answer}.\nANSWER:',
        'standard_a': '{sentence}\n{options}',
        'standard_b': 'SENTENCE: {sentence}\nQUESTION: Is it {options}?\nANSWER:',
        'CoT': 'SENTENCE: {sentence}\nQUESTION: Is it {options}?\nANSWER: Let\'s think step by step.',
        'context faithful prompting': 'Bob said, \"{sentence}\"\nQUESTION: Is it {options} in Bob\'s opinion?\nANSWER:',
        'gkp': 'Knowledge: {knowledge}\nInput: {sentence}\nOptions:{options}\nOutput:'
    },
    'cnn_dm': {
        'standard_a': '{passage}\n',
        'standard_b': 'PASSAGE: {passage}\nSUMMARIZATION:'
    },
    'gsm8k': {
        'standard_a': '{sentence}\n',
        'standard_b': 'SENTENCE: {sentence}\nANSWER:',
        'zero_sensitivity_a': '{sentence} The answer is {answer}.\n',
        'zero_sensitivity_b': 'SENTENCE: {sentence} The answer is {answer}.\nANSWER:',
        'CoT': 'SENTENCE: {sentence}\nANSWER: Let\'s think step by step.'
    }
}

def concatenate(dataset, prompt_type, prefix, sentence, item):
    if dataset in ['cola', 'sst2']:
        if prompt_type in ['zero_sensitivity_a', 'zero_sensitivity_b']:
            return prefix + prompts[dataset][prompt_type].format(sentence=sentence, answer=str(item['label']))
        else:
            return prefix + prompts[dataset][prompt_type].format(sentence=sentence)
    if dataset in ['rte', 'mnli']:
        if prompt_type in ['zero_sensitivity_a', 'zero_sensitivity_b']:
            return prefix + prompts[dataset][prompt_type].format(sentence1=sentence, sentence2=item['original_hypothesis'], answer=str(item['label']))
        else:
            return prefix + prompts[dataset][prompt_type].format(sentence1=sentence, sentence2=item['original_hypothesis'])
    if dataset in ['csqa']:
        if prompt_type in ['zero_sensitivity_a', 'zero_sensitivity_b']:
            return prefix + prompts[dataset][prompt_type].format(sentence=sentence, options=', '.join([f'({i}) {option}' for i, option in enumerate(item['options'])]), answer=str(item['label']))
        elif prompt_type in ['gkp']:
            return prefix + prompts[dataset][prompt_type].format(sentence=sentence, options=', '.join([f'({i}) {option}' for i, option in enumerate(item['options'])]), knowledge=random.choice(item['knowledge']))
        else:
            return prefix + prompts[dataset][prompt_type].format(sentence=sentence, options=', '.join([f'({i}) {option}' for i, option in enumerate(item['options'])]))
    if dataset in ['cnn_dm']:
         return prefix + prompts[dataset][prompt_type].format(passage=sentence)
    if dataset in ['gsm8k']:
        if prompt_type in ['zero_sensitivity_a', 'zero_sensitivity_b']:
            return prefix + prompts[dataset][prompt_type].format(sentence=sentence, answer=str(item['answer']))
        else:
            return prefix + prompts[dataset][prompt_type].format(sentence=sentence)
