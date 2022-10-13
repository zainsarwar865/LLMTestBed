import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, SequentialSampler
import os
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
# import jsonlines as js
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if torch.cuda.is_available():
    device = torch.device("cuda:3")
else:
    print("try again later")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATTERN_FILE_FORMAT = "data/pattern_data/graphs_json/{}.jsonl"
VOCAB_FILE_FORMAT = "data/trex_lms_vocab/{}.jsonl"
MASK_TOKEN = "[MASK]"
CACHE_DIR = "/bigstor/rbhaskar/models/cache/"
CACHE_DIR_2 = "/bigstor/zsarwar/models/cache/"
N_1_RELATIONS = ['P937', 'P1412', 'P127', 'P103', 'P276', 'P159', 'P140', 'P136', 'P495', 'P17', 'P361', 'P36', 'P740', 'P264', 'P407', 'P138', 'P30', 'P131', 'P176', 'P449', 'P279', 'P19', 'P101', 'P364', 'P106', 'P1376', 'P178', 'P37', 'P413', 'P27', 'P20']

def preparedata(pattern_file, vocab_file):
    sentences = []
    labels = []
    indices = []
    datasets = []
    prompts = []
    
    with open(pattern_file, "r") as f:
        patterns = f.readlines()

    with open(vocab_file, "r") as f:
        vocab = f.readlines()

    dataset = os.path.splitext(os.path.basename(pattern_file))[0]
    
    for pattern in patterns:
        pattern = json.loads(pattern)
        prompt = pattern["pattern"]
        pattern = pattern["pattern"]
        pattern = pattern.replace("[Y]", MASK_TOKEN) # strings are immutable, so this doesn't affect prompt

        for sub_obj in vocab:
            sub_obj = json.loads(sub_obj)
            sub = sub_obj["sub_label"]
            obj = sub_obj["obj_label"]

            sentences.append(pattern.replace("[X]", sub))
            labels.append(obj)
            datasets.append(dataset)
            indices.append(len(indices))
            prompts.append(prompt)

    return sentences, labels, indices, datasets, prompts


def prep_inputs(sents, tokenizer):
    
    mask_token_indices = []
    batch_input_ids = tokenizer.batch_encode_plus(sents, add_special_tokens=True, padding=True, return_attention_mask=True, return_tensors='pt')

    for i, inp_ids in tqdm(enumerate(batch_input_ids['input_ids'])):
        
        mask_index = (inp_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        mask_index = torch.where(inp_ids == tokenizer.mask_token_id)[0]
        if not list(mask_index):
            print(tokenizer.convert_ids_to_tokens(inp_ids))
            break
        mask_token_indices.append(mask_index)
        # print(tokenizer.convert_ids_to_tokens(inp_ids))
        # break
    
    # try:
    return batch_input_ids, torch.tensor(mask_token_indices)
    # except TypeError as e:
    #     # print(mask_token_indices)
    #     print(e)
    # return batch_input_ids, mask_token_indices

def rich_compare(preds, labels):
    mask = []

    for pred, label in zip(preds, labels):
        pred = pred.lower()
        label = label.lower()
        mask.append(label in pred) # for more thorough comparison, do (pred == label or pred[1:] == label or pred == label[1:])

    return torch.BoolTensor(mask).to(device)

def get_predictions(model, tokenizer, dataloader):  
    all_correct = 0
    tot_samples = 0

    with torch.no_grad():
            all_correct_samples_mask = []
            all_correct_labels = []
            all_preds = []
            for i, batch in tqdm(enumerate(dataloader)):
                for x in range(len(batch)):
                    batch[x] = batch[x].to(device)
                logits = model(input_ids= batch[0],attention_mask = batch[1]).logits
                soft_preds = torch.nn.functional.softmax(logits, dim=-1)
                pred_token_ids = torch.tensor([soft_preds[i, batch[2][i]].argmax(axis=-1) for i in range(soft_preds.shape[0])], device=device)
                preds = tokenizer.convert_ids_to_tokens(pred_token_ids)
                labs = batch[3]

                corr_mask = rich_compare(preds, tokenizer.convert_ids_to_tokens(labs))
                
                tot_correct = corr_mask.count_nonzero().item()
                # corr_mask = torch.eq(pred_token_ids, batch[3])
                correct_labels = tokenizer.convert_ids_to_tokens(torch.masked_select(labs.unsqueeze(-1), corr_mask.unsqueeze(-1)))
                corr_mask = corr_mask.detach().tolist()
                all_preds += preds
                all_correct_samples_mask += corr_mask
                all_correct += tot_correct
                all_correct_labels += correct_labels
                tot_samples+= batch[0].shape[0]
                
            acc = 0 if not all_correct else all_correct / tot_samples
            print(f"Total samples : {tot_samples}")
            print(f"Correctly predicted : {all_correct}")
            
            return acc, all_correct_samples_mask, all_correct_labels, all_preds

# just dumping code from filter_correct.ipynb
print("Loading data")
sentences, labels, datasets, prompts = [], [], [], []
for relation in N_1_RELATIONS:
    s, l, _, d, p = preparedata(PATTERN_FILE_FORMAT.format(relation), VOCAB_FILE_FORMAT.format(relation))
    sentences += s
    labels += l
    datasets += d
    prompts += p
labels = np.asarray(labels)

tokenizer = BertTokenizer.from_pretrained("bert-large-cased", cache_dir=CACHE_DIR)
labels_tok_indices = []
indices = []
for i, (label, sentence) in enumerate(zip(labels, sentences)):
    token_ids = tokenizer(label, return_attention_mask=False, add_special_tokens=False, return_token_type_ids=False)['input_ids']
    # sentence_ids = tokenizer(sentence, return_attention_mask=False, add_special_tokens=False, return_token_type_ids=False)['input_ids']
    if len(token_ids) == 1: # 512 is max number of tokens for RoBERTa large and BERT large
        labels_tok_indices.append(token_ids[0])
        indices.append(i)

labels_tok_indices = torch.tensor(labels_tok_indices)
sentences = [sentences[i] for i in indices]
labels = [labels[i] for i in indices]
datasets = [datasets[i] for i in indices]
prompts = [prompts[i] for i in indices]

print("finding mask tokens")
input_ids, mask_token_indices = prep_inputs(sentences, tokenizer)
eval_dataset = TensorDataset(input_ids['input_ids'],input_ids['attention_mask'], mask_token_indices, labels_tok_indices)
eval_dataloader = DataLoader(eval_dataset, sampler = SequentialSampler(eval_dataset), batch_size=8, drop_last=False)
model = BertForMaskedLM.from_pretrained("bert-large-cased", cache_dir=CACHE_DIR)
model = model.to(device)

print("predicting")
acc, correctly_classified_mask, correctly_classified_labels, predictions = get_predictions(model, tokenizer, eval_dataloader)

with open("classified_bert_large_cased.jsonl", "w", encoding="utf-8") as f:
    f.writelines((json.dumps({
            "Index": int(index), 
            "Text": text,
            "Label": lab,
            "Prediction": prediction,
            "Dataset": dataset,
            "Prompt": prompt,
            "Correct": correct
        }) + "\n" for index, text, lab, dataset, prediction, prompt, correct in zip(indices, sentences, labels, datasets, predictions, prompts, correctly_classified_mask)))