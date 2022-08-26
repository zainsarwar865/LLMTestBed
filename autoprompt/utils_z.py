import csv
import copy
import json
import logging
from multiprocessing.sharedctypes import Value
import random
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence

MAX_CONTEXT_LEN = 50

logger = logging.getLogger(__name__)

def add_task_specific_tokens(tokenizer):
	tokenizer.add_special_tokens({"additional_special_tokens" : ['[T]', '[P]', '[Y]']})
	tokenizer.trigger_token = '[T]'
	tokenizer.predict_token = ['P']
	tokenizer.lama_y = ['Y']
	tokenizer.trigger_token_id = tokenizer.convert_tokens_to_ids("[T]")
	tokenizer.predict_token_id = tokenizer.convert_tokens_to_ids("[P]")
	tokenizer.lama_y_token_id = tokenizer.convert_tokens_to_ids("[Y]")


def encode_label(tokenizer, label, tokenize=False):
	if isinstance(label, str):
		if(tokenize):
			tokens = tokenizer.tokenize(labels)
			if(len(tokens) > 1):
				raise ValueError("Label is more than one ")
			if(tokens[0] == tokenizer.unk_token):
				raise ValueError(f"The label {label} maps to an unkown token")
			label = tokens[0]
		encoded = torch.tensor(tokenizer.convert_tokens_to_ids([label])).unsqueeze(0)
	elif isinstance(label, list):
		encoded = torch.tensor(tokenizer.convert_tokens_to_ids(label)).unsqueeze(0)
	elif(isinstance(label, int)):
		encoded = torch.tensor([label]).unsqueeze(0)
	return encoded

class TriggerTemplatizer:
	def __init__(self,
				 template,
				 config,
				 tokenizer,
				 label_field='label',
				 label_map=None,
				 tokenize_labels=False,
				 add_special_tokens=False,
				 use_ctx=False):
	
		if not hasattr(tokenizer, 'predict_token') or not hasattr(tokenizer,'trigger_token'):
			raise ValueError("Missing predict_token and trigger_token")

		self._template = template
		self._config = config
		self._label_field = label_field
		self._tokenizer = tokenizer
		self._label_map = label_map
		self.tokenizer_labels = tokenize_labels
		self._add_special_tokens=False
		self._use_ctx=use_ctx


	@property
	def num_trigger_tokens(self):
		return sum(t == '[T]' for t in  self._template.split(' '))


	def __call__(self, format_kwargs):
		format_kwargs = format_kwargs.copy()
		label = format_kwargs.pop(self._label_field)
		text = self._template.format(**format_kwargs)

		if label is None:
			raise Exception("Bad data format")

		model_inputs = tokenizer.batch_encode_plus(text, 
													add_special_tokens = self._add_special_tokens,
													return_tensors = 'pt')

		input_ids = model_inputs['input_ids']
		trigger_mask = input_ids.eq(self._tokenizer.trigger_token_id)
		predict_mask = inpuy_ids.eq(self._tokenizer.predict_token_id)
		input_ids[predict_mask] = self._tokenizer.predict_token_id

		model_inputs['trigger_mask'] = trigger_mask
		model_inputs['predict_mask'] = predict_mask

				# For relation extraction with BERT, update token_type_ids to reflect the two different sequences
		if self._use_ctx and self._config.model_type == 'bert':
			sep_token_indices = (input_ids.squeeze(0) == self._tokenizer.convert_tokens_to_ids(self._tokenizer.sep_token)).nonzero().flatten()
			sequence_b_indices = torch.arange(sep_token_indices[0], sep_token_indices[1] + 1).long().unsqueeze(0)
			model_inputs['token_type_ids'].scatter_(1, sequence_b_indices, 1)
			
			# Encode the label(s)
		if self._label_map is not None:
			label = self._label_map[label]
		label_id = encode_label(
			tokenizer=self._tokenizer,
			label=label,
			tokenize=self._tokenize_labels
		)

		return model_inputs, label_id


def pad_squeeze_sequence(sequence, *args, **kwargs):
	return pad_sequence([seq.squeeze(0) for seq in sequence], *args, *kwargs)



class Collator:
	def __init__(self, pad_token_id = 0):
		self._pad_token_id = pad_token_id
	
	def __call__(self, features):
		model_inputs, labels = list(zip(*features))
		padded_inputs = {}
		proto_input = model_inputs[0]
		keys = list(proto_input.keys())

		for key in keys:

			if key == 'input_ids':
				padding_value = self._pad_token_id
			else:
				padding_value = 0
			sequences = [x[key] for x in model_inputs]
			padded = pad_squeeze_sequence(sequence, batch_first=True, padding_value=padding_value)
			padded_inputs[key] = padded
		
		labels = pad_squeeze_sequence(abels, batch_first=True, padding_value=0)
		return padded_inputs, labels
				

def load_tsv(fname):
	with open(fname, 'r') as f:
		reader = csv.DictReader(f, delimiter='\t')
		for row in reader:
			yield row


def load_jsonl(fname):
	with open(fname, 'r') as f:
		for line in f:
			yield json.loads(line)


LOADERS = {
	".tsv" : load_tsv,
	".jsonl" : load_jsonl
}


def load_trigger_dataset(fname, templatizer, use_ctx, limit=None):
    loader = LOADERS[fname.suffix]
    instances = []

    for x in loader(fname):
        try:
            if use_ctx:
                pass
            else:
                model_inputs, label_ids = templatizer(x)
        except ValueError as e:
            logger.warning('Encountered error "%s" when processing "%s".  Skipping.', e, x)
            continue
        else:   
            instances.append((model_inputs, label_ids))    
    if limit:
        return random.sample(instances, limit)
    else:
        
        return instances
 