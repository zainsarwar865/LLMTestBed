#%set_env TRANSFORMERS_CACHE=/bigstor/zsarwar/models/cache
#%set_env CUDA_VISIBLE_DEVICES=3
import time
import argparse
import json
import logging
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelWithLMHead, AutoModelForMaskedLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
from nltk import tokenize
import autoprompt.utils_v4 as utils_v4
import spacy
from spacy import displacy
import nltk

NER = spacy.load("en_core_web_sm")
logger = logging.getLogger(__name__)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_full_backward_hook(self.hook)
        # module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient

class PredictWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """
    def __init__(self, model):
        self._model = model

    def __call__(self, model_inputs, trigger_ids):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        predict_mask = model_inputs.pop('predict_mask')
        logits = self._model(**model_inputs).logits
        predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        return predict_logits

class AccuracyFn:
    """
    Computing the accuracy when a label is mapped to multiple tokens is difficult in the current
    framework, since the data generator only gives us the token ids. To get around this we
    compare the target logp to the logp of all labels. If target logp is greater than all (but)
    one of the label logps we know we are accurate.
    """
    def __init__(self, tokenizer, label_map, device, tokenize_labels=False):
        self._all_label_ids = []
        self._pred_to_label = []
        logger.info(label_map)
        for label, label_tokens in label_map.items():
            self._all_label_ids.append(utils.encode_label(tokenizer, label_tokens, tokenize_labels).to(device))
            self._pred_to_label.append(label)
        logger.info(self._all_label_ids)

    def __call__(self, predict_logits, gold_label_ids):
        # Get total log-probability for the true label
        gold_logp = get_loss(predict_logits, gold_label_ids)

        # Get total log-probability for all labels
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]

        # Add up the number of entries where loss is greater than or equal to gold_logp.
        ge_count = all_label_logp.le(gold_logp.unsqueeze(-1)).sum(-1)
        correct = ge_count.le(1)  # less than in case of num. prec. issues

        return correct.float()

    # TODO: @rloganiv - This is hacky. Replace with something sensible.
    def predict(self, predict_logits):
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]
        return predictions


def load_pretrained(model_name):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    config = AutoConfig.from_pretrained(model_name )
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    utils_v4.add_task_specific_tokens(tokenizer)
    return config, model, tokenizer


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_embeddings(model, config):
    """
    Returns the wordpiece embedding module.
    """
    base_model = getattr(model, config.model_type)
    embeddings = base_model.embeddings.word_embeddings
    return embeddings

def compute_accuracy(predict_logits, labels):
    target_logp = F.log_softmax(predict_logits, dim=-1)
    max_pred = torch.argmax(target_logp, dim=-1).unsqueeze(-1)
    mask = max_pred.eq(labels)
    correct = mask.nonzero().shape[0]
    total = labels.shape[0]
    acc = correct / total
    return correct

def hotflip_attack(averaged_grad,
                   normalized_embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            normalized_embedding_matrix,
            averaged_grad
        )

        if filter is not None:
            gradient_dot_embedding_matrix -= filter
            
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1

    _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)
    return top_k_ids

def get_pred_label(predict_logits, labels, tokenizer):
    target_logp = F.log_softmax(predict_logits, dim=-1)
    max_pred = torch.argmax(target_logp, dim=-1).unsqueeze(-1)
    return max_pred

def get_loss(predict_logits, label_ids):
    predict_logp = F.log_softmax(predict_logits, dim=-1)
    target_logp = predict_logp.gather(-1, label_ids)
    target_logp = target_logp - 1e32 * label_ids.eq(0)  # Apply mask
    target_logp = torch.logsumexp(target_logp, dim=-1)
    return -target_logp

def isVariable(idx, tokenizer, allowed_words):
    word = tokenizer.decode([idx])
    word = word.replace(" ", "")
    _isVar = False
    upper_locs = [i for i, ch in enumerate(word) if ch.isupper()]
    # Check if caps in between and entire word is not upper-case
    if(len(upper_locs) > 0 and len(upper_locs) < len(word)):
        for idx in upper_locs:
            if (idx > 0):
            # Check if token is not real entity like McDonalds                
                parsed_word= NER(word)
                if (len(parsed_word.ents) == 0):
                    if(word not in allowed_words):
                        _isVar = True
                    break 
    return _isVar

def replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask):
    out = model_inputs.copy()    
    # Count number of false values
    new_len = (torch.count_nonzero(trigger_mask.eq(False)) + trigger_ids.shape[1]).item()
    # New trigger mask
    new_trigger_mask = torch.zeros(new_len, dtype=torch.bool, device=device).unsqueeze(0)
    # Get index of first true element in the old mask and fill in new_trigger_mask
    trigger_start_index = torch.where(trigger_mask == True)[1][0].item()
    new_trigger_mask[0][trigger_start_index: trigger_start_index + trigger_ids.shape[1]] = True
    # New input_ids_tensor
    new_input_ids = torch.full(new_trigger_mask.shape, fill_value=-1, device=device)
    # Fill in og ids
    og_text_ids = (torch.masked_select(out['input_ids'], trigger_mask.eq(False)))
    new_input_ids.masked_scatter_(new_trigger_mask.eq(False), og_text_ids)
    # Fill in new trigger_ids
    new_input_ids.masked_scatter_(new_trigger_mask, trigger_ids)
    # New prediction mask
    new_pred_mask = torch.full(new_trigger_mask.shape, fill_value=0, device=device,dtype=torch.bool)
    # Need to check for number of trigger tokens in both masks
    pred_mask_true_index = torch.where(out['predict_mask'])[1][0].item()
    num_trig_tokens_old = torch.count_nonzero(trigger_mask)
    num_trig_tokens_new = torch.count_nonzero(new_trigger_mask)
    diff = num_trig_tokens_new - num_trig_tokens_old
    if(trigger_start_index > pred_mask_true_index):
        # Copy/paste into the same index as is
        new_pred_mask[0][pred_mask_true_index] = True
    else:
        new_pred_mask[0][pred_mask_true_index + diff] = True
    # Finally, a new attention mask is also needed
    new_attention_mask = torch.full(new_input_ids.shape, fill_value=1, device=device)
    out['input_ids'] = new_input_ids
    out['predict_mask'] = new_pred_mask
    out['attention_mask'] = new_attention_mask    
    return out

def run_model(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading model, tokenizer, etc.')
    config, model, tokenizer = load_pretrained(args.model_name)
    model.to(device)
    embeddings = get_embeddings(model, config)
    embedding_gradient = GradientStorage(embeddings)
    predictor = PredictWrapper(model)

    if args.label_map is not None:
        label_map = json.loads(args.label_map)
        logger.info(f"Label map: {label_map}")
    else:
        label_map = None
        logger.info('No label map')
    templatizer = utils_v4.TriggerTemplatizer(
        args.template,
        config,
        tokenizer,
        label_map=label_map,
        label_field=args.label_field,
        tokenize_labels=args.tokenize_labels,
        add_special_tokens=False,
        use_ctx=args.use_ctx
    )
    # Obtain the initial trigger tokens and label mapping
    if args.initial_trigger:   
        initial_trigger = args.initial_trigger
        logger.info(f"initial trigger {initial_trigger}")
        logger.info("init ids")
        init_ids = tokenizer.convert_tokens_to_ids(initial_trigger)
        logger.info(init_ids)
        init_ids = torch.tensor(init_ids, device=device).unsqueeze(0)
        logger.info(init_ids)
        trigger_ids = tokenizer.convert_tokens_to_ids(initial_trigger)
        logger.info(f'Initial triggers are the following: {initial_trigger}')
        logger.info(f'Initial Trigger ids are: {trigger_ids}')
        logger.info(f"len trigger ids: {len(trigger_ids)}")
        logger.info(f"num trigger tokens: {templatizer.num_trigger_tokens}")
        assert len(trigger_ids) == templatizer.num_trigger_tokens
    else:
        logger.info(f"no initial trigger provided, using {templatizer.num_trigger_tokens} mask tokens")
        init_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
        init_ids = torch.tensor(init_ids, device=device).unsqueeze(0)
        trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)
    best_trigger_ids = trigger_ids.clone()
    # NOTE: Accuracy can only be computed if a fixed pool of labels is given, which currently
    # requires the label map to be specified. Since producing a label map may be cumbersome (e.g.,
    # for link prediction tasks), we just use (negative) loss as the evaluation metric in these cases.
    if label_map:
        evaluation_fn = AccuracyFn(tokenizer, label_map, device)
    else:
        evaluation_fn = lambda x, y: -get_loss(x, y)
    logger.info('Loading datasets')
    collator = utils_v4.Collator(pad_token_id=tokenizer.pad_token_id)
    if args.perturbed:
        train_dataset = utils_v4.load_augmented_trigger_dataset(args.train, templatizer, limit=args.limit)
    else:
        train_dataset = utils_v4.load_trigger_dataset(args.train, templatizer, use_ctx=args.use_ctx, limit=args.limit)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator)
    if args.perturbed:
        dev_dataset = utils_v4.load_augmented_trigger_dataset(args.train, templatizer)
    else:
        dev_dataset = utils_v4.load_trigger_dataset(args.dev, templatizer, use_ctx=args.use_ctx)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
    allowed_words = ['iPhone', 'McC', 'YouTube', 'McDonald', 'LinkedIn', 'MPs', 'WhatsApp', 'iOS', 'McCain', 'McG', 'McD', 'McConnell', 'McGregor', 'McCarthy', 'iPad', 'LeBron', 'JPMorgan', 'IoT', 'OnePlus', 'realDonaldTrump', 'BuzzFeed', 'iTunes', 'iPhones', 'SpaceX', 'McLaren', 'PhD', 'PlayStation', 'McKin', 'McCabe', 'McCoy', 'TVs', 'FedEx', 'McGr', 'McGu', 'McMahon', 'CEOs', 'McMaster', 'JavaScript', 'WikiLeaks', 'eBay', 'McKenzie', 'McInt', 'BlackBerry', 'McCorm', 'DeVos', 'PayPal', 'MacBook', 'McCull', 'PCs', 'McKay', 'MacDonald', 'McCann', 'McGee', 'NGOs', 'GHz', 'McKenna', 'McCartney', 'HuffPost', 'McGill', 'WiFi', 'McDonnell', 'iPads', 'GoPro', 'iPod', 'MacArthur', 'VMware', 'macOS', 'CDs', 'McAuliffe', 'WordPress', 'iCloud', 'YouTube', 'GeForce', 'GPUs', 'CPUs', 'GitHub', 'PowerPoint', 'eSports', 'ObamaCare', 'iPhone', 'UFOs', 'mRNA', 'StarCraft', 'LinkedIn']
    """
    filter = torch.zeros(tokenizer.vocab_size, dtype=torch.float32, device=device)
    if args.filter:
        logger.info('Filtering label tokens.')
        if label_map:
            for label_tokens in label_map.values():
                label_ids = utils.encode_label(tokenizer, label_tokens).unsqueeze(0)
                filter[label_ids] = 1e32
        else:
            for _, label_ids in train_dataset:
                filter[label_ids] = 1e32
        logger.info('Filtering special tokens and capitalized words.')
        for word, idx in tokenizer.get_vocab().items():
            if len(word) == 1 or idx >= tokenizer.vocab_size:
                continue
            # Filter special tokens.
            if idx in tokenizer.all_special_ids:
                logger.info('Filtered: %s, index: %d', word, idx)
                filter[idx] = 1e32
            
            if isVariable(idx, tokenizer, allowed_words):
                logger.debug(f"Filtered {word}")
                print(word)
                filter[idx] = 1e32


    # creating the filter for the first iteration of token generation
    first_iter_filter = filter.detach().clone()
    if args.model_name == "roberta-large":
        with open("/home/zsarwar/NLP/autoprompt/roberta_full_words_capital_no_diacritic.json", "r", encoding="utf-8") as f:
            whole_word_tokens = json.load(f)
        
        for index in range(tokenizer.vocab_size):
            if index not in whole_word_tokens.values():
                first_iter_filter[index] = 1e32
    # end creating first iter filter

    # Save filter
    torch.save(first_iter_filter, "/home/zsarwar/NLP/autoprompt/data/first_iter_filter.pt")
    torch.save(filter, "/home/zsarwar/NLP/autoprompt/data/filter.pt")
    """
    first_iter_filter = torch.load("/home/zsarwar/NLP/autoprompt/data/first_iter_filter.pt", map_location=device)
    filter = torch.load("/home/zsarwar/NLP/autoprompt/data/filter.pt", map_location=device)
    logger.info('Evaluating baseline')
    logger.info(f"Baseline trigger ids are : {trigger_ids}")
    numerator = 0
    numerator_acc = 0
    denominator = 0
    for model_inputs, labels in tqdm(dev_loader):
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        with torch.no_grad():
            predict_logits = predictor(model_inputs, trigger_ids)
        numerator += evaluation_fn(predict_logits, labels).sum().item()
        denominator += labels.size(0)
        numerator_acc += compute_accuracy(predict_logits, labels)
    dev_metric = numerator / (denominator + 1e-13)
    acc_metric_base = numerator_acc / (denominator + 1e-13)
    logger.info(f'Dev metric: {dev_metric}')
    logger.info(f'Dev acc metric baseline is : {acc_metric_base}')
    best_dev_metric = 10
    best_dev_acc_metric = 1

    
    # Measure elapsed time of trigger search
    start = time.time()
    # precalculating the normalized embeddings
    embed_norm = torch.linalg.vector_norm(embeddings.weight, dim=1)
    normalized_embedding_weights = torch.transpose(
        torch.divide(torch.transpose(embeddings.weight, 0, 1), embed_norm),
        0,
        1
    )
    # intializing GPT-2
    gpt_model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    gpt_tokenizer.pad_token_id = gpt_tokenizer.eos_token_id
    gpt_model = gpt_model.to(device)
    """
    # To deal with special tokens later
    tokenizer_special_tokens = []
    for word, idx in tokenizer.get_vocab().items():
            if idx >= tokenizer.vocab_size:
                continue
            if idx in tokenizer.all_special_ids and word != "":
                tokenizer_special_tokens.append(word)
    for token in tokenizer.additional_special_tokens:
        tokenizer_special_tokens.append(token)
    """
    new_example = True
    total_samples = 0
    total_incorrect = 0
    model.zero_grad()
    averaged_grad = None
    # Accumulate
    for model_inputs, labels in tqdm(train_loader):
        new_example=True
        total_samples+=1    
        # Start from scratch for each example
        trigger_ids = init_ids.clone()
        model.zero_grad()
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        with torch.no_grad():   
            predict_logits = predictor(model_inputs, trigger_ids)
            eval_metric = evaluation_fn(predict_logits, labels)
            eval_acc_metric = compute_accuracy(predict_logits, labels)
        for token_to_flip in range(templatizer.num_trigger_tokens):
            model.zero_grad()
            predict_logits = predictor(model_inputs, trigger_ids)
            loss = get_loss(predict_logits, labels).mean()
            loss.backward()
            grad = embedding_gradient.get()
            bsz, _, emb_dim = grad.size()
            selection_mask = model_inputs['trigger_mask'].unsqueeze(-1)
            grad = torch.masked_select(grad, selection_mask)
            grad = grad.view(bsz, templatizer.num_trigger_tokens, emb_dim)
            averaged_grad = grad.sum(dim=0)
            candidates = hotflip_attack(averaged_grad[token_to_flip],
                                        normalized_embedding_weights,
                                        increase_loss=True,
                                        num_candidates=args.num_cand,
                                        filter=filter if token_to_flip > 0 else first_iter_filter)
            current_score = 0
            current_acc = 0
            candidate_scores = torch.zeros(args.num_cand, device=device)
            candidate_accs = torch.zeros(args.num_cand, device=device)
            candidate_pred_labels = torch.zeros(args.num_cand, device=device, dtype=int)
            denom = 0
            fluent_candidates = []
            fluent_text = []
            # Update current score
            current_acc = eval_acc_metric
            current_score = eval_metric.sum()
            denom = labels.size(0)    
            # Changes start from here
            # Batched og prompts in id form
            original_prompt_ids = model_inputs['input_ids'][0].unsqueeze(0)
            original_prompt_ids = original_prompt_ids.repeat(args.num_cand, 1)
            rep_token_idx = torch.where(original_prompt_ids == tokenizer.mask_token_id)[1][0]
            original_prompt_ids[:,rep_token_idx] = labels[0].item()
            # Batched trigger prompts in id form
            temp_trigger = trigger_ids.clone()
            temp_triggers = temp_trigger.repeat(len(candidates), 1)
            temp_triggers[:, token_to_flip] = candidates
            # Batched og + trigger prompts in text form
            original_prompts = tokenizer.batch_decode(original_prompt_ids, skip_special_tokens=True)
            candidates_strs = tokenizer.batch_decode(candidates.unsqueeze(1))
            temp_strings = [original_prompts[i] + candidates_strs[i] for i in range(len(original_prompts))]
            # Encode for GPT-2 Generations and generate
            gpt_encoded_prompts = gpt_tokenizer.batch_encode_plus(temp_strings, add_special_tokens=True, return_attention_mask=True, padding='longest', return_tensors='pt').to(device) 
            gpt_outputs = gpt_model.generate(inputs=gpt_encoded_prompts['input_ids'], attention_mask=gpt_encoded_prompts['attention_mask'], do_sample=True, top_p=0.96, output_scores=False, return_dict_in_generate=True, max_length=80)
            num_tokens = gpt_encoded_prompts['input_ids'][0].numel()
            # Need Entire GPT-2 Text here for fluent_text
            gpt_all_tokens = gpt_outputs['sequences']
            gpt_all_tokens_str = gpt_tokenizer.batch_decode(gpt_all_tokens, skip_special_tokens=True)
            #NLTK for all_tokens    
            gpt_all_str_sents = [tokenize.sent_tokenize(sent) for sent in gpt_all_tokens_str]
            gpt_all_sents_two = [' '.join(all_sents[0:2]) for all_sents in gpt_all_str_sents]
            fluent_text = gpt_all_sents_two
            # Separate the newly generated tokens
            gpt_new_tokens = gpt_outputs['sequences'][:, num_tokens:]
            gpt_new_tokens_str = gpt_tokenizer.batch_decode(gpt_new_tokens, skip_special_tokens=True)
            # NLTK for new_tokens
            gpt_new_str_sents = [tokenize.sent_tokenize(sent) for sent in gpt_new_tokens_str]
            gpt_new_sents_first = [all_sents[0] if(len(all_sents) >= 1 ) else "SKIPPING" for all_sents in gpt_new_str_sents]
            skip_indices = [i for i, sent in enumerate(gpt_new_sents_first) if sent == "SKIPPING"]
            # Insert trigger token in the beginning
            gpt_new_sents_first = [candidates_strs[i] + gpt_new_sents_first[i] for i in range(len(gpt_new_sents_first))]
            # BERT-ready to be replaced prompts ~ fluent_candidates
            gen_tokens_bert = tokenizer.batch_encode_plus(gpt_new_sents_first, add_special_tokens=False)
            fluent_candidates = gen_tokens_bert['input_ids']
            # Evaluate with adversarial prompts
            for j in range(len(fluent_candidates)):
                if j not in skip_indices:
                    trigg_toks = torch.tensor(fluent_candidates[j], device=device).unsqueeze(0)
                    with torch.no_grad():
                        predict_logits = predictor(model_inputs, trigg_toks)
                        eval_metric = evaluation_fn(predict_logits, labels)
                        pred_label = get_pred_label(predict_logits, labels, tokenizer)
                        eval_attack_acc_metric = compute_accuracy(predict_logits, labels)
                    candidate_scores[j] = eval_metric.sum()
                    candidate_accs[j] = eval_attack_acc_metric
                    candidate_pred_labels[j] = pred_label
                else:
                    logger.info("Skipping because of empty generation sequence")
            # Print and save successful prompts
            if(candidate_accs == 0).any():
                total_incorrect+=1
                #print(f" Original : {original_prompts[0]}")
                logger.info(f"Original  : {original_prompts[0]}")
                real_label = tokenizer.convert_ids_to_tokens(labels)
                for index, candidate_acc in enumerate(candidate_accs):
                    if index not in skip_indices:
                        if candidate_acc != 0:
                            continue
                        adv_lab = candidate_pred_labels[index].item()
                        # Replace only the first instance of the true label with the predicted (adversarial) label
                        adv_text_pred = fluent_text[index].replace(tokenizer.convert_tokens_to_string(real_label[0]), tokenizer.decode(adv_lab), 1).replace("\n", "")
                        trigger_ids = fluent_candidates[index]
                        logger.info(f"Adversarial : {adv_text_pred}")
                        #print(f"Adversarial : {adv_text_pred}")
                logger.info(f"\n\n")
                break
    flip_rate = total_incorrect / total_samples + 1e-32
    logger.info(f"Total incorrect are : {total_incorrect}")
    logger.info(f"Total samples are : {total_samples}")
    logger.info(f"Flip rate is : {flip_rate}")
    #print(f"Flip rate is : {flip_rate}")       
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=Path, default='/home/zsarwar/NLP/autoprompt/data/correctly_classified_roberta_large_autoprompt_format_shorter.jsonl', help='Train data path')
parser.add_argument('--dev', type=Path, default='/home/zsarwar/NLP/autoprompt/data/correctly_classified_roberta_large_autoprompt_format_shorter.jsonl',help='Dev data path')
parser.add_argument('--template', type=str,default='<s> {Pre_Mask}[P]{Post_Mask}[T][T][T][T][T]</s>', help='Template string')
parser.add_argument('--label-map', type=str, default=None, help='JSON object defining label map')
# LAMA-specific
parser.add_argument('--tokenize-labels', action='store_true',
                    help='If specified labels are split into word pieces.'
                            'Needed for LAMA probe experiments.')
parser.add_argument('--filter', action='store_true', default=True,
                    help='If specified, filter out special tokens and gold objects.'
                            'Furthermore, tokens starting with capital '
                            'letters will not appear in triggers. Lazy '
                            'approach for removing proper nouns.')
parser.add_argument('--print-lama', action='store_true',
                    help='Prints best trigger in LAMA format.')
parser.add_argument('--logfile', type=str, default='v5_all')
parser.add_argument('--initial-trigger', nargs='+', type=str, default=None, help='Manual prompt')
parser.add_argument('--label-field', type=str, default='Prediction',
                    help='Name of the label field')
parser.add_argument('--bsz', type=int, default=1, help='Batch size')
parser.add_argument('--eval-size', type=int, default=1, help='Eval size')
parser.add_argument('--iters', type=int, default=1,
                    help='Number of iterations to run trigger search algorithm')
parser.add_argument('--accumulation-steps', type=int, default=1)
parser.add_argument('--model-name', type=str, default='roberta-large',
                    help='Model name passed to HuggingFace AutoX classes.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--use-ctx', action='store_true',
                    help='Use context sentences for relation extraction only')
parser.add_argument('--perturbed', action='store_true',
                    help='Perturbed sentence evaluation of relation extraction: replace each object in dataset with a random other object')
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--num-cand', type=int, default=10)
parser.add_argument('--sentence-size', type=int, default=50)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
if args.debug:
    level = logging.DEBUG
else:
    level = logging.INFO
logfile = "/home/zsarwar/NLP/autoprompt/autoprompt/Results/"+ str(args.train).split("/")[-1].split(".")[0]  +  "_" + args.logfile    
logging.basicConfig(filename=logfile,level=level)
run_model(args)
