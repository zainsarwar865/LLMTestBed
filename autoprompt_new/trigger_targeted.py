#%set_env TRANSFORMERS_CACHE=/bigstor/zsarwar/models/cache
#%set_env CUDA_VISIBLE_DEVICES=0
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
import spacy
from spacy import displacy
import nltk 

from transformers import set_seed as ss

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

class BasePredictWrapper:
    def __init__(self, model):
        self._model = model

    def __call__(self, model_inputs, trigger_ids):
        model_inputs = model_inputs.copy()
        predict_mask = model_inputs.pop('predict_mask')
        logits = self._model(**model_inputs).logits
        predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        return predict_logits, model_inputs

class PredictWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """
    def __init__(self, model):
        self._model = model

    def __call__(self, model_inputs, trigger_ids):
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        predict_mask = model_inputs.pop('predict_mask')
        logits = self._model(**model_inputs).logits
        predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        return predict_logits, model_inputs

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
    config = AutoConfig.from_pretrained(model_name)
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
    ss(seed)

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
            gradient_dot_embedding_matrix += filter
            
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

def is_all_capps_or_num(idx, tokenizer):
    word = tokenizer.decode([idx])
    word = word.replace(" ", "")
    _is_all_caps_nums = False
    word_upper = word.upper()
    if(word_upper == word):
        _is_all_caps_nums = True
    # Check if it contains a number    
    if (any(char.isdigit() for char in word)):
        _is_all_caps_nums = True
    return _is_all_caps_nums


def is_non_lowercase_word(idx, tokenizer):
    non_lc = False
    word = tokenizer.decode([idx])
    res = re.search("^[a-z]+$", word)
    if(not res):
        non_lc = True
    return non_lc

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
    if("token_type_ids" in out):
        new_tok_type_ids = torch.zeros(new_trigger_mask.shape, device=device, dtype=torch.int32)
        out['token_type_ids'] = new_tok_type_ids
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
    base_predictor = BasePredictWrapper(model)

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
        model=args.model_name,
        label_map=label_map,
        label_field=args.label_field,
        tokenize_labels=args.tokenize_labels,
        add_special_tokens=False,
        remove_periods=args.remove_periods,
        replace_period_with_comma=args.replace_period_with_comma,
        use_ctx=args.use_ctx
    )

    base_templatizer = utils_v4.BaseTemplatizer(
        args.base_template,
        config,
        tokenizer,
        model=args.model_name,
        label_map=label_map,
        label_field=args.label_field,
        tokenize_labels=args.tokenize_labels,
        add_special_tokens=False,
        remove_periods=args.remove_periods,
        replace_period_with_comma=args.replace_period_with_comma,
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
        train_dataset = utils_v4.load_trigger_dataset(args.train, templatizer, start_idx=args.start_idx, end_idx=args.end_idx, use_ctx=args.use_ctx, limit=args.limit)
        base_train_dataset=utils_v4.load_trigger_dataset(args.train, base_templatizer, start_idx=args.start_idx, end_idx=args.end_idx, use_ctx=args.use_ctx, limit=args.limit)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator)
    base_train_loader = DataLoader(base_train_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator)
    
    #allowed_words = ['iPhone', 'McC', 'YouTube', 'McDonald', 'LinkedIn', 'MPs', 'WhatsApp', 'iOS', 'McCain', 'McG', 'McD', 'McConnell', 'McGregor', 'McCarthy', 'iPad', 'LeBron', 'JPMorgan', 'IoT', 'OnePlus', 'realDonaldTrump', 'BuzzFeed', 'iTunes', 'iPhones', 'SpaceX', 'McLaren', 'PhD', 'PlayStation', 'McKin', 'McCabe', 'McCoy', 'TVs', 'FedEx', 'McGr', 'McGu', 'McMahon', 'CEOs', 'McMaster', 'JavaScript', 'WikiLeaks', 'eBay', 'McKenzie', 'McInt', 'BlackBerry', 'McCorm', 'DeVos', 'PayPal', 'MacBook', 'McCull', 'PCs', 'McKay', 'MacDonald', 'McCann', 'McGee', 'NGOs', 'GHz', 'McKenna', 'McCartney', 'HuffPost', 'McGill', 'WiFi', 'McDonnell', 'iPads', 'GoPro', 'iPod', 'MacArthur', 'VMware', 'macOS', 'CDs', 'McAuliffe', 'WordPress', 'iCloud', 'YouTube', 'GeForce', 'GPUs', 'CPUs', 'GitHub', 'PowerPoint', 'eSports', 'ObamaCare', 'iPhone', 'UFOs', 'mRNA', 'StarCraft', 'LinkedIn']
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
        
            if is_all_capps_or_num(idx, tokenizer):
                logger.debug(f"Filtered {word}")
                print(word)
                filter[idx] = 1e32
                    
            if is_non_lowercase_word(idx, tokenizer):
                logger.debug(f"Filtered {word}")
                print(word)
                filter[idx] = 1e32

    # creating the filter for the first iteration of token generation
    first_iter_filter = filter.detach().clone()
    if args.model_name == "roberta-large" or args.model_name == 'bert-large-cased':
        with open(args.filtered_vocab, "r", encoding="utf-8") as f:
            whole_word_tokens = json.load(f)
        for index in range(tokenizer.vocab_size):
            if index not in whole_word_tokens.values():
                first_iter_filter[index] = 1e32
    # end creating first iter filter
    # Save filter
    torch.save(first_iter_filter, f"/home/zsarwar/NLP/autoprompt/data/filters/first_iter_filter_{args.model_name}.pt")
    torch.save(filter, f"/home/zsarwar/NLP/autoprompt/data/filters/filter_{args.model_name}.pt")
    """
    #first_iter_filter = torch.load(f"/home/zsarwar/NLP/autoprompt/data/filters/first_iter_filter_{args.model_name}.pt", map_location=device)
    #filter = torch.load(f"/home/zsarwar/NLP/autoprompt/data/filters/filter_{args.model_name}.pt", map_location=device)
        
    if(args.filter): 
        filter = torch.load(f"/home/zsarwar/NLP/autoprompt/data/filters/filter_no_special_tokens_only_lowercase_tokens_{args.model_name}.pt", map_location=device)
    else:
        filter = torch.load(f"/home/zsarwar/NLP/autoprompt/data/filters/filter_no_special_tokens_{args.model_name}.pt", map_location=device)

    all_model_inputs_base = []
    all_labels_base = []
    all_pred_labels_base = []
    all_indices_base = []
    all_probs_base = []
    all_losses_base = []
    logger.info('Evaluating real baseline')
    numerator = 0
    numerator_acc = 0
    denominator = 0
    for idx, (model_inputs, labels) in tqdm(enumerate(base_train_loader)):
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        all_indices_base.append(idx)
        labels = labels.to(device)
        with torch.no_grad():
            predict_logits, m_inputs = base_predictor(model_inputs, trigger_ids)
            eval_metric = evaluation_fn(predict_logits, labels)
            all_losses_base.append(eval_metric.to("cpu"))
            all_probs_base.append(predict_logits[0].to("cpu"))
        pred_label = get_pred_label(predict_logits, labels, tokenizer)
        logger.info(f"Index : {idx}")
        logger.info(f"Input : {tokenizer.decode(m_inputs['input_ids'][0])}")
        logger.info(f"Label : {tokenizer.decode(labels[0])}")
        logger.info(f"Pred : {tokenizer.decode(pred_label[0])}")
        logger.info(f"\n\n")
        m_inputs = { k : v.to("cpu") for k, v in m_inputs.items()}
        all_model_inputs_base.append(m_inputs)
        all_labels_base.append(labels.to("cpu"))
        all_pred_labels_base.append(pred_label.to("cpu"))
        numerator += evaluation_fn(predict_logits, labels).sum().item()
        denominator += labels.size(0)
        numerator_acc += compute_accuracy(predict_logits, labels)
    dev_metric = numerator / (denominator + 1e-13)
    acc_metric_base = numerator_acc / (denominator + 1e-13)
    logger.info(f'Dev metric base: {dev_metric}')
    logger.info(f'Dev acc metric real baseline is : {acc_metric_base}')
    best_dev_metric = 10
    best_dev_acc_metric = 1
    all_model_inputs_template = []
    all_labels_template = []
    all_pred_labels_template = []
    all_indices_template = []
    all_probs_template = []
    all_losses_template = []
    logger.info('Evaluating baseline with template')
    logger.info(f"Baseline trigger ids are : {trigger_ids}")
    numerator = 0
    numerator_acc = 0
    denominator = 0
    for idx, (model_inputs, labels) in tqdm(enumerate(train_loader)):
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        all_indices_template.append(idx)
        labels = labels.to(device)
        with torch.no_grad():
            predict_logits, m_inputs = predictor(model_inputs, trigger_ids)
            eval_metric = evaluation_fn(predict_logits, labels)
            all_losses_template.append(eval_metric.to("cpu"))
            all_probs_template.append(predict_logits[0].to("cpu"))
        pred_label = get_pred_label(predict_logits, labels, tokenizer)
        logger.info(f"Index : {idx}")
        logger.info(f"Input : {tokenizer.decode(m_inputs['input_ids'][0])}")
        logger.info(f"Label : {tokenizer.decode(labels[0])}")
        logger.info(f"Pred : {tokenizer.decode(pred_label[0])}")
        logger.info(f"\n\n")
        m_inputs = { k : v.to("cpu") for k, v in m_inputs.items()}  
        all_model_inputs_template.append(m_inputs)
        all_labels_template.append(labels.to("cpu"))
        all_pred_labels_template.append(pred_label.to("cpu"))
        numerator += evaluation_fn(predict_logits, labels).sum().item()
        denominator += labels.size(0)
        numerator_acc += compute_accuracy(predict_logits, labels)
    dev_metric = numerator / (denominator + 1e-13)
    acc_metric_base = numerator_acc / (denominator + 1e-13)
    logger.info(f'Template Dev metric: {dev_metric}')
    logger.info(f'Template Dev acc metric baseline is : {acc_metric_base}')
    best_dev_metric = 10
    best_dev_acc_metric = 1

    # precalculating the normalized embeddings
    embed_norm = torch.linalg.vector_norm(embeddings.weight, dim=1)
    normalized_embedding_weights = torch.transpose(
        torch.divide(torch.transpose(embeddings.weight, 0, 1), embed_norm),
        0,
        1
    )
    if args.include_gpt:
        # intializing GPT-2
        gpt_model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
        gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
        gpt_tokenizer.pad_token_id = gpt_tokenizer.eos_token_id
        gpt_tokenizer.padding_side = "left"
        gpt_model = gpt_model.to(device)
    
    all_model_inputs_triggers = []
    all_labels_triggers = []
    all_pred_labels_triggers = []
    all_gpt_encodings = []
    all_gpt_generations = []
    all_adv_tokens = []
    all_indices_triggers = []
    all_sub_indices_triggers = []
    all_first_success_ranks = []
    all_probs_triggers = []
    all_losses_triggers = []
    all_gradient_vectors = []

    target_labels_dict = np.load("/home/zsarwar/NLP/autoprompt/data/labels/freq_labels.npy", allow_pickle=True)
    target_labels_dict = target_labels_dict.item()
    curr_relation = str(args.train).split(".")[0].split("_")[-1]
    target_labels_list = torch.tensor(list(target_labels_dict[curr_relation].keys())[0], device=device).unsqueeze(0).unsqueeze(0)

    logger.info(f"Target label : {tokenizer.convert_ids_to_tokens(target_labels_list[0])}")
    new_example = True
    total_samples = 0
    total_incorrect = 0
    model.zero_grad()
    averaged_grad = None
    # Accumulate
    for idx, (model_inputs, labels) in tqdm(enumerate(train_loader)):
        logger.info(f"Total successes  : {total_incorrect}")
        curr_losses = []
        new_example=True
        total_samples+=1    
        # Start from scratch for each example
        all_indices_triggers.append(idx)
        trigger_ids = init_ids.clone()
        tgt_labels = target_labels_list
        model.zero_grad()
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        with torch.no_grad():   
            predict_logits, _ = predictor(model_inputs, trigger_ids)
            eval_metric = evaluation_fn(predict_logits, tgt_labels)
            eval_acc_metric = compute_accuracy(predict_logits, tgt_labels)
        for token_to_flip in range(templatizer.num_trigger_tokens):
            model.zero_grad()
            predict_logits, _ = predictor(model_inputs, trigger_ids)
            loss = get_loss(predict_logits, tgt_labels  ).mean()
            loss.backward()
            grad = embedding_gradient.get()
            bsz, _, emb_dim = grad.size()
            selection_mask = model_inputs['trigger_mask'].unsqueeze(-1)
            grad = torch.masked_select(grad, selection_mask)
            grad = grad.view(bsz, templatizer.num_trigger_tokens, emb_dim)
            averaged_grad = grad.sum(dim=0)
            all_gradient_vectors.append(averaged_grad.to("cpu"))
            all_labels_triggers.append(labels.to("cpu"))
            # Compute adv tokens in any case                
            if(not args.random_tokens): 
                candidates = hotflip_attack(averaged_grad[token_to_flip],
                                            normalized_embedding_weights,
                                            increase_loss=False,
                                            num_candidates=args.num_cand,
                                            filter=filter)
                #candidates = torch.tensor([1699], device=device)
            else:
                candidates = torch.randint(150, 28896, (args.num_cand,), device=device)
            all_adv_tokens.append(candidates.to("cpu"))
            current_score = 0
            current_acc = 0
            candidate_scores = torch.zeros(args.num_cand, device=device)
            candidate_accs = torch.zeros(args.num_cand, device=device)
            candidate_pred_labels = torch.zeros(args.num_cand, device=device, dtype=int)
            denom = 0
            all_candidates = []
            entire_text = []
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
            if(args.template_trigger_phrase):
                original_prompts = tokenizer.batch_decode(original_prompt_ids, skip_special_tokens=False)
            else:
                original_prompts = tokenizer.batch_decode(original_prompt_ids, skip_special_tokens=True)
            candidates_strs = tokenizer.batch_decode(candidates.unsqueeze(1))
            if(args.include_adv_token):
                if("roberta" in args.model_name):
                    if(args.template_trigger_phrase):
                        pre_text = [original_prompts[i][::-1].replace('[Trigger_Token]'[::-1], candidates_strs[i][::-1], 1)[::-1] for i in range(len(original_prompts))]
                    else:
                        pre_text = [candidates_strs[i] + original_prompts[i] for i in range(len(original_prompts))]
                elif("bert" in args.model_name):
                    if(args.template_trigger_phrase):
                        pre_text = [original_prompts[i][::-1].replace('[Trigger_Token]'[::-1], candidates_strs[i][::-1], 1)[::-1] for i in range(len(original_prompts))]
                    else:
                        pre_text = [original_prompts[i] + " " + candidates_strs[i] for i in range(len(original_prompts))]    
            else:
                pre_text = [original_prompts[i] for i in range(len(original_prompts))]

            skip_indices = []
            curr_attempt = 0
            found_adv_gen = False
            non_gpt_session=False 
            batch_gpt_encoded_prompts = []
            batch_gpt_tokens = []
            batch_curr_inputs = []
            batch_curr_pred_labels = []
            batch_sub_indices = []
            batch_probs = []
            while(curr_attempt < args.tot_gpt_attempts and not found_adv_gen):
                if(args.include_gpt):
                    # Encode for GPT-2 Generations and generate
                    gpt_encoded_prompts = gpt_tokenizer.batch_encode_plus(pre_text, add_special_tokens=True, return_attention_mask=True, padding='longest', return_tensors='pt').to(device) 
                    batch_gpt_encoded_prompts.append(batch_gpt_encoded_prompts.append({k : v.to("cpu") for k, v in gpt_encoded_prompts.items()}))
                    #all_gpt_encodings.append(gpt_encoded_prompts)
                    with torch.no_grad():
                        set_seed(curr_attempt)
                        gpt_outputs = gpt_model.generate(inputs=gpt_encoded_prompts['input_ids'], attention_mask=gpt_encoded_prompts['attention_mask'], do_sample=True, top_p=0.96, output_scores=False, return_dict_in_generate=True, max_length=100)
                    num_tokens = gpt_encoded_prompts['input_ids'][0].numel()
                    # Need Entire GPT-2 Text here for entire text
                    gpt_all_tokens = gpt_outputs['sequences']
                    batch_gpt_tokens+=(gpt_all_tokens.to("cpu"))
                    #all_gpt_generations.append(gpt_all_tokens)
                    gpt_all_tokens_str = gpt_tokenizer.batch_decode(gpt_all_tokens, skip_special_tokens=True)
                    #NLTK for all_tokens    
                    gpt_all_str_sents = [tokenize.sent_tokenize(sent) for sent in gpt_all_tokens_str]
                    
                    if(args.remove_periods or args.replace_period_with_comma):
                        gpt_gen_with_og = [all_sents[0] for all_sents in gpt_all_str_sents]
                    else:
                        gpt_gen_with_og = [' '.join(all_sents[0:2]) for all_sents in gpt_all_str_sents]             
                    entire_text = gpt_gen_with_og
                    # Separate the newly generated tokens
                    gpt_new_tokens = gpt_outputs['sequences'][:, num_tokens:]
                    gpt_new_tokens_str = gpt_tokenizer.batch_decode(gpt_new_tokens, skip_special_tokens=True)
                    # NLTK for new_tokens
                    gpt_new_str_sents = [tokenize.sent_tokenize(sent) for sent in gpt_new_tokens_str]
                    gpt_gen = [all_sents[0] if(len(all_sents) >= 1 ) else "SKIPPING" for all_sents in gpt_new_str_sents]
                    skip_indices = [i for i, sent in enumerate(gpt_gen) if sent == "SKIPPING"]
                    # Maybe add length check?
                elif(args.include_adv_token):
                    skip_indices = []
                    entire_text = pre_text
                #TODO Add condition for wikipedia        
                # Retokenize trigger tokens into bert with adv. tokens or gpt_generations or wikipedia text or any combination of them.
                # Everything needs to be in text here    
                # Insert trigger token in the beginning
                if(args.include_adv_token):
                    if(args.include_gpt):
                        if("roberta" in args.model_name):
                            final_trigg_text = [candidates_strs[i] + gpt_gen[i] for i in range(len(gpt_gen))]
                        elif("bert" in args.model_name):
                            final_trigg_text = [" " + candidates_strs[i] + gpt_gen[i] for i in range(len(gpt_gen))]   
                    else:
                        if("roberta" in args.model_name):
                            final_trigg_text = candidates_strs
                        elif("bert" in args.model_name):
                            final_trigg_text = [" " + candidates_strs[i] for i in range(len(candidates_strs))]
                            # Only so that non-GPT setting does not have to loop
                elif(args.include_gpt):
                    if("roberta" in args.model_name):
                        final_trigg_text = gpt_gen
                    elif("bert" in args.model_name):
                        final_trigg_text = [" " + gpt_gen[i] for i in range(len(gpt_gen))]
                # Tokenize trigger text into to-be-attacked models token ids
                final_trigg_tokens = tokenizer.batch_encode_plus(final_trigg_text, add_special_tokens=False)
                all_candidates = final_trigg_tokens['input_ids']
                # Evaluate with adversarial prompts
                curr_inputs = []
                curr_pred_labels = []
                sub_indices = []
                for j in range(len(all_candidates)): 
                    if j not in skip_indices:
                        trigg_toks = torch.tensor(all_candidates[j], device=device).unsqueeze(0)
                        with torch.no_grad():
                            predict_logits, m_inpts = predictor(model_inputs, trigg_toks)
                            eval_metric = evaluation_fn(predict_logits, tgt_labels)
                            batch_probs.append(predict_logits[0].to("cpu"))
                            curr_losses.append(eval_metric.to("cpu")    )
                            pred_label= get_pred_label(predict_logits, tgt_labels, tokenizer)
                            eval_attack_acc_metric = compute_accuracy(predict_logits, tgt_labels)
                        m_inputs = { k : v.to("cpu") for k, v in m_inputs.items()}    
                        curr_inputs.append(m_inpts)
                        curr_pred_labels.append(pred_label.to("cpu"))
                        candidate_scores[j] = eval_metric.sum()
                        candidate_accs[j] = eval_attack_acc_metric
                        candidate_pred_labels[j] = pred_label
                    else:
                        logger.info("Skipping because of empty generation sequence")
                        batch_probs.append(-100)
                        curr_inputs.append(-100)
                        curr_pred_labels.append(-100)
                        curr_losses.append(-100)
                batch_curr_inputs+=curr_inputs
                batch_curr_pred_labels+=curr_pred_labels
                # Print and save successful prompts
                logger.info(f"Batch  : {curr_attempt}")
                all_cands = torch.where(candidate_accs == 1)[0]
                true_cands = torch.tensor([cand for cand in all_cands if cand not in skip_indices], device=device)
                if(true_cands.shape[0] >= 1):    
                    first_succ_idx = true_cands[0] + args.num_cand*curr_attempt
                    logger.info(f"Index  : {idx}")
                    logger.info(f"Original  : {original_prompts[0]}")
                    real_label = tokenizer.convert_ids_to_tokens(labels)
                    target_label = tokenizer.convert_ids_to_tokens(tgt_labels)
                    logger.info(f"Target label : {target_label}")
                    for index in true_cands:
                        sub_indices.append(index)
                        if(not found_adv_gen):
                            total_incorrect+=1
                            all_first_success_ranks.append(first_succ_idx.to("cpu"))
                            found_adv_gen = True
                        adv_lab = candidate_pred_labels[index].item()
                        #Replace only the first instance of the true label with the predicted (adversarial) label
                        if(args.template_trigger_phrase):
                            encoded_entire_text = tokenizer.encode(entire_text[index], add_special_tokens=False)
                            encoded_entire_text[rep_token_idx] = adv_lab
                            entire_text[index] = tokenizer.decode(encoded_entire_text, skip_special_tokens=False)
                        else:
                            encoded_entire_text = tokenizer.encode(entire_text[index])
                            encoded_entire_text[rep_token_idx] = adv_lab
                            entire_text[index] = tokenizer.decode(encoded_entire_text, skip_special_tokens=True)

                        adv_text_pred = entire_text[index]
                        trigger_ids = all_candidates[index]
                        logger.info(f"Adversarial {index + args.num_cand*curr_attempt}: {adv_text_pred}")       
                    logger.info(f"\n\n")
                    #all_sub_indices_triggers.append(sub_indices)
                    batch_sub_indices+=sub_indices
                    break
                else:
                    curr_attempt+=1
                    if(curr_attempt == args.tot_gpt_attempts):
                        all_first_success_ranks.append(-100)
                        break

                    if(not found_adv_gen and not args.include_gpt):
                        all_first_success_ranks.append(-100)
                        break    
            # GPT_Generations finished, now appending batched inputs
            all_gpt_encodings.append(batch_gpt_encoded_prompts)
            all_gpt_generations.append(batch_gpt_tokens)
            all_model_inputs_triggers.append(batch_curr_inputs)
            all_pred_labels_triggers.append(batch_curr_pred_labels)
            all_sub_indices_triggers.append(batch_sub_indices)
            all_losses_triggers.append(curr_losses)
            all_probs_triggers.append(batch_probs)
            break
    
    flip_rate = total_incorrect / total_samples + 1e-32
    logger.info(f"Total incorrect are : {total_incorrect}")
    logger.info(f"Total samples are : {total_samples}")
    logger.info(f"Flip rate is : {flip_rate}")

    all_adversarial_probs = []
    all_real_baseline_probs = []
    all_template_baseline_probs = []
    all_true_labels = []
    all_adv_labels = []
    # Subsample adversarial instances and their first success indices as well as their original counterparts
    for i in range(len(all_model_inputs_base)):
            true_label = all_labels_triggers[i][0]
            adv_labels = all_pred_labels_triggers[i]
            adv_probs = all_probs_triggers[i]
            base_real_probs = all_probs_base[i]
            base_template_probs = all_probs_template[i]
            all_adversarial_probs.append(adv_probs)
            all_real_baseline_probs.append(base_real_probs)
            all_template_baseline_probs.append(base_template_probs)
            all_true_labels.append(true_label)
            all_adv_labels.append(adv_labels)
            
    all_adv_probs = []
    for all_probs in all_adversarial_probs:
        adv_probs_temp = [F.softmax(probs, dim=-1) for probs in all_probs]
        all_adv_probs.append(adv_probs_temp)
    all_adversarial_probs = all_adv_probs
    all_real_baseline_probs = [F.softmax(probs, dim=-1) for probs in all_real_baseline_probs]
    all_template_baseline_probs = [F.softmax(probs, dim=-1) for probs in all_template_baseline_probs]

    top_k = len(all_probs_base[0])
    base_real_top_k = [torch.topk(probs, k=top_k, largest=True) for probs in all_real_baseline_probs]
    base_template_top_k = [torch.topk(probs, k=top_k, largest=True) for probs in all_template_baseline_probs]
    all_adv_top_k = []
    for all_probs in all_adversarial_probs:
        adv_top_k_temp = [torch.topk(probs, k=top_k, largest=True) for probs in all_probs]
        all_adv_top_k.append(adv_top_k_temp)
    adv_top_k = all_adv_top_k

    # Track ranking of baseline top_1 before and after adversarial triggers
    base_real_rank_prob_tracker = []
    for i, label in enumerate(all_true_labels):
        idx_true_base = torch.where(base_real_top_k[i][1] == label)[0]
        prob_true_base = base_real_top_k[i][0][idx_true_base]
        all_trackers = []
        for j in range(len(adv_top_k[i])):
            idx_true_adv = torch.where(adv_top_k[i][j][1] == label)[0]
            prob_true_adv = adv_top_k[i][j][0][idx_true_adv]
            prob_rank_track = [[idx_true_base, prob_true_base], [idx_true_adv, prob_true_adv]]
            all_trackers.append(prob_rank_track)
        base_real_rank_prob_tracker.append(all_trackers)


    # Track ranking of baseline top_1 before and after adversarial triggers
    base_template_rank_prob_tracker = []
    for i, label in enumerate(all_true_labels):
        idx_true_base = torch.where(base_template_top_k[i][1] == label)[0]
        prob_true_base = base_template_top_k[i][0][idx_true_base]
        all_trackers = []
        for j in range(len(adv_top_k[i])):
            idx_true_adv = torch.where(adv_top_k[i][j][1] == label)[0]
            prob_true_adv = adv_top_k[i][j][0][idx_true_adv]
            prob_rank_track = [[idx_true_base, prob_true_base], [idx_true_adv, prob_true_adv]]
            all_trackers.append(prob_rank_track)    
        
        base_template_rank_prob_tracker.append(all_trackers)


    # Track ranking of adv top_1 before and after adversarial triggers
    adv_rank_prob_tracker = []
    for i in range(len(all_adv_labels)):
        all_trackers = []        
        for j in range(len(all_adv_labels[i])):
            idx_true_base_real = torch.where(base_real_top_k[i][1] == all_adv_labels[i][j][0])
            prob_true_base_real = base_real_top_k[i][0][idx_true_base_real]
            idx_true_base_template = torch.where(base_template_top_k[i][1] == all_adv_labels[i][j][0])
            prob_true_base_template = base_template_top_k[i][0][idx_true_base_template]
            idx_true_adv = torch.where(adv_top_k[i][j][1] == all_adv_labels[i][j][0])
            prob_true_adv = adv_top_k[i][j][0][idx_true_adv]
            prob_rank_track = [(idx_true_base_real, prob_true_base_real),  (idx_true_base_template, prob_true_base_template), (idx_true_adv, prob_true_adv)]
            all_trackers.append(prob_rank_track)
        adv_rank_prob_tracker.append(all_trackers)

    top_k = 1000
    base_real_top_k = [torch.topk(probs, k=top_k, largest=True) for probs in all_real_baseline_probs]
    base_template_top_k = [torch.topk(probs, k=top_k, largest=True) for probs in all_template_baseline_probs]
    all_adv_top_k = []
    for all_probs in all_adversarial_probs:
        adv_top_k_temp = [torch.topk(probs, k=top_k, largest=True) for probs in all_probs]
        all_adv_top_k.append(adv_top_k_temp)
    adv_top_k = all_adv_top_k



    
    # Saving results
    all_results_dict = {}
    results_baseline_real = {}
    results_baseline_real['all_model_inputs_base'] = all_model_inputs_base
    results_baseline_real['all_labels_base'] = all_labels_base
    results_baseline_real['all_pred_labels_base'] = all_pred_labels_base
    results_baseline_real['all_indices_base'] = all_indices_base
    #results_baseline_real['all_probs_base'] = all_probs_base
    results_baseline_real['all_losses_base'] = all_losses_base
    results_baseline_real['base_real_top_k'] = base_real_top_k
    results_baseline_real['base_real_rank_prob_tracker'] = base_real_rank_prob_tracker

    all_results_dict['results_baseline_real'] = results_baseline_real

    results_baseline_template = {}
    results_baseline_template['all_model_inputs_template'] = all_model_inputs_template
    results_baseline_template['all_labels_template'] = all_labels_template
    results_baseline_template['all_pred_labels_template'] = all_pred_labels_template
    results_baseline_template['all_indices_template'] = all_indices_template
    #results_baseline_template['all_probs_template'] = all_probs_template
    results_baseline_template['all_losses_template'] = all_losses_template
    results_baseline_template['base_template_top_k'] = base_template_top_k
    results_baseline_template['base_template_rank_prob_tracker'] = base_template_rank_prob_tracker

    all_results_dict['results_baseline_template'] = results_baseline_template

    
    results_adversarial = {}

    results_adversarial['all_model_inputs_triggers'] = all_model_inputs_triggers
    results_adversarial['all_labels_triggers'] = all_labels_triggers
    results_adversarial['all_pred_labels_triggers'] = all_pred_labels_triggers
    results_adversarial['all_gpt_encodings'] = all_gpt_encodings
    results_adversarial['all_gpt_generations'] = all_gpt_generations
    results_adversarial['all_adv_tokens'] = all_adv_tokens
    results_adversarial['all_gradient_vectors'] = all_gradient_vectors
    results_adversarial['all_indices_triggers'] = all_indices_triggers
    results_adversarial['all_sub_indices_triggers'] = all_sub_indices_triggers
    results_adversarial['all_first_success_ranks'] = all_first_success_ranks
    results_adversarial['all_losses_triggers'] = all_losses_triggers
    #results_adversarial['all_probs_triggers'] = all_probs_triggers
    results_adversarial['adv_rank_prob_tracker'] = adv_rank_prob_tracker
    results_adversarial['adv_top_k'] = adv_top_k
    all_results_dict['results_adversarial'] = results_adversarial
    
    np.save(numpy_file, all_results_dict, allow_pickle=True)
    


parser = argparse.ArgumentParser()
parser.add_argument('--label-map', type=str, default=None, help='JSON object defining label map')
# LAMA-specific
parser.add_argument('--tokenize-labels', action='store_true',
                help='If specified labels are split into word pieces.'
                        'Needed for LAMA probe experiments.')
parser.add_argument('--filter', action='store_true', default=False,
                help='If specified, filter out special tokens and gold objects.'
                        'Furthermore, tokens starting with capital '
                        'letters will not appear in triggers. Lazy '
                        'approach for removing proper nouns.')
parser.add_argument('--print-lama', action='store_true',
                help='Prints best trigger in LAMA format.')
parser.add_argument('--logfile', type=str, default='debug_jupyter')
parser.add_argument('--label-field', type=str, default='Prediction',
                help='Name of the label field')
parser.add_argument('--bsz', type=int, default=1, help='Batch size')
parser.add_argument('--eval-size', type=int, default=1, help='Eval size')
parser.add_argument('--iters', type=int, default=1,
                help='Number of iterations to run trigger search algorithm')
parser.add_argument('--accumulation-steps', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--use-ctx', action='store_true',
                help='Use context sentences for relation extraction only')
parser.add_argument('--perturbed', action='store_true',
                help='Perturbed sentence evaluation of relation extraction: replace each object in dataset with a random other object')
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--sentence-size', type=int, default=50)
parser.add_argument('--debug', action='store_true')
# Arguments needed in bashfile
parser.add_argument('--train', type=Path)
parser.add_argument('--template', type=str,default='<s>{Pre_Mask}[Predict_Token]{Post_Mask}[Trigger_Token]</s>', help='Template string', required=False)
parser.add_argument('--base_template', type=str,default='<s>{Pre_Mask}[Predict_Token]{Post_Mask}</s>', help='Template string for baseline evaluation', required=False)
parser.add_argument('--filtered_vocab', type=str, default=None, help='JSON object defining label map')
parser.add_argument('--model-name', type=str, default='bert-large-cased')
parser.add_argument('--include_gpt', action='store_true', default=False)
parser.add_argument('--include_adv_token', action='store_true', default=False )
parser.add_argument('--include_wikipedia_padding', action='store_true', default=False )
parser.add_argument('--remove_periods', default=False, action='store_true')
parser.add_argument('--num-cand', type=int, default=10)
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=500)
parser.add_argument('--tot_gpt_attempts', type=int, default=10)
parser.add_argument('--replace_period_with_comma', action='store_true', default=False)
parser.add_argument('--template_trigger_phrase', action='store_true', default=False)
parser.add_argument('--initial-trigger', nargs='+', type=str, default=['this'], help='Manual prompt')
parser.add_argument('--random_tokens', default=False, action='store_true')

args = parser.parse_args()
if args.debug:
        level = logging.DEBUG
else:
        level = logging.INFO
if 'roberta' in args.model_name:
    args.template = "<s>{Pre_Mask}[Predict_Token]{Post_Mask}[Trigger_Token]</s>"
    args.train = Path("/home/zsarwar/NLP/autoprompt/data/datasets/final/roberta_large_single_entity_2500.jsonl")
    args.base_template = '<s>{Pre_Mask}[Predict_Token]{Post_Mask}</s>'
elif 'bert' in args.model_name:
    if args.template_trigger_phrase:
        args.template = "[CLS]{Pre_Mask}[Predict_Token]{Post_Mask}[SEP]"
        #args.train = Path("/home/zsarwar/NLP/autoprompt/data/datasets/final/correctly_classified_bert_large_cased_with_baseline_and_template_phrase_single_entity.jsonl")
        #args.train = Path("/home/zsarwar/NLP/autoprompt/data/datasets/final/correctly_classified_bert_large_cased_with_baseline_and_template_phrase_single_entity_shuffled.jsonl")
    else:
        args.template = "[CLS]{Pre_Mask}[Predict_Token]{Post_Mask}[Trigger_Token][SEP]"
        #args.train = Path("/home/zsarwar/NLP/autoprompt/data/datasets/final/bert_large_cased_2500.jsonl")

    args.base_template = "[CLS]{Pre_Mask}[Predict_Token]{Post_Mask}[SEP]"
logfile = "/home/zsarwar/NLP/autoprompt/autoprompt/Results/Logs/Relationships/targeted/"+ str(args.train).split("/")[-1].split(".")[0]  +  "_" + args.logfile    
numpy_file = "/home/zsarwar/NLP/autoprompt/autoprompt/Results/Arrays/Relationships/targeted/" + str(args.train).split("/")[-1].split(".")[0]  +  "_" + args.logfile + ".npy"
logging.basicConfig(filename=logfile,level=level)
if(args.template_trigger_phrase):
    import autoprompt.utils_v4_extended as utils_v4
else:
    import autoprompt.utils_v4

run_model(args)