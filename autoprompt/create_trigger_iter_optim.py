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
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer
from tqdm import tqdm
    
import autoprompt.utils as utils


logger = logging.getLogger(__name__)


class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_full_backward_hook(self.hook)

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
        predict_mask = model_inputs.pop('predict_mask')
        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        logits, *_ = self._model(**model_inputs)
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
    config = AutoConfig.from_pretrained(model_name, cache_dir='/bigstor/zsarwar/models/cache')
    model = AutoModelWithLMHead.from_pretrained(model_name, cache_dir='/bigstor/zsarwar/models/cache')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, cache_dir='/bigstor/zsarwar/models/cache')
    utils.add_task_specific_tokens(tokenizer)
    return config, model, tokenizer


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_embeddings(model, config):
    """Returns the wordpiece embedding module."""
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
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
            
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        
    embed_norm = torch.linalg.vector_norm(embedding_matrix, dim=1)
    avg_grad_norm = torch.linalg.vector_norm(averaged_grad, dim=0)

    embed_grad_norm = torch.multiply(embed_norm, avg_grad_norm)

    cos_sim = torch.divide(gradient_dot_embedding_matrix, embed_grad_norm)
    _, top_k_ids = cos_sim.topk(num_candidates)

    return top_k_ids

def replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask):
    """Replaces the trigger tokens in input_ids."""
    out = model_inputs.copy()
    input_ids = model_inputs['input_ids']
    trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1)
    try:
        filled = input_ids.masked_scatter(trigger_mask, trigger_ids)
    except RuntimeError:
        filled = input_ids
    out['input_ids'] = filled
    return out


def get_pred_label(predict_logits, labels, tokenizer):
    target_logp = F.log_softmax(predict_logits, dim=-1)
    max_pred = torch.argmax(target_logp, dim=-1).unsqueeze(-1)
    #logger.info(f"max_pred is {max_pred}")
    
    return max_pred



def get_loss(predict_logits, label_ids):
    predict_logp = F.log_softmax(predict_logits, dim=-1)
    target_logp = predict_logp.gather(-1, label_ids)
    target_logp = target_logp - 1e32 * label_ids.eq(0)  # Apply mask
    target_logp = torch.logsumexp(target_logp, dim=-1)
    return -target_logp


def isupper(idx, tokenizer):
    """
    Determines whether a token (e.g., word piece) begins with a capital letter.
    """
    _isupper = False
    # We only want to check tokens that begin words. Since byte-pair encoding
    # captures a prefix space, we need to check that the decoded token begins
    # with a space, and has a capitalized second character.
    if isinstance(tokenizer, transformers.GPT2Tokenizer):
        decoded = tokenizer.decode([idx])
        if decoded[0] == ' ' and decoded[1].isupper():
            _isupper = True
    # For all other tokenization schemes, we can just check the first character
    # is capitalized.
    elif tokenizer.decode([idx])[0].isupper():
            _isupper = True
    return _isupper


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

    templatizer = utils.TriggerTemplatizer(
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
        initial_trigger = ["Ġthe", "Ġthe", "Ġgood", "Ġthe", "Ġgood", "Ġbad"]
        init_ids = tokenizer.convert_tokens_to_ids(initial_trigger)
        init_ids = torch.tensor(init_ids, device=device).unsqueeze(0)
        trigger_ids = tokenizer.convert_tokens_to_ids(initial_trigger)
        logger.info(f'Initial trigger is the following: {initial_trigger}')
        
        logger.info(f'Trigger ids: {trigger_ids}')
        
        assert len(trigger_ids) == templatizer.num_trigger_tokens
    else:
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
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)

    if args.perturbed:
        train_dataset = utils.load_augmented_trigger_dataset(args.train, templatizer, limit=args.limit)
    else:
        train_dataset = utils.load_trigger_dataset(args.train, templatizer, use_ctx=args.use_ctx, limit=args.limit)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    if args.perturbed:
        dev_dataset = utils.load_augmented_trigger_dataset(args.train, templatizer)
    else:
        dev_dataset = utils.load_trigger_dataset(args.dev, templatizer, use_ctx=args.use_ctx)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    # To "filter" unwanted trigger tokens, we subtract a huge number from their logits.
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
                logger.info('Filtered: %s', word)
                filter[idx] = 1e32
            # Filter capitalized words (lazy way to remove proper nouns).
            """
            if isupper(idx, tokenizer):
                logger.debug('Filtered: %s', word)
                filter[idx] = 1e32
            """

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
    
    new_example = True
    for i in range(args.iters):
        total_samples = 0
        total_incorrect = 0
        logger.info(f'Iteration: {i}')
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
        
            # Baseline performance with initial triggers            
            with torch.no_grad():   
                predict_logits = predictor(model_inputs, trigger_ids)
                eval_metric = evaluation_fn(predict_logits, labels)
                eval_acc_metric = compute_accuracy(predict_logits, labels)
            if(eval_acc_metric == 0):
                total_incorrect+=1
                logger.info("Adversarial on init triggers")
                logger.info(f"Accuracy is : {eval_acc_metric}")
                input_text = tokenizer.decode(model_inputs['input_ids'][0])
                real_label = tokenizer.convert_ids_to_tokens(labels)
                pred_label = get_pred_label(predict_logits, labels, tokenizer)
                pred_token = tokenizer.convert_ids_to_tokens(pred_label)
                logger.info(f"Input text : {input_text}")
                logger.info(f"Real label was : {real_label}")
                logger.info(f"Pred label was : {pred_token}")
                logger.info(f"Trigger tokens were : {tokenizer.convert_ids_to_tokens(trigger_ids.squeeze(0))}")
                logger.info(f"\n\n")
                continue

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
                                            embeddings.weight,
                                            increase_loss=True,
                                            num_candidates=args.num_cand,
                                            filter=filter)
                current_score = 0
                current_acc = 0
                candidate_scores = torch.zeros(args.num_cand, device=device)
                candidate_accs = torch.zeros(args.num_cand, device=device)
                candidate_pred_labels = torch.zeros(args.num_cand, device=device, dtype=int)
                denom = 0

            
                # Update current score
                current_acc = eval_acc_metric
                current_score = eval_metric.sum()
                denom = labels.size(0)
                # Actual attack starts
                for i, candidate in enumerate(candidates):
                    temp_trigger = trigger_ids.clone()
                    temp_trigger[:, token_to_flip] = candidate
                    with torch.no_grad():
                        predict_logits = predictor(model_inputs, temp_trigger)
                        eval_metric = evaluation_fn(predict_logits, labels)
                        pred_label = get_pred_label(predict_logits, labels, tokenizer)
                        
                        eval_attack_acc_metric = compute_accuracy(predict_logits, labels)

                    candidate_scores[i] = eval_metric.sum()
                    candidate_accs[i] = eval_attack_acc_metric
                    candidate_pred_labels[i] = pred_label
                    
                
                
                if (candidate_accs == 0).any():
                    total_incorrect+=1
                    best_candidate_acc = candidate_accs.min()
                    best_candidate_idx = candidate_accs.argmin()
                    trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
                    input_text = tokenizer.decode(model_inputs['input_ids'][0])
                    real_label = tokenizer.convert_ids_to_tokens(labels)
                    logger.info(f"Input text : {input_text}")
                    logger.info(f"Real label was : {real_label}")
                    logger.info(f"Pred label was : {tokenizer.convert_ids_to_tokens(candidate_pred_labels[best_candidate_idx].item())}")
                    logger.info(f"Trigger tokens were : {tokenizer.convert_ids_to_tokens(trigger_ids.squeeze(0))}")
                    logger.info(f"\n\n")
                    break

                if (candidate_scores < current_score).any():
                    logger.info('Better trigger with higher loss detected.')
                    best_candidate_score = candidate_scores.min()
                    best_candidate_idx = candidate_scores.argmin()
                    trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
                            

    flip_rate = total_incorrect / total_samples + 1e-32
    trig_tokens = tokenizer.convert_ids_to_tokens(trigger_ids.squeeze(0))
    logger.info(f"Total incorrect are : {total_incorrect}")
    logger.info(f"Total samples are : {total_samples}")
    logger.info(f"Trigger tokens are : {trig_tokens}")
    logger.info(f"Flip rate is : {flip_rate}")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True, help='Train data path')
    parser.add_argument('--dev', type=Path, required=True, help='Dev data path')
    parser.add_argument('--template', type=str, help='Template string')
    parser.add_argument('--label-map', type=str, default=None, help='JSON object defining label map')

    # LAMA-specific
    parser.add_argument('--tokenize-labels', action='store_true',
                        help='If specified labels are split into word pieces.'
                             'Needed for LAMA probe experiments.')
    parser.add_argument('--filter', action='store_true',
                        help='If specified, filter out special tokens and gold objects.'
                             'Furthermore, tokens starting with capital '
                             'letters will not appear in triggers. Lazy '
                             'approach for removing proper nouns.')
    parser.add_argument('--print-lama', action='store_true',
                        help='Prints best trigger in LAMA format.')
    parser.add_argument('--logfile', type=str, required=True, help='Train data path')

    parser.add_argument('--initial-trigger', nargs='+', type=str, default=None, help='Manual prompt')
    parser.add_argument('--label-field', type=str, default='label',
                        help='Name of the label field')

    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--eval-size', type=int, default=256, help='Eval size')
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--accumulation-steps', type=int, default=10)
    parser.add_argument('--model-name', type=str, default='bert-base-cased',
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
    logging.basicConfig(filename=args.logfile,level=level)

    run_model(args)