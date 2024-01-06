export CUDA_VISIBLE_DEVICES='0'
python3 -m autoprompt.create_trigger \
    --train /home/zsarwar/NLP/autoprompt/data/fact-retrieval/original/P17/train.jsonl \
    --dev /home/zsarwar/NLP/autoprompt/data/fact-retrieval/original/P17/dev.jsonl \
    --template '<s> {sub_label} [T] [T] [T] [T] [T] [P] . </s>' \
    --num-cand 10 \
    --accumulation-steps 1 \
    --model-name roberta-large \
    --bsz 56 \
    --eval-size 56 \
    --iters 10 \
    --label-field 'obj_label' \
    --tokenize-labels \
    --filter \
    --print-lama