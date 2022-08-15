export CUDA_VISIBLE_DEVICES="1"

python3 -m autoprompt.create_trigger_iter_optim \
--train /home/zsarwar/NLP/Sorting-Through-The-Noise/data/Varying_key_entity/Correctly_classified_roberta-large_auto_small.jsonl \
--dev /home/zsarwar/NLP/Sorting-Through-The-Noise/data/Varying_key_entity/Correctly_classified_roberta-large_auto_small.jsonl \
--template '<s> {Text} [P] . [T] [T] [T] [T] [T] [T] </s>' \
--num-cand 10 \
--accumulation-steps 1 \
--model-name roberta-large \
--bsz 1 \
--eval-size 1 \
--iters 1  \
--initial-trigger "hello" \
--label-field 'Label' \
--filter \
--print-lama > test.txt