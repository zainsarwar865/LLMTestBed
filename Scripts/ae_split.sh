export CUDA_VISIBLE_DEVICES="2"

python3 -m autoprompt.create_trigger_iter_optim \
--train /home/zsarwar/NLP/Sorting-Through-The-Noise/data/Varying_key_entity/Correctly_classified_roberta-large_auto_small_split.jsonl \
--dev /home/zsarwar/NLP/Sorting-Through-The-Noise/data/Varying_key_entity/Correctly_classified_roberta-large_auto_small_split.jsonl \
--template '<s> {Background_Text} . [T] [T] [T] [T] [T] [T] [T] . {Pred_Text} [P] . </s>' \
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