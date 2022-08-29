export CUDA_VISIBLE_DEVICES="2"

python3 -m autoprompt.create_trigger_eval \
--train /home/zsarwar/NLP/Sorting-Through-The-Noise/data/Varying_key_entity/Correctly_classified_roberta-large_auto_small_split.jsonl \
--dev /home/zsarwar/NLP/Sorting-Through-The-Noise/data/Varying_key_entity/Correctly_classified_roberta-large_auto_small_split.jsonl \ 
--template '<s> [T] [T] [T] [T] [T] [T] . {Background_Text} . {Pred_Text} [P] . </s>' \
--num-cand 10 \
--accumulation-steps 1 \
--model-name roberta-large \
--bsz 50 \
--eval-size 1 \
--iters 1  \
--initial-trigger "in the middle of the night i go walking in my sleep through the valley of fear to the river so deep i must be looking for something taken out of my soul something i'd never loose" \
--label-field 'Label' \
--filter \
--logfile "logs_universal_eval_t.b.p.txt" \
--print-lama > test.txt


#--train /home/zsarwar/NLP/Sorting-Through-The-Noise/data/Varying_key_entity/Correctly_classified_roberta-large_auto_small_split.jsonl \
#--dev /home/zsarwar/NLP/Sorting-Through-The-Noise/data/Varying_key_entity/Correctly_classified_roberta-large_auto_small_split.jsonl \
#--train /home/zsarwar/NLP/Sorting-Through-The-Noise/data/Varying_key_entity/Correctly_classified_roberta-large_auto_split_1k.jsonl \
#--dev /home/zsarwar/NLP/Sorting-Through-The-Noise/data/Varying_key_entity/Correctly_classified_roberta-large_auto_split_1k.jsonl \


#--train /home/zsarwar/NLP/Sorting-Through-The-Noise/data/Varying_key_entity/Correctly_classified_roberta-large_auto_small_split.jsonl \
#--dev /home/zsarwar/NLP/Sorting-Through-The-Noise/data/Varying_key_entity/Correctly_classified_roberta-large_auto_small_split.jsonl \ 


#--train /home/zsarwar/NLP/Sorting-Through-The-Noise/data/Varying_key_entity/Correctly_classified_roberta-large_auto_split_1k.jsonl \
#--dev /home/zsarwar/NLP/Sorting-Through-The-Noise/data/Varying_key_entity/Correctly_classified_roberta-large_auto_split_1k.jsonl \


#--template '<s> {Background_Text} . {Pred_Text} [P] . [T] [T] [T] [T] [T] [T] . </s>' \
#--template '<s> {Background_Text} . [T] [T] [T] [T] [T] [T] . {Pred_Text} [P] . </s>' \
#--template '<s> [T] [T] [T] [T] [T] [T] . {Background_Text} . {Pred_Text} [P] . </s>' \

#--template '<s> {Background_Text} . {Pred_Text} [P] [T] [T] [T] [T] [T] [T] </s>' \
#--template '<s> {Background_Text} . [T] [T] [T] [T] [T] [T] {Pred_Text} [P] </s>' \
#--template '<s> [T] [T] [T] [T] [T] [T] {Background_Text} . {Pred_Text} [P] </s>' \


#--logfile "logs_new_T.B.P.txt" \
#--logfile "logs_new_B.T.P.txt" \
#--logfile "logs_new_B.P.T.txt" \