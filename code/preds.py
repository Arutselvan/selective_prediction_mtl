import os

preds_to_do = ['python code/run_qa.py --model_name_or_path ./output/ --do_predict --test_file "snli_squad_eval.json"  --max_seq_length 256 --output_dir ./predictions-snli-hetero',
                'python code/run_qa.py --model_name_or_path ./output/ --do_predict --test_file "swag_squad_eval.json"  --max_seq_length 256 --output_dir ./predictions-swag-hetero',
                'python code/run_qa.py --model_name_or_path ./output/ --do_predict --test_file "csqa_squad_eval.json"  --max_seq_length 256 --output_dir ./predictions-csqa-hetero',
                'python code/run_qa.py --model_name_or_path ./output/ --do_predict --test_file "anli_squad_eval.json"  --max_seq_length 256 --output_dir ./predictions-anli-hetero',
                'python code/run_qa.py --model_name_or_path ./output/ --do_predict --test_file "siqa_squad_eval.json"  --max_seq_length 256 --output_dir ./predictions-siqa-hetero']
for command in preds_to_do:
    os.system(command)