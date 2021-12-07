# Investigating the Impact of Multi-Task Learning strategies on Selective Prediction

Clone the repo
```
git clone https://github.com/Arutselvan/selective_prediction_mtl
```

Change the current directory to the cloned repository
```
cd selective_prediction_mtl
```

The datasets for both train and eval are present in the `dataset` folder.

To convert the dataset into QA format [Question, Context, Answer]:
```
python code/squad_converter.py
```
Both training files, the combined dataset (`main.json`) and evaluation samples will be generated and stored in the `dataset` folder.

To perform training (Heterogeneous Sampling) and evaluation on all datasets' eval files, run the below command (from the root directory of the repo)
```
python code/run_qa.py --model_name_or_path bert-base-cased --do_train --train_file main.json --validation_files "['snli_squad_eval.json', 'swag_squad_eval.json', 'csqa_squad_eval.json', 'anli_squad_eval.json', 'siqa_squad_eval.json']"  --max_seq_length 256 --output_dir ./output-hetero --overwrite_output_dir --num_train_epochs 5 --evaluation_strategy epoch --per_device_train_batch_size 16 --per_device_eval_batch_size 32
```
To perform training (Homogenous Sampling) and evaluation on all datasets' eval files, run the below command (from the root directory of the repo)
```
python code/run_qa.py --model_name_or_path bert-base-cased --do_train --train_file main.json --validation_files "['snli_squad_eval.json', 'swag_squad_eval.json', 'csqa_squad_eval.json', 'anli_squad_eval.json', 'siqa_squad_eval.json']"  --max_seq_length 256 --output_dir ./output-homo --overwrite_output_dir --num_train_epochs 5 --evaluation_strategy epoch --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --sampling Homogenous
```

Note: The evaluation accuracy metrics won't output anything/will output zero because the the predictions format was changed for the purpose of selective prediction.

The predictions will be of this format:
```
"csqa_0": {
    "prediction": "fail to work",
    "maxProb": "0.9998784"
 }
```
To create files with exact match and max probability for both homogenous and heterogenous models' predictions, run
```
python code/evaluate.py
```
This will create another `.json` file of format:
```
{
    "expected_prediction": "Levin was very successful at running the store",
    "prediction": "Levin was very successful at running the store",
    "correct": true,
    "maxProb": "0.99999547"
}
```
Note: For the above command to execute correctly, all steps needs to followed exactly as the folder paths are hardcoded.

To view plots of selective prediction metrics for all datasets (for both homogenous and heterogenous sampling), run
```
python code/evaluate_sp.py
```

This will create plot files comparing various selective prediction metrics of the models for all datasets. The plots are shown one by one for each dataset. (Close the current plot to view the next)
Note: The texts of the graph created might be cluttered on some devices. Make the graph full screen to view it clearly.

Example plot:

![snli_sp_metrics_mtl](https://user-images.githubusercontent.com/18646185/144694212-68eaf830-49d7-49d8-8336-fac60aa80627.png)
