# Pointer Generator Network (PGN) for PET report summarization 

This is the implementation for [paper]. In this work, we aim to derive impressions from PET findings and relevant background information. PGN is a bidrectional-LSTM with copy mechanism to solve the out-of-vocabulary problem. The architecture was modified by Zhang et. al. in "Learning to Summarize Radiology Findings" [https://arxiv.org/abs/1809.04698] to accomodate clnical information, including the patient history and the indication for the examination. 

Structure: 
convert_json.py: convert spreadsheet to json file 
prepare_vocab: prepare our corpus by mapping individual words to the embeddings. 
train.py: code for model training 
eval.py: code for impression generation
compute_rough_score.py: compute ROUGE-1, ROUGE-2, ROUGE-3 and ROUGE-L between the generated impression and the original clinial impression. 

Usage:
To prepare the dataset and vocabulary: 
python prepare_vocab.py dataset/PET-CT dataset/vocab --glove_dir dataset/glove --wv_file radglove.800M.100d.txt --wv_dim 100 --lower

To run training:
python train.py --id 01 --data_dir dataset/PET-CT --max_dec_len 512 --background --num_epoch 30 --batch_size 25

To run inference:
os.makedirs(saved_prediction, exist_ok=True)
python eval.py  saved_models/01 --model best_model.pt --data_dir dataset/PET-CT --dataset test --gold saved_predictions/01/test_ref.txt --out saved_predictions/01/test_pred.txt


Acknowledgments:
The codes were adapted from the original implementation [https://github.com/yuhaozhang/summarize-radiology-findings].
