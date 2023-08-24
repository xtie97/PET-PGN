import os

# prepare dataset using pretrained word embeddings 
os.system('python prepare_vocab.py dataset/PET-CT dataset/vocab --glove_dir dataset/glove --wv_file radglove.800M.100d.txt --wv_dim 100 --lower')

# training codes 
os.system('python train.py --id 01 --data_dir dataset/PET-CT --max_dec_len 512 --background --num_epoch 30 --batch_size 25') # --max_dec_len: maximal decoder length
# with coverage loss to avoid repetition
#os.system('python train.py --id 02 --data_dir dataset/PET-CT --max_dec_len 512 --background --cov --cov_loss_epoch 15 --cov_alpha 1.0 --num_epoch 30') 
# coverage loss lambda=1, start from epoch 15

# evaluation codes (predict PET impression)
saved_model = 'saved_models/01'
saved_prediction = 'saved_predictions/01'
os.makedirs(saved_prediction, exist_ok=True)
os.system(('python eval.py  {} --model best_model.pt --data_dir dataset/PET-CT --dataset test --gold {}/test_ref.txt --out {}/test_pred.txt').format(saved_model, 
                                                                                                                                                     saved_prediction, 
                                                                                                                                                     saved_prediction)) 



