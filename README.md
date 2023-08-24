# Pointer Generator Network (PGN) for PET Report Summarization :bookmark_tabs:

Welcome to the **Pointer Generator Network (PGN) for PET report summarization** implementation. This model is included in our [paper](#link-to-paper) and sets a baseline for deriving impressions based on PET findings and relevant background information.

## :mag_right: Overview

PGN is based on a **bidirectional-LSTM** with a copy mechanism to address the out-of-vocabulary (OOV) issue. The architecture has been modified by [Zhang et al.](https://arxiv.org/abs/1809.04698) in their work regarding radiology report summarization.

## :file_folder: Repository Structure

- `convert_json.py`: Converts spreadsheets into JSON files.
- `prepare_vocab.py`: Prepares our corpus by mapping individual words to their pretrained word embeddings.
- `train.py`: The script for model training.
- `eval.py`: The script for impression generation. 
- `compute_rough_score.py`: Computes ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-3, and ROUGE-L) of generated impressions with original clinical impressions as reference.
- `saved_models/01/best_model.pt`: model weights 

## :computer: Usage Instructions

**Prepare Dataset and Vocabulary**:
```bash
python prepare_vocab.py dataset/PET-CT dataset/vocab --glove_dir dataset/glove --wv_file radglove.800M.100d.txt --wv_dim 100 --lower
```

**Initial Training**:
```bash
python train.py --id 01 --data_dir dataset/PET-CT --max_dec_len 512 --background --num_epoch 30 --batch_size 25
```

**Inference / Impression Generation**:
python eval.py  saved_models/01 --model best_model.pt --data_dir dataset/PET-CT --dataset test --gold saved_predictions/01/test_ref.txt --out saved_predictions/01/test_pred.txt

## 👏 Acknowledgments

The code was adapted from the original [implementation in GitHub](https://github.com/yuhaozhang/summarize-radiology-findings).

For any inquiries or feedback, feel free to raise an issue or contribute via pull requests!

