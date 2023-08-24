# Pointer Generator Network (PGN) for PET Report Summarization :bookmark_tabs:

Welcome to the **Pointer Generator Network (PGN) for PET report summarization** implementation. This model is included in our [paper](#link-to-paper) and sets a baseline for deriving impressions from PET findings and relevant background information.

## :mag_right: Overview

PGN is based on a **bidirectional-LSTM** armed with a copy mechanism to address the out-of-vocabulary challenges. The architecture has been refined by Zhang et al. in their work, ["Learning to Summarize Radiology Findings"](https://arxiv.org/abs/1809.04698), enhancing its capacity to assimilate clinical data—ranging from patient histories to examination indications.

## :file_folder: Repository Structure

- `convert_json.py`: Transforms spreadsheets into JSON files.
- `prepare_vocab.py`: Orchestrates our corpus by linking words with their embeddings.
- `train.py`: The central hub for model training.
- `eval.py`: Handles the generation of impressions.
- `compute_rough_score.py`: Computes ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-3, and ROUGE-L) juxtaposing generated impressions with their original clinical counterparts.

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

The codes were adapted from the original implementation [https://github.com/yuhaozhang/summarize-radiology-findings].

For any inquiries or feedback, feel free to raise an issue or contribute via pull requests!

