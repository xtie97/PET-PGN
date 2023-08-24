"""
Evaluating saved models.
"""

import argparse
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable

from data.loader import DataLoader
from utils import helper, constant, torch_utils, text_utils, bleu, rouge
from utils.torch_utils import set_cuda
from utils.vocab import Vocab
from model.trainer import Trainer
from rouge_score import rouge_scorer
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='Directory of the model file.')
parser.add_argument('--data_dir', default='', help='Directory to look for data. By default use the path in loaded args.')
parser.add_argument('--model', default='best_model.pt', help='Name of the model file.')
parser.add_argument('--dataset', default='test', help="Data split to use for evaluation: dev or test.")
parser.add_argument('--batch_size', type=int, default=20, help="Batch size for evaluation.")
parser.add_argument('--gold', default='', help="Optional: a file where to write gold summarizations. Default to not write.")
parser.add_argument('--out', default='', help="Optional: a file where to write predictions. Default to not write.")
parser.add_argument('--use_bleu', action='store_true', help="Use BLEU instead of ROUGE metrics for scoring.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
trainer = Trainer(model_file=model_file)
opt, vocab = trainer.opt, trainer.vocab

# load data
data_dir = args.data_dir if len(args.data_dir) > 0 else opt['data_dir']
data_file = data_dir + '/{}.jsonl'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, args.batch_size))
batch = DataLoader(data_file, args.batch_size, opt, vocab, evaluation=True)
test_gold = batch.save_gold(args.gold)

helper.print_config(opt)

print("Evaluating on the {} set...".format(args.dataset))
predictions = []
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

for b in tqdm(batch):
    preds = trainer.predict(b)
    predictions += preds
predictions = text_utils.postprocess(predictions)
if args.use_bleu:
    test_bleu = bleu.get_bleu(predictions, test_gold)
    print("{} set bleu score: {:.2f}".format(args.dataset, test_bleu))

rouge1_list = []
rouge2_list = []
rougel_list = []
pred_list = [] 
ref_list = []
for ii in range(len(predictions)):
    gen = ' '.join(predictions[ii])
    ref = ' '.join(test_gold[ii])
    if gen.startswith(']'):
        gen = '[1' + gen
    if ref.startswith(']'):
        ref = '[1' + ref
    gen = gen.replace('[ ', '[').replace(' ]', ']')
    ref = ref.replace('[ ', '[').replace(' ]', ']')
    scores = rouge.score(gen, ref)
    rouge1 = list(scores['rouge1'])[2]
    rouge2 = list(scores['rouge2'])[2]
    rougel = list(scores['rougeL'])[2]
    pred_list.append(gen)
    ref_list.append(ref)
    rouge1_list.append(rouge1)
    rouge2_list.append(rouge2)
    rougel_list.append(rougel)

# save to excel file
test_file = data_dir + '/{}.xlsx'.format(args.dataset)
df = pd.read_excel(test_file)
df['AI_impression'] = pred_list
df['impression_ref'] = ref_list
df['rouge1'] = rouge1_list
df['rouge2'] = rouge2_list
df['rougeL'] = rougel_list
print('Average Rouge 1: {:.2f}'.format(np.mean(rouge1_list)))
print('Average Rouge 2: {:.2f}'.format(np.mean(rouge2_list)))
print('Average Rouge L: {:.2f}'.format(np.mean(rougel_list)))
df.to_excel(args.model_dir + '/{}_results_{}.xlsx'.format(args.dataset, args.model.split('.')[0]), index=False)
