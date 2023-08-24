import os 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install rouge-score==0.1.2") 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U sentence-transformers")

import pandas as pd
from tqdm import tqdm 
from rouge_score import rouge_scorer
import numpy as np 
import nltk 
nltk.download('punkt')

def predict_text(df):
    #Import the pretrained model
    BART_impression = df['AI_impression'].tolist()
    #impression = df['impression_ref'].tolist() 
    impression = df['impressions'].tolist() 
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL', 'rougeLsum'], use_stemmer=True, split_summaries=True)
    results_rouge1 = {'precision': [], 'recall': [], 'f': []} 
    results_rouge2 = {'precision': [], 'recall': [], 'f': []} 
    results_rouge3 = {'precision': [], 'recall': [], 'f': []} 
    results_rouge4 = {'precision': [], 'recall': [], 'f': []} 
    results_rougeL = {'precision': [], 'recall': [], 'f': []} 

    for i in tqdm(np.arange(len(impression))):  
        gen_text = BART_impression[i]
        gt_text = impression[i]
        scores = scorer.score(gen_text, gt_text)
        results_rouge1['precision'].append(list(scores['rouge1'])[0]) 
        results_rouge1['recall'].append(list(scores['rouge1'])[1]) 
        results_rouge1['f'].append(list(scores['rouge1'])[2]) 

        results_rouge2['precision'].append(list(scores['rouge2'])[0]) 
        results_rouge2['recall'].append(list(scores['rouge2'])[1]) 
        results_rouge2['f'].append(list(scores['rouge2'])[2]) 

        results_rouge3['precision'].append(list(scores['rouge3'])[0]) 
        results_rouge3['recall'].append(list(scores['rouge3'])[1]) 
        results_rouge3['f'].append(list(scores['rouge3'])[2]) 

        results_rouge4['precision'].append(list(scores['rouge4'])[0]) 
        results_rouge4['recall'].append(list(scores['rouge4'])[1]) 
        results_rouge4['f'].append(list(scores['rouge4'])[2]) 

        results_rougeL['precision'].append(list(scores['rougeL'])[0]) 
        results_rougeL['recall'].append(list(scores['rougeL'])[1]) 
        results_rougeL['f'].append(list(scores['rougeL'])[2]) 

    index = np.flip(np.argsort(results_rouge2['f']))
    index = index[:10]
    #print(index+1+1)
   
    results_rouge1['precision'] = np.around(np.mean(results_rouge1['precision']), 3)
    results_rouge1['recall'] = np.around(np.mean(results_rouge1['recall']), 3)
    results_rouge1['f'] = np.around(np.mean(results_rouge1['f']), 3)

    results_rouge2['precision'] = np.around(np.mean(results_rouge2['precision']), 3)
    results_rouge2['recall'] = np.around(np.mean(results_rouge2['recall']), 3)
    results_rouge2['f'] = np.around(np.mean(results_rouge2['f']), 3)

    results_rouge3['precision'] = np.around(np.mean(results_rouge3['precision']), 3)
    results_rouge3['recall'] = np.around(np.mean(results_rouge3['recall']), 3)
    results_rouge3['f'] = np.around(np.mean(results_rouge3['f']), 3)

    results_rouge4['precision'] = np.around(np.mean(results_rouge4['precision']), 3)
    results_rouge4['recall'] = np.around(np.mean(results_rouge4['recall']), 3)
    results_rouge4['f'] = np.around(np.mean(results_rouge4['f']), 3)

    results_rougeL['precision'] = np.around(np.mean(results_rougeL['precision']), 3)
    results_rougeL['recall'] = np.around(np.mean(results_rougeL['recall']), 3)
    results_rougeL['f'] = np.around(np.mean(results_rougeL['f']), 3)

    return results_rouge1['f'], results_rouge2['f'] , results_rouge3['f'], results_rouge4['f'], results_rougeL['f']


if __name__ == '__main__':
    # Testing 
    # current 27 is the best among all 30 models 
    df = pd.read_excel('saved_predictions/01/test_results.xlsx')
    
    r1, r2, r3, r4, rl = predict_text(df)
    print('rouge1: ', r1)
    print('rouge2: ', r2)
    print('rouge3: ', r3)
    print('rouge4: ', r4)
    print('rougeL: ', rl)
    