import pandas as pd 
import nltk
nltk.download('punkt') # download punkt package
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
import json
 
# convert excel to json file
def excel_to_json(df, json_path):
    # background, findings, impression
    df['merged_information'] = df['merged_information'].apply(lambda x: x.replace('\n',' '))
    df['findings'] = df['findings'].apply(lambda x: x.replace('\n',' '))
    df['impressions'] = df['impressions'].apply(lambda x: x.replace('\n',' '))
    
    accession = df['Accession Number'].tolist()
    findings = df['findings'].tolist()
    impressions = df['impressions'].tolist()
    backgrounds = df['merged_information'].tolist()

    for ii in tqdm(range(len(findings))):
        accession_id = accession[ii]
        findings_words = word_tokenize(findings[ii])
        impressions_words = word_tokenize(impressions[ii])
        backgrounds_words = word_tokenize(backgrounds[ii])
        
        # Data to be written
        dictionary = {"accession": accession_id,
                      "background": backgrounds_words,
                      "findings": findings_words,
                      "impression": impressions_words,
                      }
        if ii == 0:
            with open(json_path, "w") as outfile:
                json.dump(dictionary, outfile)
                outfile.write('\n')
        else:
            with open(json_path, "a") as outfile:
                json.dump(dictionary, outfile)
                outfile.write('\n')

def filter_test_data(df):
    # select the recent 5 years data
    exam_date = df['Exam Date Time']
    exam_date_new = []
    for ii in tqdm(range(len(exam_date))):
        if '2018' in exam_date[ii] or '2019' in exam_date[ii] or '2020' in exam_date[ii] \
        or '2021' in exam_date[ii] or '2022' in exam_date[ii] or '2023' in exam_date[ii]:
            exam_date_new.append(exam_date[ii])
        else:
            exam_date_new.append('Remove')
    df['Exam Date Time'] = exam_date_new
    # drop the data with no exam date
    df = df[df['Exam Date Time'] != 'Remove'].reset_index(drop=True)
    df = df[df['Study Description'] == 'PET CT WHOLE BODY'].reset_index(drop=True)
    return df 

if __name__ == '__main__':
    #Get data
    df_train = pd.read_excel('./archive/train.xlsx')
    df_test = pd.read_excel('./archive/test.xlsx')
    save_path = './dataset/PET-CT/'
    df_val = df_train.sample(n=2000, random_state=42)
    df_train = df_train.drop(df_val.index)

    df_train.to_excel(save_path + 'train.xlsx', index=False)
    excel_to_json(df_train, json_path=save_path + 'train.jsonl')

    df_val.to_excel(save_path + 'dev.xlsx', index=False)
    excel_to_json(df_val, json_path=save_path + 'dev.jsonl')
    
    df_test.to_excel(save_path + 'test.xlsx', index=False)
    excel_to_json(df_test, json_path=save_path + 'test.jsonl')

    df_test_filter = filter_test_data(df_test) # recent 5 years + whole body PET/CT (dominant report type)
    df_test_filter.to_excel(save_path + 'test_filter.xlsx', index=False)
    excel_to_json(df_test_filter, json_path=save_path + 'test_filter.jsonl')