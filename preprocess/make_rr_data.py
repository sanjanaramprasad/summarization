import pandas as pd 
import json
import re, string
import os 


def read_json(filename):
    with open(filename, 'r') as fp:
        all_lines = json.load(fp)
    return all_lines

def _make_review_pmid_map(input_data):
    review_pmid_map = {}
    print(input_data.keys())
    for k , rid in input_data['ReviewID'].items():
        pmids = input_data['PMID'][k]
        review_pmid_map[rid] = pmids
    return review_pmid_map



def _find_pmid(abstracts, ptext):
    for i,abst in enumerate(abstracts):
            if ptext in abst:
                return i
    return None
  
def preprocess_str(text):
    import re
    puncts = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~↑↓―'
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    text = regex.sub(' ', text)
    text = [w.lower().strip(puncts) for w in text.split()if w.strip()]
    
    return ' '.join(text)

def preprocess(row_vals):
    new_row_vals = []
    for each_val in row_vals:
        each_val = each_val if type(each_val) is list else [str(each_val)]
        #print(each_val)
        each_val = [preprocess_str(each) for each in each_val]
        each_val = ' <sep> '.join(each_val)
        new_row_vals.append(each_val)
    return new_row_vals

def make_review_data(review_pmid_map, text_inputs, rr_input, text_target):
    review_data = {'PMID' : [], 'Abstract' : [], 'ReviewID': [], 'SummaryConclusions': []}
    for rid, pmids in review_pmid_map.items():
        pmids = [int(each) for each in pmids]
        df_sample = text_inputs[text_inputs['PMID'].isin(pmids)]
        df_sample = df_sample[df_sample['Abstract'].notna()]
        df_pmids = list(df_sample['PMID'])
        df_abstract = list(df_sample['Abstract'])
        target = text_target[text_target['ReviewID'] == rid]
        target = list(target['Target'])[0]
        #target = preprocess_str(target)
        pmid_data = []
        abstract_data = []
        pico_data = {}
        for pico_ele in rr_input[rid]:
            punchline_text = pico_ele['punchline_text']
            pmid_ind = _find_pmid(df_abstract, punchline_text)
            if pmid_ind:
                pmid = df_pmids[pmid_ind]
                abstract = df_abstract[pmid_ind]
                pmid_data.append(pmid)
                abstract_data.append(abstract)
                
                for k , v in pico_ele.items():
                    if k not in pico_data:
                        pico_data[k] = []
                    pico_data[k].append(v)
        if pmid_data:
            if not df_abstract:
                print(rid)
            pmid_data = preprocess(pmid_data)
            abstract_data = preprocess(abstract_data)
            
            #review_data['PMID'].append(pmid_data)
            #review_data['Abstract'].append(abstract_data)
            for k , v in pico_data.items():
                if 'mesh' not in k:
                    v = preprocess(v)
                    pico_data[k] = v
                    if k not in review_data:
                        review_data[k] = []
                    #v = ["<%s> "%k + each + " </%s>"%k for each in v]
                    #v = " ".join(v)
                    review_data[k].append(v)
            review_data['PMID'].append(pmid_data)
            review_data['Abstract'].append(abstract_data)
            review_data['ReviewID'].append(rid)
            review_data['SummaryConclusions'].append(preprocess_str(target))
                
    return review_data


def _sanity_check(df):
    for ind, row in df.iterrows():
        pmids = list(row['PMID'])
        pop = list(row['population'])
        Abstract = list(row['Abstract'])
        
        assert( len(pmids) == len(pop) ==len(Abstract))       
        

if __name__ == '__main__':
    main_path = '/Users/sanjana/summarization/datasets/'

    text_train_inputs = pd.read_csv(main_path + 'robotreviewer/train-inputs.csv')
    text_train_targets = pd.read_csv(main_path + 'robotreviewer/train-targets.csv')

    text_dev_inputs = pd.read_csv(main_path + 'robotreviewer/dev-inputs.csv')
    text_dev_targets = pd.read_csv(main_path + 'robotreviewer/dev-targets.csv')

    text_test_inputs = pd.read_csv(main_path + 'robotreviewer/test-inputs.csv')
    text_test_targets = pd.read_csv(main_path + 'robotreviewer/test-targets.csv')

    rr_train_input = read_json(main_path + 'robotreviewer/RR-train.json')
    rr_train_output = read_json(main_path + 'robotreviewer/abstracts-summarization-train.json')

    rr_dev_input = read_json(main_path + 'robotreviewer/RR-dev.json')
    rr_dev_output = read_json(main_path + 'robotreviewer/abstracts-summarization-dev.json')

    rr_test_input = read_json(main_path + 'robotreviewer/RR-test.json')
    rr_test_output = read_json(main_path + 'robotreviewer/abstracts-summarization-test.json')

    rr_train_review_pmid_map = _make_review_pmid_map(rr_train_output)
    rr_dev_review_pmid_map = _make_review_pmid_map(rr_dev_output)
    rr_test_review_pmid_map = _make_review_pmid_map(rr_test_output)

    review_data_train = make_review_data(rr_train_review_pmid_map, text_train_inputs, rr_train_input, text_train_targets)
    review_data_dev = make_review_data(rr_dev_review_pmid_map, text_dev_inputs, rr_dev_input, text_dev_targets)
    review_data_test = make_review_data(rr_test_review_pmid_map, text_test_inputs, rr_test_input, text_test_targets)

    out_file_train = main_path + 'train_rr_data.csv'
    out_file_dev = main_path + 'dev_rr_data.csv'
    out_file_test = main_path + 'test_rr_data.csv'

    review_data_train = pd.DataFrame(review_data_train)
    _sanity_check(review_data_train)
    review_data_train.to_csv(out_file_train)

    review_data_dev = pd.DataFrame(review_data_dev)
    _sanity_check(review_data_dev)
    review_data_dev.to_csv(out_file_dev)

    review_data_test = pd.DataFrame(review_data_test)
    _sanity_check(review_data_test)
    review_data_test.to_csv(out_file_test)