from DataToTextProcessor_encoder import SummaryDataModule
#from Data2TextProcessor_1 import SummaryDataModule
from transformers import BartTokenizer
import torch.optim as optim



from torch import nn 
import torch



def make_data(tokenizer, SummaryDataModule,  data_type = 'robo', path = '/home/ramprasad.sa', files = ['robo_train_sep.csv', 'robo_dev_sep.csv', 'robo_test_sep.csv']):
    if data_type == 'robo':
        train_file = path + '/summarization/datasets/%s'%(files[0])
        dev_file = path + '/summarization/datasets/%s'%(files[1])
        test_file = path + '/summarization/datasets/%s'%(files[2])

    data_files = [train_file, dev_file, test_file]
    summary_data = SummaryDataModule(tokenizer, data_files = data_files,  batch_size = 1, max_len = 256)
    summary_data.prepare_data()
    return summary_data


def get_data(data):
        input_ids_col0 = data[0] if len(data) >1 else None
        attention_mask_col0 = data[1] if len(data) >1 else None

        input_ids_col1 = data[2] if len(data) >3 else None
        attention_mask_col1 = data[3] if len(data) >3 else None

        input_ids_col2 = data[4] if len(data) >5 else None
        attention_mask_col2 = data[5] if len(data) >5 else None

        input_ids_col3 = data[6] if len(data) >7 else None
        attention_mask_col3 = data[7] if len(data) >7 else None

        input_ids_col4 = data[8] if len(data) >9 else None
        attention_mask_col4 = data[9] if len(data) >9 else None

        return input_ids_col0, attention_mask_col0, \
            input_ids_col1, attention_mask_col1, \
            input_ids_col2, attention_mask_col2, \
            input_ids_col3, attention_mask_col3, \
            input_ids_col4, attention_mask_col4

additional_special_tokens = ["<sep>",
            "<outcomes>", "</outcomes>",
            "<punchline_text>", "</punchline_text>",
            "<population>", "</population>",
            "<interventions>", "</interventions>"]
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', bos_token="<s>",
                                                    eos_token="</s>",
                                                    pad_token = "<pad>")




class BartForDataToTextGenerationTester():



    def test_model_forward_bart_encoder(self, encoder_combination_type):
        from BartForDataToText_EncoderMod import BartForDataToText
        #from DataToTextProcessor_encoder import SummaryDataModule
        model = BartForDataToText.from_pretrained('facebook/bart-base')
        model._make_duplicate_encoders(layer_share = False)
        model.resize_token_embeddings(len(tokenizer))
        print("Loading Data ...")
        summary_data = make_data(tokenizer, SummaryDataModule, path = '/home/ramprasad.sa', files = ['train_rr_data.csv', 
                            'dev_rr_data.csv', 'test_rr_data.csv'])
        summary_data.setup("stage")
        test_data = summary_data.test_dataloader(data_type = 'robo')
        print("Done.")
        it = iter(test_data)
        
        data = next(it)
        input_ids_col0, attention_mask_col0, input_ids_col1, attention_mask_col1, \
            input_ids_col2, attention_mask_col2, \
            input_ids_col3, attention_mask_col3, \
            input_ids_col4, attention_mask_col4 = get_data(data)

        

        print("forward...") 
        outputs = model(
            input_ids_col0 = input_ids_col0,
            input_ids_col1 = input_ids_col1,
            input_ids_col2 = input_ids_col2, 
            input_ids_col3 = input_ids_col3,
            input_ids_col4 = input_ids_col4,
            attention_mask_col0 = attention_mask_col0,
            attention_mask_col1 = attention_mask_col1,
            attention_mask_col2 = attention_mask_col2,
            attention_mask_col3 = attention_mask_col3,
            attention_mask_col4 = attention_mask_col4,
            labels = data[-1],
            encoder_combination_type = encoder_combination_type,
            use_cache = False
        )
        tgt_ids = data[-1]
        optimizer = optim.Adam(model.parameters())
        loss = outputs[0]
        #ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        #loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        optimizer.zero_grad()
        loss.backward()
        print("OUTPUTS", outputs[0])
        print('=' *13)


    


 
        
obj = BartForDataToTextGenerationTester()
obj.test_model_forward_bart_encoder(encoder_combination_type = 'recursive_cross_attn')
#obj.test_model_forward_bart_encoder_loop_per_study(encoder_combination_type = 'linearize')
