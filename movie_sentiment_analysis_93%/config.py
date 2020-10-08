import transformers
import torch 


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "/home/hasan/Desktop/bert-sentiment/model.bin"
TRAINING_FILE = "/home/hasan/Data Set/movie review/IMDB Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)




