import tkinter as tk
import torch

from transformers import BertTokenizer, BertModel
import pandas as pd
from Model import BERT_CNN, BERT_BiLSTM, BERTClass
from GUI import SongGenreClassifier


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BERTClass()
    model.to(device)
    model.load_state_dict(torch.load(r"E:\Data_analysis\DS\DS-project-20221\Source_Code\GUI\model_state_dict\BERT_Class.bin", map_location=device))

    root = tk.Tk()
    app = SongGenreClassifier(root, tokenizer, model, device)
    root.mainloop()
