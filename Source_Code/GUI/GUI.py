import pandas as pd
import torch
import tkinter as tk
import numpy as np
from sys import exit
from tkinter import messagebox
from transformers import BertTokenizer, BertModel
from Text_Preprocess import TextPreprocessor
from Model import BERT_CNN, BERT_BiLSTM, BERTClass


data = pd.read_csv(r'E:\Data_analysis\DS\DS-project-20221\Source_Code\GUI\data_final_preprocess.csv')
data = data.drop(columns=['Unnamed: 0'])
print(data.columns)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SongGenreClassifier:
    def __init__(self, master, tokenizer, model_default, device=DEVICE):
        self.master = master
        master.title("Song Genre Classifier")
        master.geometry("650x820")
        master.config(bg="#fefbe9")

        # Create a label for the textbox
        self.choose = tk.Label(master, text="Choose your model:", fg="#321313", bg="#fefbe9", font=("Helvetica", 18))
        self.choose.pack(pady=(30,0))
        # self.choose.place(x=100, y =70)

        # Create buttons to select the model
        self.bertclass_button = tk.Button(
            master, 
            text=" BERT_Linear ", 
            command=lambda: self.get_model_1(), 
            font=("Helvetica", 18), 
            bg="#e2c275", 
            activebackground="#eadca6", 
            fg="#fff"
        )

        self.bertclass_button.pack(pady=(0,10))
        # self.bertclass_button.place(x=400, y=20)

        self.bert_cnn_button = tk.Button(
            master, 
            text="  BERT_CNN  ", 
            command=lambda: self.get_model_2(), 
            font=("Helvetica", 18), 
            bg="#e2c275", 
            activebackground="#eadca6", 
            fg="#fff"
        )

        self.bert_cnn_button.pack(pady=(0,10))
        # self.bert_cnn_button.place(x=400, y=70)

        self.bert_bilstm_button = tk.Button(
            master, 
            text="BERT_BiLSTM", 
            command=lambda: self.get_model_3(), 
            font=("Helvetica", 18), 
            bg="#e2c275", 
            activebackground="#eadca6", 
            fg="#fff"
        )

        self.bert_bilstm_button.pack(pady=(0,10))
        # self.bert_bilstm_button.place(x=400, y=120)

        self.model_label = tk.Label(master, text="Default BERT_CNN", fg="#321313", bg="#fefbe9", font=("Helvetica", 24))
        self.model_label.pack(pady=(30,0))

        # Create a label for the textbox
        self.label = tk.Label(master, text="Enter song lyrics:", fg="#321313", bg="#fefbe9", font=("Helvetica", 18))
        self.label.pack(pady=(30,0))
        # self.label.place(x=100, y=70)

        # Create a text box for the user to input lyrics
        self.textbox = tk.Text(master, height=10, width=40, font=("Helvetica", 14))
        self.textbox.pack(pady=(0,20))

        # Create a button to submit the lyrics and get the genre prediction
        self.button = tk.Button(
            master, 
            text="Predict Genre", 
            command=self.predict_genre, 
            font=("Helvetica", 18), 
            bg="#e2c275", 
            activebackground="#eadca6", 
            fg="#fff"
        )

        self.button.pack()
        
        # Create a label to display the predicted genre
        self.genre_label = tk.Label(master, text="Predicted genre", fg="#321313", bg="#fefbe9", font=("Helvetica", 24))
        self.genre_label.pack(pady=(30,0))

        # Create a label to display the confidence 
        self.confidence = tk.Label(master, text="Confidence", fg="#321313", bg="#fefbe9", font=("Helvetica", 24))
        self.confidence.pack(pady=(30,0))

        # Load the deep learning model and tokenizer
        self.tokenizer = tokenizer
        self.model = model_default
        self.device = device


    def get_model_2(self):
        print('BERT_CNN is choosed.')
        bert_cnn = BERT_CNN(num_classes=6)
        bert_cnn.to(self.device)
        bert_cnn.load_state_dict(
            torch.load(
                r"E:\Data_analysis\DS\DS-project-20221\Source_Code\GUI\model_state_dict\BERT_CNN.bin", 
                map_location=self.device
            )
        )
        self.model_label.configure(text="You choose model BERT_CNN.", bg="#fefbe9")
        self.model = bert_cnn
    

    def get_model_1(self):
        print('BERTClass is choosed.')
        bert = BERTClass()
        bert.to(self.device)
        bert.load_state_dict(
            torch.load(
                r"E:\Data_analysis\DS\DS-project-20221\Source_Code\GUI\model_state_dict\BERT_Class.bin", 
                map_location=self.device
            )
        )
        self.model_label.configure(text="You choose model BERT_Linear.", bg="#fefbe9")
        self.model = bert

    
    def get_model_3(self):
        print('BERT_LSTM is choosed.')
        bert_lstm = BERT_BiLSTM()
        bert_lstm.to(self.device)
        bert_lstm.load_state_dict(
            torch.load(
                r"E:\Data_analysis\DS\DS-project-20221\Source_Code\GUI\model_state_dict\BERT_LSTM.bin",
                map_location=self.device
            )
        )
        self.model_label.configure(text="You choose model BERT_LSTM.", bg="#fefbe9")
        self.model = bert_lstm


    def popupError(self):
        self.genre_label.configure(text=f"Predicted genre", bg="#fefbe9")
        self.confidence.configure(text=f"Confidence", bg="#fefbe9")
        messagebox.showwarning("Empty box error!", "You haven't input your lyric!!!")


    def predict_genre(self):
        # Get the text from the textbox
        lyrics = self.textbox.get("1.0", "end-1c")
        
        if lyrics == '':
            self.popupError()
        else:
            preprocess_lyric = TextPreprocessor(lyrics).preprocess_text()

            # Convert the lyrics to an input encoding for the model
            encodings = self.tokenizer.encode_plus(
                preprocess_lyric,
                None,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                return_token_type_ids=True,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            # Make a prediction using the model
            self.model.eval()
            with torch.no_grad():
                input_ids = encodings['input_ids'].to(self.device, dtype=torch.long)
                attention_mask = encodings['attention_mask'].to(self.device, dtype=torch.long)
                token_type_ids = encodings['token_type_ids'].to(self.device, dtype=torch.long)
                output = self.model(input_ids, attention_mask, token_type_ids)
                final_output = torch.sigmoid(output).cpu().detach().numpy().tolist()
                cp_output = output
                cp_output = torch.softmax(cp_output, dim=1).cpu().detach().numpy()
                truth = cp_output.tolist()
                confident = max(truth[0]) * 100
                predicted_genre = data.columns[:6].to_list()[int(np.argmax(final_output, axis=1))]

        # Set the genre label text to the predicted genre
        self.genre_label.configure(text=f"Predicted genre: {predicted_genre}", bg="#fefbe9")
        self.confidence.configure(text=f"Confidence: {int(confident)} %", bg="#fefbe9")

        # Clear the textbox
        self.textbox.delete("1.0", "end")
