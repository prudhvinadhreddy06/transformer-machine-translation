import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from encoder import Encoder
from decoder import Decoder
from utils import create_padding_mask, create_look_ahead_mask
import numpy as np
from collections import Counter
from typing import List,Tuple,Dict
import re
import nltk

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        print(f"Initializing Transformer with src_vocab_size={src_vocab_size}, tgt_vocab_size={tgt_vocab_size}")
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        print(f"Final layer output size: {tgt_vocab_size}")
        

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.final_layer(dec_output)
        return output
    
def load_data(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def clean_text(text: str) -> str:
    # Remove special characters and digits
    clean = ""
    for txt in text:
        clean += re.sub(r'[^a-zA-ZÀ-Ÿ\s]', '', txt)
    # Convert to lowercase
    clean = clean.lower()
    return clean

def tokenize_text(text: str, language: str) -> List[str]:
    cleaned_text = clean_text(text)
    if language == 'french':
        return nltk.word_tokenize(cleaned_text, language='french')
    else:  # Default to English
        return nltk.word_tokenize(cleaned_text)

def build_vocab(texts: List[str], language: str,max_vocab_size: int = 10000) -> Tuple[Dict[str, int], Dict[int, str]]:
    text = tokenize_text(texts,language)
    word_freq = Counter()
    
    if language == 'english':
        for text in texts:
            word_freq.update(tokenize_text(text, 'english'))
    else :
        for text in texts:
            word_freq.update(tokenize_text(text, 'french'))

    
    vocab = ['<pad>', '<sos>', '<eos>', '<unk>']
    vocab.extend([word for word, _ in word_freq.most_common(max_vocab_size - 4)])
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return word2idx, idx2word

def preprocess_data(texts, word2idx, max_length):
    processed = []
    for text in texts:
        tokens = ['<sos>'] + text.split() + ['<eos>']
        ids = [word2idx.get(token, word2idx['<unk>']) for token in tokens]
        if len(ids) < max_length:
            ids += [word2idx['<pad>']] * (max_length - len(ids))
        else:
            ids = ids[:max_length-1] + [word2idx['<eos>']]
        processed.append(ids)
    return torch.LongTensor(processed)

def create_dataloaders(src_texts, tgt_texts, src_word2idx, tgt_word2idx, max_length, batch_size):
    src_data = preprocess_data(src_texts, src_word2idx, max_length)
    tgt_data = preprocess_data(tgt_texts, tgt_word2idx, max_length)
    dataset = TensorDataset(src_data, tgt_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, train_dataloader, val_dataloader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            #print(tgt_input)
            #print(tgt_output)
            src_mask = create_padding_mask(src).to(device)
            tgt_mask = create_look_ahead_mask(tgt_input.size(1)).to(device)
            
            optimizer.zero_grad()
            
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_dataloader)
        val_loss = evaluate_model(model, val_dataloader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    torch.save(model.state_dict(), 'transformer.pt')



if __name__ == '__main__':
    # Hyperparameters
    d_model = 512
    num_layers = 16 #increases
    num_heads = 8 
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1 
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.0001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    home = '/kaggle/input/'
    # Load data
    train_src = load_data(home+'en-fr-iwslt-2016/train.en')
    train_tgt = load_data(home+'en-fr-iwslt-2016/train.fr')
    val_src = load_data(home+'en-fr-iwslt-2016/dev.en')
    val_tgt = load_data(home+'en-fr-iwslt-2016/dev.fr')
    test_src = load_data(home+'en-fr-iwslt-2016/test.en')
    test_tgt = load_data(home+'en-fr-iwslt-2016/test.fr')

    # Build vocabularies
    src_word2idx, src_idx2word = build_vocab(train_src,'english')
    tgt_word2idx, tgt_idx2word = build_vocab(train_tgt + val_tgt + test_tgt,'french')
    
    
    # Create dataloaders
    train_dataloader = create_dataloaders(train_src, train_tgt, src_word2idx, tgt_word2idx, max_seq_length, batch_size)
    val_dataloader = create_dataloaders(val_src, val_tgt, src_word2idx, tgt_word2idx, max_seq_length, batch_size)
    test_dataloader = create_dataloaders(test_src, test_tgt, src_word2idx, tgt_word2idx, max_seq_length, batch_size)

    # Initialize model
    model = Transformer(
        len(src_word2idx), len(tgt_word2idx), d_model, num_layers, num_heads, d_ff, max_seq_length, dropout
    )

    # Train model
    train_model(model, train_dataloader, val_dataloader, num_epochs, learning_rate, device)