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
from train import load_data,build_vocab,create_dataloaders,Transformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask = create_padding_mask(src).to(device)
            tgt_mask = create_look_ahead_mask(tgt_input.size(1)).to(device)
            
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def translate(model, src_sentence, src_word2idx, tgt_idx2word, max_length, device):
    model.eval()
    src_tokens = ['<sos>'] + src_sentence.split() + ['<eos>']
    src_ids = [src_word2idx.get(token, src_word2idx['<unk>']) for token in src_tokens]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
    
    src_mask = create_padding_mask(src_tensor).to(device)
    
    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_mask)
        
    tgt_tensor = torch.LongTensor([[src_word2idx['<sos>']]]).to(device)
    
    output_tokens = []
    
    for _ in range(max_length - 1):  # -1 to account for <sos> token
        tgt_mask = create_look_ahead_mask(tgt_tensor.size(1)).to(device)
        
        with torch.no_grad():
            output = model.decoder(tgt_tensor, enc_output, src_mask, tgt_mask)
            output = model.final_layer(output)
        
        predicted_token_id = output[:, -1, :].argmax(dim=1)
        predicted_token = tgt_idx2word[predicted_token_id.item()]
        output_tokens.append(predicted_token)
        
        if predicted_token == '<eos>':
            break
        
        tgt_tensor = torch.cat([tgt_tensor, predicted_token_id.unsqueeze(0)], dim=1)
    
        #print(f"Debug - Output tokens: {output_tokens}")
    return ' '.join(token for token in output_tokens if token not in ['<sos>', '<eos>', '<pad>'])



def evaluate_and_save(model, test_src, test_tgt, src_word2idx, tgt_idx2word, max_seq_length, device, output_file):
    model.eval()
    smooth = SmoothingFunction().method1
    total_bleu_score = 0
    num_sentences = len(test_src)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Source Sentence\tBLEU Score\tTranslation\tReference\n")

        for src_sentence, tgt_sentence in tqdm(zip(test_src, test_tgt), total=num_sentences, desc="Evaluating"):
            model_output = translate(model, src_sentence, src_word2idx, tgt_idx2word, max_seq_length, device)
            
            # Tokenize sentences
            reference = [tgt_sentence.split()]
            hypothesis = model_output.split()
            
            # Calculate BLEU score
            bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smooth)
            total_bleu_score += bleu_score
            
            # Write to file
            f.write(f"{src_sentence}\t{bleu_score:.4f}\t{model_output}\t{tgt_sentence}\n")

    avg_bleu_score = total_bleu_score / num_sentences
    
    # Write average BLEU score at the end of the file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\nAverage BLEU Score: {avg_bleu_score:.4f}")

    print(f'Average BLEU Score: {avg_bleu_score:.4f}')
    print(f'Results written to {output_file}')


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
    
    model_path = 'transformer.pt'
    model.load_state_dict(torch.load(model_path,map_location=device))

    # Test model
    test_loss = evaluate_model(model, test_dataloader, nn.CrossEntropyLoss(ignore_index=0), device)
    print(f"Test Loss: {test_loss:.4f}")

    # Translate a sample sentence
    sample_sentence = "Hello, how are you?"
    translated_sentence = translate(model, sample_sentence, src_word2idx, tgt_idx2word, max_seq_length, device)
    print(f"Sample translation:")
    print(f"Input: {sample_sentence}")
    print(f"Output: {translated_sentence}") 
    output_file = "translation_evaluation_results.txt"
    evaluate_and_save(model, test_src, test_tgt, src_word2idx, tgt_idx2word, max_seq_length, device, output_file)

    