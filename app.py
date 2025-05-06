# app.py

import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizers
model = tf.keras.models.load_model('translator.h5')

with open('eng_tokenizer.pkl', 'rb') as f:
    eng_tokenizer = pickle.load(f)
with open('hin_tokenizer.pkl', 'rb') as f:
    hin_tokenizer = pickle.load(f)

reverse_hin_word_index = {v:k for k,v in hin_tokenizer.word_index.items()}

def translate(sentence):
    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=10, padding='post')
    
    decoder_input = np.zeros((1, 1))
    output_sentence = ''
    
    # Encode input
    enc_out, enc_state = model.layers[2](model.layers[1](seq))
    
    # First input to the decoder is start token (or zero)
    dec_state = enc_state
    for i in range(10):
        dec_emb = model.layers[4](decoder_input)
        dec_out, dec_state = model.layers[5](dec_emb, initial_state=dec_state)
        preds = model.layers[6](dec_out)
        
        predicted_id = np.argmax(preds[0, -1, :])
        if predicted_id == 0:
            break
        output_sentence += reverse_hin_word_index.get(predicted_id, '') + ' '
        
        decoder_input = np.array([[predicted_id]])
    
    return output_sentence.strip()

if __name__ == "__main__":
    while True:
        inp = input("\nEnter English Sentence (or 'quit'): ")
        if inp.lower() == 'quit':
            break
        print("Hindi Translation:", translate(inp))
