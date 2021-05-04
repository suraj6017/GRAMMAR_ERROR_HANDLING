#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:



#all imports
import pickle
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from keras.layers import Dense
from keras.layers import Embedding
from tensorflow.keras.layers import LSTM


train = pickle.load(open('/content/drive/MyDrive/train.pkl','rb'))

embedding_matrix = pickle.load(open('/content/drive/MyDrive/embedding_matrix.pkl','rb'))


tknizer_ERRONEOUS_SENTENCE = pickle.load(open('/content/drive/MyDrive/tknizer_ERRONEOUS_SENTENCE.pkl','rb'))
tknizer_CORRECT_SENTENCE = pickle.load(open('/content/drive/MyDrive/tknizer_CORRECT_SENTENCE.pkl','rb'))


vocab_size_CORRECT_SENTENCE=len(tknizer_CORRECT_SENTENCE.word_index.keys())
print(vocab_size_CORRECT_SENTENCE)
vocab_size_ERRONEOUS_SENTENCE=len(tknizer_ERRONEOUS_SENTENCE.word_index.keys())
print(vocab_size_ERRONEOUS_SENTENCE)

embedding_matrix = pickle.load(open('/content/drive/MyDrive/embedding_matrix.pkl','rb'))


vanilla = tf.keras.models.load_model('/content/drive/MyDrive/save_model/enc_dec')
# vanilla.summary()

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, input_length, enc_units):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.enc_units= enc_units
        self.lstm_output = 0
        self.lstm_state_h=0
        self.lstm_state_c=0
        
    def build(self, input_shape):
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                           mask_zero=True, name="embedding_layer_encoder", input_shape=(self.vocab_size,))
        self.lstm = LSTM(self.enc_units, return_state=True, return_sequences=True, name="Encoder_LSTM")
        
    def call(self, input_sentances, training=True):
        input_embedd                        = self.embedding(input_sentances)
        self.lstm_output, self.lstm_state_h,self.lstm_state_c = self.lstm(input_embedd)
        return self.lstm_output, self.lstm_state_h,self.lstm_state_c
    def get_states(self):
        return self.lstm_state_h,self.lstm_state_c


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, input_length, dec_units):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.input_length = input_length
    
    def build(self, input_shape):
        # we are using embedding_matrix weights and not training the embedding layer
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                           mask_zero=True, name="embedding_layer_decoder", weights=[embedding_matrix],input_shape=(self.vocab_size,))
        self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True, name="Encoder_LSTM")
        
    def call(self, target_sentances, state_h, state_c):
        target_embedd           = self.embedding(target_sentances)
        lstm_output, _,_        = self.lstm(target_embedd, initial_state=[state_h, state_c])
        return lstm_output

class vanilla_model(Model):
    def __init__(self, encoder_inputs_length,decoder_inputs_length, output_vocab_size):
        super().__init__() 
        self.encoder = Encoder(vocab_size=vocab_size_ERRONEOUS_SENTENCE + 1, embedding_dim=100, input_length=encoder_inputs_length, enc_units=256)
        self.decoder = Decoder(vocab_size=vocab_size_CORRECT_SENTENCE + 1, embedding_dim=100, input_length=decoder_inputs_length, dec_units=256)
        self.dense   = Dense(output_vocab_size, activation='softmax')
        
        
    def call(self, data):
        input,output = data[0], data[1]
        encoder_output, encoder_h, encoder_c = self.encoder(input)
        decoder_output                       = self.decoder(output, encoder_h, encoder_c)
        output                               = self.dense(decoder_output)
        return output

vanilla = vanilla_model(encoder_inputs_length=16,decoder_inputs_length=16,output_vocab_size=vocab_size_CORRECT_SENTENCE)



#FUNCTION inference TO CORRECT THE INCORRECT SENTENCES
def inference(enc_inp,dec_inp):
            
    translation=""

    e_input=[]
    for i in enc_inp.split():
        if tknizer_ERRONEOUS_SENTENCE.word_index.get(i) == None:
            e_input.append(0)
        else:
            e_input.append(tknizer_ERRONEOUS_SENTENCE.word_index.get(i))

    e_output, e_hidden, e_cell = vanilla.layers[0](np.array([e_input], dtype='int32'))

    d_input=[]
    for i in dec_inp.split():
        if tknizer_CORRECT_SENTENCE.word_index.get(i) == None:
            d_input.append(0)
        else:
            d_input.append(tknizer_CORRECT_SENTENCE.word_index.get(i))

    prediction = vanilla.layers[2](vanilla.layers[1](np.array([d_input], dtype='int32'),e_hidden,e_cell))

    for word in prediction[0]:
        word = tknizer_CORRECT_SENTENCE.index_word[tf.argmax(word).numpy()]
        if word == "<end>":
            break
        translation += word + " "
    return translation



app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))

# @app.route('/')
# def hello_world():
#     return 'hello_world'


@app.route('/')
def form():
    return """
        <!doctype html>
        <html>
          <head>
            <title>GRAMMAR ERRROR HANDLING AND CORRECTION</title>
          </head>
          <body>
            <h1>GRAMMAR ERRROR HANDLING AND CORRECTION</h1>
            <form method="POST" action="/predict" enctype="multipart/form-data">
              <p><input type="text" name="file"></p>
              <p><input type="submit" value="Submit"></p>
            </form>
          </body>
        </html>
    """


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    string_inp = request.form
    print(string_inp)
    print('THE INPUT IS HAVING TYPE : ' ,type(string_inp))
    print('SUCCESS in READING')
      
    start_time = time.time()
    vanilla = tf.keras.models.load_model('save_model/enc_dec') #LOADING MY PRETRAINED MODEL
    a = string_inp
    b =  '<start> ' + string_inp
    pred = inference(a,b)
    print('THE CORRECTED SENTENCES IS', pred)
    
    print('DONE PREDICTION')
    end_time = time.time()
    print('TIME TAKEN TO PREDICT IS : {}'.format(end_time - start_time))
    
    return render_template('index.html',pred='THE PREDICTION FOR THE INPUT STRING IS',pred)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)

