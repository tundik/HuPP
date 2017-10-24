# HuPP
HuPP - Hungarian Punctuation Prediction  

First word-based RNN models for Punctuation Predictions in Hungarian,  and our lightweight English models  

Command line parameters ( in the following order):  
- chunk_size : corpora is split to smaller sub-sequences ( e.g.  by #200-200.. words  )  
- vocabulary_size: number of words in the vocabulary (e.g. 20000)  
- embedding_size:  the size of embedding dimension ( e.g. 100 )  
- hidden_size: number of hidden states in the LSTM/BiLSTM layer ( e.g. 256 )  
- batch_size: batch size during training period   ( e.g.  200)  
- no_question: if this parameter yes, only 'COMMA' and 'PERIOD' evaluated   ('T/F': 0/1)  
- optimizer : training optimizer ( e.g. 'adam' )  
- patience : patience for Early Stopping ( e.g. 2)
