from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
model.eval()
sentences = [ 
              "Hello I'm a single sentence",
              "And another sentence",
              "And the very very last one",
              "Hello I'm a single sentence",
              "And another sentence",
              "And the very very last one",
              "Hello I'm a single sentence",
              "And another sentence",
              "And the very very last one",
            ]
batch_size = 9
for idx in range(0, len(sentences), batch_size):
    batch = sentences[idx : min(len(sentences), idx+batch_size)]
    
    # encoded = tokenizer(batch)
    encoded = tokenizer.batch_encode_plus(batch,max_length=50, padding='max_length', truncation=True)
  
    encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
    with torch.no_grad():
        
        outputs = model(**encoded)
        
    
    # print(outputs.last_hidden_state.size())
    first_hidden_state = outputs[0] # Shape : (batch_size, 256, 768)
    cls_tokens = first_hidden_state[:,0,:] # Shape : (2, 768)
    # print(cls_tokens.size()) 

    # print(cls_tokens[0, :5]) # tensor([-0.1360,  0.2572,  0.0720, -0.2640, -0.3469])


