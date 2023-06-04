"https://towardsdatascience.com/scientific-documents-similarity-search-with-deep-learning-using-transformers-scibert-d47c4e501590"

from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

model.eval()

abstract_str = "We use cosmological hydrodynamic simulations with stellar feedback from the FIRE (Feedback In Realistic Environments) project to study the physical nature of Lyman limit systems (LLSs) at z ≤ 1. At these low redshifts, LLSs are closely associated with dense gas structures surrounding galaxies, such as galactic winds, dwarf satellites and cool inflows from the intergalactic medium. Our analysis is based on 14 zoom-in simulations covering the halo mass range M<SUB>h</SUB> ≈ 10<SUP>9</SUP>-10<SUP>13</SUP> M<SUB>⊙</SUB> at z = 0, which we convolve with the dark matter halo mass function to produce cosmological statistics. We find that the majority of cosmologically selected LLSs are associated with haloes in the mass range 10<SUP>10</SUP> ≲ M<SUB>h</SUB> ≲ 10<SUP>12</SUP> M<SUB>⊙</SUB>. The incidence and H I column density distribution of simulated absorbers with columns in the range 10^{16.2} ≤ N_{H I} ≤ 2× 10^{20} cm<SUP>-2</SUP> are consistent with observations. High-velocity outflows (with radial velocity exceeding the halo circular velocity by a factor of ≳ 2) tend to have higher metallicities ([X/H] ∼ -0.5) while very low metallicity ([X/H] &lt; -2) LLSs are typically associated with gas infalling from the intergalactic medium. However, most LLSs occupy an intermediate region in metallicity-radial velocity space, for which there is no clear trend between metallicity and radial kinematics. The overall simulated LLS metallicity distribution has a mean (standard deviation) [X/H] = -0.9 (0.4) and does not show significant evidence for bimodality, in contrast to recent observational studies, but consistent with LLSs arising from haloes with a broad range of masses and metallicities."

# sentences = abstract_str.split(".")
# print(sentences)
sentences = [abstract_str]

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
    print(cls_tokens.size()) 

    # print(cls_tokens[0, :5]) # tensor([-0.1360,  0.2572,  0.0720, -0.2640, -0.3469])