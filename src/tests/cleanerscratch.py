import torch
from transformers import BertTokenizer, AutoModelForSequenceClassification

# Get the SciBERT pretrained model path from Allen AI repo
pretrained_model = 'allenai/scibert_scivocab_uncased'

# Get the tokenizer from the previous path
sciBERT_tokenizer = BertTokenizer.from_pretrained(pretrained_model, 
                                          do_lower_case=True)

# Get the model
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model,
                                                          output_attentions=False,
                                                          output_hidden_states=True)

def convert_single_abstract_to_embedding(
        tokenizer, model, in_text, 
    ):
    
    input_ids = tokenizer.encode(
                        in_text, 
                        add_special_tokens = True, 
                        padding = True,
                   )    

    # Create attention masks    
    attention_mask = [int(i>0) for i in input_ids]
    
    # Convert to tensors.
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)

    # Add an extra dimension for the "batch" (even though there is only one 
    # input in this batch.)
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    #input_ids = input_ids.to(device)
    #attention_mask = attention_mask.to(device)
    
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():        
        _, encoded_layers = model( # discard logits
                                    input_ids = input_ids, 
                                    token_type_ids = None, 
                                    attention_mask = attention_mask,
                                    return_dict=False)

    layer_i = 12 # The last BERT layer before the classifier.
    batch_i = 0 # Only one input in the batch.
    token_i = 0 # The first token, corresponding to [CLS]
        
    # Extract the embedding.
    embedding = encoded_layers[layer_i][batch_i][token_i]

    # Move to the CPU and convert to numpy ndarray.
    embedding = embedding.detach().cpu().numpy()

    return embedding


abstract_str = "We use cosmological hydrodynamic simulations with stellar feedback from the FIRE (Feedback In Realistic Environments) project to study the physical nature of Lyman limit systems (LLSs) at z ≤ 1. At these low redshifts, LLSs are closely associated with dense gas structures surrounding galaxies, such as galactic winds, dwarf satellites and cool inflows from the intergalactic medium. Our analysis is based on 14 zoom-in simulations covering the halo mass range M<SUB>h</SUB> ≈ 10<SUP>9</SUP>-10<SUP>13</SUP> M<SUB>⊙</SUB> at z = 0, which we convolve with the dark matter halo mass function to produce cosmological statistics. We find that the majority of cosmologically selected LLSs are associated with haloes in the mass range 10<SUP>10</SUP> ≲ M<SUB>h</SUB> ≲ 10<SUP>12</SUP> M<SUB>⊙</SUB>. The incidence and H I column density distribution of simulated absorbers with columns in the range 10^{16.2} ≤ N_{H I} ≤ 2× 10^{20} cm<SUP>-2</SUP> are consistent with observations. High-velocity outflows (with radial velocity exceeding the halo circular velocity by a factor of ≳ 2) tend to have higher metallicities ([X/H] ∼ -0.5) while very low metallicity ([X/H] &lt; -2) LLSs are typically associated with gas infalling from the intergalactic medium. However, most LLSs occupy an intermediate region in metallicity-radial velocity space, for which there is no clear trend between metallicity and radial kinematics. The overall simulated LLS metallicity distribution has a mean (standard deviation) [X/H] = -0.9 (0.4) and does not show significant evidence for bimodality, in contrast to recent observational studies, but consistent with LLSs arising from haloes with a broad range of masses and metallicities."

e = convert_single_abstract_to_embedding(
    tokenizer=sciBERT_tokenizer,
    model=model,
    in_text=abstract_str,
)
print(type(e))
print(e.shape)