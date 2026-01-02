Problem - genomic sequences like DNA are huge (> millions of base pairs), it is difficult to define a vocabulary over these sequences to train a machine learning model over. 
Solution - dynamic tokenization as in train the model to skim over repretivive sequences and focus on information dense parts in the sequence. 

The method to train this model is introduced in the paper, through the following steps: 
1. ToMe - a model that goes over the sequences and if they are repetitive, moves them into a single token and leave the other parts as is. 
2. Latent encoder - the model transforms the tokenized sequences into a latent space that is then trained using an appropriate objective. 

Training tasks: 
- re constructing the ToMe breakdown into the original sequence 
- masked training over the encoded embeddings over the dynamic tokens 

As reported in the paper, this approach gets an improved performance on tasks like promoter classification.