# tmp_repo
Temporary repository to facilitate the manipulation of REVE.

REVE forward takes as input:

eeg: (B,C,T), T needs to be a multiple of 100 ideally.
pos: (B,3), the second dimension being (x,y,z) coordinate, ideally in the 10-5 surface.
return_output: Boolean to return the output of each transformer block.


Weights of the models are accessible at:

https://drive.google.com/drive/folders/1b2f8hRtNBeNN25ytZ4U12uRNVq5n3B_L?usp=share_link


Incomming:

- Notebook to finetune REVE on PhysionetMI
- Notebook to visualize embeddings
- Notebook to visualize attention maps