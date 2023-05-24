# TN-LCFRS
Official Implementation of ACL2023:  [Unsupervised Discontinuous Constituency Parsing with Mildly Context-Sensitive Grammars](https://arxiv.org/pdf/2212.09140.pdf). 

# Data
For simplicity, you can download pre-processed data and also pretrained models at [Google Drive](https://drive.google.com/drive/folders/1kbdBfm4q-ExwS80qUdvtuyOvSlKd_E0o?usp=sharing) 

Or, you can generate the pickle files by your own using discobracket files and '''corpus_reader.py''' function



# Run
## Dependencies
Follow the instruction of [TN-PCFG](https://github.com/sustcsonglin/TN-PCFG) to install dependnecies. Then install [triton](https://github.com/openai/triton)=2.0.0 and use CUDA=11.4

## Train
python train.py --conf=path/to/config 

e.g.: python train.py --conf=config/tnlcfrs_de.yaml

## Evaluate
python evaluate.py --load_from_dir=path/to/your/saved_dir  --test_file=path/to/your/test_pickle_file

e.g.: python evaluate.py --load_from_dir=log/LCFRS_rank_full22022-11-08-00_04_02 --test_file='data/dev_nopunct_negra.pickle'


## Useful command
### Draw Trees
discodop treedraw --fmt=discbracket --output=svg something.txt > something.html
 


# Contact
Feel free to contact bestsonta@gmail.com if you have any questions.




 
  
