# SAE

Annotated data and code for our COLING 2022 paper: [A Structure-Aware Argument Encoder for Literature Discourse Analysis]().

## Files and Folders

- **data folder**：Data (For details, see "Data.zip". Directly copy all files in the "data" folder in the .zip file)
  
- **log folder**：Log file during traning
  
- **model folder**
  
  - BERT_LSTM_arg.py：Model code (SAE, SE)
  - BERT_arg.py：Model code (p-SAE)
  
- **output folder**：Save the best model on the test set during training
  
- **preds文件夹**：Save predict result in zero-shot transfer learning

- **pretrained_model folder**：Pretrained model (Download "bert-base-uncased" (https://huggingface.co/bert-base-uncased/tree/main) and put it here)

- **util folder**：Utility function code

  - config.py：Configuration file to get command line parameters
  - measure.py：Define performance metrics (accuracy, precision, recall, F1) and print metrics reports

- **train.py**：Training code

- **train_inference.py**：Code for inference only, used for zero-shot transfer learning, predict results saved in the "preds" folder

- **data_pre.py**：Daraloader code

- **command.md**：Command line example file

  

## Experiment Environment

- cuda10.1

  - python 3.8.5, cudatoolkit: 10.1, torch: 1.7.1, transformer, nltk, scikit-learn
  - pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
- cuda11.1

  - python 3.8.5, cudatoolkit: 11.1, torch: 1.8.1, transformer, nltk, scikit-learn
  
  - pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
  
    

## Get Started

You can try command lines in the "command.md" file.

## Annotation Tool

The annotation tool is available at https://github.com/lemuria-wchen/amcl. 


