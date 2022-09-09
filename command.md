# 1. SAE

## 1.1 CCSA

python train.py \
--type=argencoding \
--arg_encoding=True \
--lstm=True \
--PTE=position \
--train_size=0.6 \
--val_size=0.2 \
--epochs=10 \
--label_dim=5 \
--arg_encoding_type=max \
--data_split_type=order \
--log_name=CCSA_log_SAE \
--model_save_name=CCSA_model_SAE \
--data_load_path=./data/sentence_type/CCSA_type.json

## 1.2 Biomedical

python train.py \
--type=argencoding \
--arg_encoding=True \
--lstm=True \
--PTE=position \
--train_size=0.5 \
--val_size=0.25 \
--epochs=50 \
--label_dim=2 \
--arg_encoding_type=max \
--data_split_type=order \
--log_name=BiomedicalPublications_log_SAE \
--model_save_name=BiomedicalPublications_model_SAE \
--data_load_path=./data/sentence_type/BiomedicalPublications_type.json

# 2. p-SAE

Environment: cuda11.1 (Error on cuda 101)

## 2.1 CCSA

python train.py \
--type=tokentype \
--arg_encoding=False \
--lstm=True \
--PTE=position \
--train_size=0.6 \
--val_size=0.2 \
--epochs=10 \
--label_dim=5 \
--arg_encoding_type=max \
--data_split_type=order \
--log_name=CCSA_log_p-SAE \
--model_save_name=CCSA_model_p-SAE \
--data_load_path=./data/sentence_type/CCSA_type.json

## 2.2 Biomedical

python train.py \
--type=tokentype \
--arg_encoding=False \
--lstm=True \
--PTE=position \
--train_size=0.5 \
--val_size=0.25 \
--epochs=50 \
--label_dim=2 \
--arg_encoding_type=max \
--data_split_type=order \
--log_name=BiomedicalPublications_log_p-SAE \
--model_save_name=BiomedicalPublications_model_p-SAE \
--data_load_path=./data/sentence_type/BiomedicalPublications_type.json

# 3. SE

## 3.1 CCSA

python train.py \
--type=argencoding \
--arg_encoding=False \
--lstm=True \
--PTE=no \
--train_size=0.6 \
--val_size=0.2 \
--epochs=10 \
--label_dim=5 \
--arg_encoding_type=max \
--data_split_type=order \
--log_name=CCSA_log_SE \
--model_save_name=CCSA_model_SE \
--data_load_path=./data/sentence_type/CCSA_type.json

## 3.2 Biomedical

python train.py \
--type=argencoding \
--arg_encoding=False \
--lstm=True \
--PTE=no \
--train_size=0.5 \
--val_size=0.25 \
--epochs=50 \
--label_dim=2 \
--arg_encoding_type=max \
--data_split_type=order \
--log_name=BiomedicalPublications_log_SE \
--model_save_name=BiomedicalPublications_model_SE \
--data_load_path=./data/sentence_type/BiomedicalPublications_type.json



# 4. BERT

## 4.1 CCSA

python train.py \
--type=argencoding \
--arg_encoding=False \
--lstm=False \
--PTE=no \
--train_size=0.6 \
--val_size=0.2 \
--epochs=10 \
--label_dim=5 \
--arg_encoding_type=max \
--data_split_type=order \
--log_name=CCSA_log_BERT \
--model_save_name=CCSA_model_BERT \
--data_load_path=./data/sentence_type/CCSA_type.json

## 4.2 Biomedical

python train.py \
--type=argencoding \
--arg_encoding=False \
--lstm=False \
--PTE=no \
--train_size=0.5 \
--val_size=0.25 \
--epochs=50 \
--label_dim=2 \
--arg_encoding_type=max \
--data_split_type=order \
--log_name=BiomedicalPublications_log_noargencoding_BERT_order_noposition_0 \
--model_save_name=BiomedicalPublications_model_noargencoding_BERT_order_noposition_0 \
--data_load_path=./data/sentence_type/BiomedicalPublications_type.json

# 5. Transfer Learning: CCSA -> Biomedical

## 5.1 SAE

python train_inference.py \
--type=argencoding \
--arg_encoding=True \
--lstm=True \
--PTE=position \
--train_size=0.5 \
--val_size=0.25 \
--label_dim=5 \
--transfer_label_dim=2 \
--transfer_type=5-2 \
--arg_encoding_type=max \
--data_split_type=order \
--preds_name=Zero-shot_TL_preds_SAE \
--log_name=Zero-shot_TL_log_SAE \
--model_save_name=Zero-shot_TL_model_SAE \
--data_load_path=./data/sentence_type/BiomedicalPublications_type.json

## 5.2 p-SAE

python train_inference.py \
--type=tokentype \
--arg_encoding=False \
--lstm=True \
--PTE=position \
--train_size=0.5 \
--val_size=0.25 \
--label_dim=5 \
--transfer_label_dim=2 \
--transfer_type=5-2 \
--arg_encoding_type=max \
--data_split_type=order \
--preds_name=Zero-shot_TL_preds_p-SAE \
--log_name=Zero-shot_TL_log_p-SAE \
--model_save_name=Zero-shot_TL_model_p-SAE \
--data_load_path=./data/sentence_type/BiomedicalPublications_type.json

## 5.3 SE

python train_inference.py \
--type=argencoding \
--arg_encoding=False \
--lstm=True \
--PTE=no \
--train_size=0.5 \
--val_size=0.25 \
--label_dim=5 \
--transfer_label_dim=2 \
--transfer_type=5-2 \
--arg_encoding_type=max \
--data_split_type=order \
--preds_name=Zero-shot_TL_preds_SE \
--log_name=Zero-shot_TL_log_SE \
--model_save_name=Zero-shot_TL_model_SE \
--data_load_path=./data/sentence_type/BiomedicalPublications_type.json


