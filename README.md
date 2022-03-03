# Dialogue Summaries as Dialogue States (DS2)

0. Installing the directory as pip will resolve all path issues
```bash
pip install -e .
pip install -r requirements.txt # requires python 3.8
```

1. Get MWOZ data (for 2.0 change all 2.1 to 2.0)

For 2.0
```bash
python scripts/create_data_mwoz.py --mwz_ver=2.0 --main_dir=data_mwoz_2.0 --target_path=data_mwoz_2.0/mwz
```

For 2.1
```bash
python scripts/create_data_mwoz.py --mwz_ver=2.1 --main_dir=data_mwoz_2.1 --target_path=data_mwoz_2.1/mwz
```

2. Training and Inference - Cross-domain 
**Pre-training**
Example using T5 on Cross-domain pre-training. Note that this code will not work yet because we did not release our pretrained model checkpoints yet due to anonymity issues. We will release the checkpoints upon de-anonymization of the paper. Hence, we recommend using the following options to check our code.
- `model_name=bart` and `model_checkpoint=Salesforce/bart-large-xsum-samsum` 

```bash
CUDA_VISIBLE_DEVICES={gpu} python ds2/scripts/train_ds2.py \
    --dev_batch_size=8 \
    --test_batch_size=8 \
    --train_batch_size=2 \
    --n_epochs=100 \
    --num_beams=1 \
    --test_num_beams=1 \
    --val_check_interval=1.0 \
    --fewshot=0.01 \
    --grad_acc_steps=1 \
    --model_name=bart \
    --model_checkpoint=Salesforce/bart-large-xsum-samsum \
    --except_domain=attraction \
    --mode=finetune \
    --exp_name=bart-CD-1-Attr-pre \
    --seed=577 \
    --version=2.1
```

**Fine-tuning**
```bash
CUDA_VISIBLE_DEVICES={gpu} python ds2/scripts/train_ds2.py \
    --dev_batch_size=8 \
    --test_batch_size=8 \
    --train_batch_size=2 \
    --n_epochs=100 \
    --num_beams=1 \
    --test_num_beams=1 \
    --val_check_interval=1.0 \
    --fewshot=0.01 \
    --grad_acc_steps=1 \
    --model_name=bart \
    --model_checkpoint=Salesforce/bart-large-xsum-samsum \
    --only_domain=attraction \
    --mode=finetune \
    --load_pretrained={bart-CD-1-Attr-pre/ckpt_path} \
    --exp_name=bart-CD-1-Attr \
    --seed=577 \
    --version=2.1
```

3. Training and Inference - Multi-domain
```bash
CUDA_VISIBLE_DEVICES={gpu} python ds2/scripts/train_ds2.py \
    --dev_batch_size=8 \
    --test_batch_size=8 \
    --train_batch_size=2 \
    --n_epochs=100 \
    --num_beams=1 \
    --test_num_beams=1 \
    --val_check_interval=1.0 \
    --fewshot=0.01 \
    --grad_acc_steps=1 \
    --model_name=bart \
    --model_checkpoint=Salesforce/bart-large-xsum-samsum \
    --mode=finetune \
    --exp_name=bart-MD-1 \
    --seed=577 \
    --version=2.1
```

4. Training and Inference - Cross-task
```bash
CUDA_VISIBLE_DEVICES={gpu} python ds2/scripts/train_ds2.py \
    --dev_batch_size=8 \
    --test_batch_size=8 \
    --train_batch_size=2 \
    --n_epochs=100 \
    --num_beams=1 \
    --test_num_beams=1  \
    --val_check_interval=1.0 \
    --fewshot=0.01 \
    --grad_acc_steps=1 \
    --model_name=bart \
    --model_checkpoint=Salesforce/bart-large-xsum-samsum \
    --mode=finetune  \
    --only_domain=attraction \
    --exp_name=bart-CT-Attr-1 \
    --seed=577 \
    --version=2.1
```