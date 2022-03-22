# Dialogue Summaries as Dialogue States (DS2)

**Updates:** We release the [T5 models](https://huggingface.co/jaynlp/t5-large-samsum) as anonymity period is over.

Paper link: https://arxiv.org/abs/2203.01552

### Citation
```
@article{shin2022dialogue,
  title={Dialogue Summaries as Dialogue States (DS2), Template-Guided Summarization for Few-shot Dialogue State Tracking},
  author={Shin, Jamin and Yu, Hangyeol and Moon, Hyeongdon and Madotto, Andrea and Park, Juneyoung},
  journal={arXiv preprint arXiv:2203.01552},
  year={2022}
}
```

## How to use the code
0. Installing the directory as pip will resolve all path issues
```bash
pip install -e .
pip install -r requirements.txt # requires python 3.8
```

**1. Get MWOZ data (for 2.0 change all 2.1 to 2.0)
**

For 2.0
```bash
python scripts/create_data_mwoz.py --mwz_ver=2.0 --main_dir=data_mwoz_2.0 --target_path=data_mwoz_2.0/mwz
```

For 2.1
```bash
python scripts/create_data_mwoz.py --mwz_ver=2.1 --main_dir=data_mwoz_2.1 --target_path=data_mwoz_2.1/mwz
```

**2. Training and Inference - Cross-domain 
**

**Pre-training**

~~Example using T5 on Cross-domain pre-training. Note that this code will not work yet because we did not release our pretrained model checkpoints yet due to anonymity issues. We will release the checkpoints upon de-anonymization of the paper. Hence, we recommend using the following options to check our code.~~

We have uploaded the T5 pre-trained on Dialogue Summarization model on HuggingFace Model Hub at https://huggingface.co/jaynlp/t5-large-samsum. Now you can choose between BART and T5 as such:
- `model_name=bart` and `model_checkpoint=Salesforce/bart-large-xsum-samsum` 
- `model_name=t5` and `model_checkpoint=jaynlp/t5-large-samsum` 

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

**3. Training and Inference - Multi-domain
**

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

**4. Training and Inference - Cross-task
**

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
