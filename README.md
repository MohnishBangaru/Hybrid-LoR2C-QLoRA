# Hybrid LoR2C QLoRA
How to train LoR2c Model
```
#python 3.10
pip install fsspec==2025.3.2 accelerate==0.22.0 transformers==4.31.0 evaluate appdirs bitsandbytes datasets fire sentencepiece scipy scikit-learn wandb fire
pip install -q --force-reinstall numpy==1.24.4 pandas==1.5.3 pyarrow==10.0.1
unzip "archive.zip" && cd "peft-0.5.0" && pip install -e .

python3 trainer.py
```
### The Metrics (7b vs 1b)
![alt text](image.png)

![alt text](7b_vs_1b_metrics.png)

### The preview of the LOR2C Model
![alt text](lor2c_preview.png)