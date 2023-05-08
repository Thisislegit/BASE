# BASE: Bridging the Gap between Cost and Latency for Query Optimization

## Usage

BASE is a two stage training framework for learned query optimizer:  

* First stage: policy pretraining with cost from PostgreSQL cost model.

```bash
python -u RLGOOTest_NEW.py --path ./Models/now.pth --epoch_start 1 --epoch_end 100 --epsilon_decay 0.95 --epsilon_end 0.02 --capacity 60000 --batch_size 512 --sync_batch_size 50 --steps_per_epoch 1000 --max_lr 0.0008 --learning_rate 0.0003 > log_NEW.txt 2>&1
```

* Second stage (starting from the first stage): latency finetuning with latency from actual execution. 

```bash
python Transfer_Active.py
```



## Contact

xuchen.2019@outlook.com
