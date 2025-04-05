import time

out_dir = 'out-ft-sentiment'
eval_interval = 5
eval_iters = 40
wandb_log = True # feel free to turn on
wandb_project = 'sentiment-analysis'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'customer_service'
init_from = 'gpt2' # this is the GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = True  

# the number of examples per iter:
batch_size = 8
gradient_accumulation_steps = 32
max_iters = 2000

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False