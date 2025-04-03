# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-sentiment'
eval_interval = 200
eval_iters = 50
log_interval = 10

always_save_checkpoint = True

wandb_log = True
wandb_project = 'sentiment-analysis'
wandb_run_name = 'gpt-sentiment'

dataset = 'customer_service'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 512

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
num_classes = 3  # For sentiment

learning_rate = 5e-5  # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000  # make equal to max_iters usually
min_lr = 5e-6  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially
