from CuFun import tpp
import numpy as np
import torch
from copy import deepcopy
from tensorboardX import SummaryWriter
torch.set_default_tensor_type(torch.cuda.FloatTensor)

writer = SummaryWriter()

# Config

seed = 0  # optional
np.random.seed(seed)
torch.manual_seed(seed)

# General data config
dataset_name = 'real_ours/wikipedia'
split = 'whole_sequences'

# General model config
use_history = True
history_size = 64
rnn_type = 'RNN'
use_embedding = False  # Ignored
embedding_size = 32

trainable_affine = False

# Decoder config
decoder_name = 'FullyNeuralNet_Ours'  # other: ['RMTPP', 'FullyNeuralNet', 'Exponential',
# 'LogNormMix', 'FullyNeuralNet_Ours']
n_components = 64           # Number of components for a mixture model for Log-Norm-Mix
hypernet_hidden_sizes = []  # Number of units in NN


max_degree = 3   # Ignored
n_terms = 4      # Ignored

n_layers = 3     # Number of layers for MNN
layer_size = 64  # Number of units in a layer

# Training config
regularization = 1e-5  # L2 regularization parameter
learning_rate = 1e-3   # Learning rate for Adam optimizer
max_epochs = 3000      # For how many epochs to train  3000
display_step = 100     # Display training statistics after every display_step
patience = 100         # After how many consecutive epochs without improvement of val loss to stop training  100

# Data
print('Loading data...')
if '+' not in dataset_name:
    dataset = tpp.data.load_dataset(dataset_name)
else:
    dataset_names = [d.strip() for d in dataset_name.split('+')]
    dataset = tpp.data.load_dataset(dataset_names.pop(0))
    for d in dataset_names:
        dataset += tpp.data.load_dataset(dataset_names.pop(0))

# Split into train/val/test, on each sequence or assign whole sequences to different sets
if split == 'each_sequence':
    d_train, d_val, d_test = dataset.train_val_test_split_each(seed=seed)
elif split == 'whole_sequences':
    d_train, d_val, d_test = dataset.train_val_test_split_whole(seed=seed)
else:
    raise ValueError(f'Unsupported dataset split {split}')

# Calculate mean and std of the input inter-event times and normalize only input
mean_in_train, std_in_train = d_train.get_mean_std_in()
std_out_train = 1.0
d_train.normalize(mean_in_train, std_in_train, std_out_train)
d_val.normalize(mean_in_train, std_in_train, std_out_train)
d_test.normalize(mean_in_train, std_in_train, std_out_train)

# Break down long train sequences for faster batch traning and create torch DataLoaders
d_train.break_down_long_sequences(128)
collate = tpp.data.collate
dl_train = torch.utils.data.DataLoader(d_train, batch_size=64, shuffle=True, collate_fn=collate)
dl_val = torch.utils.data.DataLoader(d_val, batch_size=1, shuffle=False, collate_fn=collate)
dl_test = torch.utils.data.DataLoader(d_test, batch_size=1, shuffle=False, collate_fn=collate)

# Set the parameters for affine normalization layer depending on the decoder (see Appendix D.3 in the paper)
if decoder_name in ['RMTPP', 'FullyNeuralNet', 'Exponential', 'FullyNeuralNet_Ours', 'Const']:
    _, std_out_train = d_train.get_mean_std_out()
    mean_out_train = 0.0
else:
    mean_out_train, std_out_train = d_train.get_log_mean_std_out()


# Model setup
print('Building model...')

# General model config
general_config = tpp.model.ModelConfig(
    use_history=use_history,
    history_size=history_size,
    rnn_type=rnn_type,
    use_embedding=use_embedding,
    embedding_size=embedding_size,
    num_embeddings=len(dataset),
)

# Decoder specific config
decoder = getattr(tpp.decoders, decoder_name)(general_config,
                                              n_components=n_components,
                                              hypernet_hidden_sizes=hypernet_hidden_sizes,
                                              max_degree=max_degree,
                                              n_terms=n_terms,
                                              n_layers=n_layers,
                                              layer_size=layer_size,
                                              shift_init=mean_out_train,
                                              scale_init=std_out_train,
                                              trainable_affine=trainable_affine)

# Define model
model = tpp.model.Model(general_config, decoder)
model.use_history(general_config.use_history)
model.use_embedding(general_config.use_embedding)

# Define optimizer
opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate)


# Traning
print('Starting training...')

# Function that calculates the loss for the entire dataloader


def get_total_loss(loader):
    loader_log_prob, loader_lengths = [], []
    for input in loader:
        loader_log_prob.append(model.log_prob(input)[0].detach())
        loader_lengths.append(input.length.detach())
    return -model.aggregate(loader_log_prob, loader_lengths)


impatient = 0
best_loss = np.inf
best_model = deepcopy(model.state_dict())
training_val_losses = []

for epoch in range(max_epochs):
    model.train()
    for input in dl_train:
        opt.zero_grad()
        log_prob = model.log_prob(input)[0]
        loss = -model.aggregate(log_prob, input.length)
        loss.backward()
        opt.step()

    term1 = model.log_prob(input)[1][0]
    term2 = model.log_prob(input)[1][1]
    writer.add_scalars('./comparison', {"term1": term1, "term2": term2}, epoch)
    model.eval()
    loss_val = get_total_loss(dl_val)
    training_val_losses.append(loss_val.item())

    if (best_loss - loss_val) < 1e-4:
        impatient += 1
        if loss_val < best_loss:
            best_loss = loss_val.item()
            best_model = deepcopy(model.state_dict())
    else:
        best_loss = loss_val.item()
        best_model = deepcopy(model.state_dict())
        impatient = 0

    if impatient >= patience:
        print(f'Breaking due to early stopping at epoch {epoch}')
        break

    if (epoch + 1) % display_step == 0:
        print(f"Epoch {epoch+1:4d}, loss_train_last_batch = {loss:.4f}, loss_val = {loss_val:.4f}")

# Evaluation

model.load_state_dict(best_model)
model.eval()

pdf_loss_train = get_total_loss(dl_train)
pdf_loss_val = get_total_loss(dl_val)
pdf_loss_test = get_total_loss(dl_test)

print(f'Time NLL\n'
      f' - Test:  {pdf_loss_test.item():.4f}\n'
      f' - Val:   {pdf_loss_val.item():.4f}\n'
      f' - Train: {pdf_loss_train:.4f}')
