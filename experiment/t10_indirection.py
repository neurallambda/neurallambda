'''

sorting an AST

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import einsum

from datasets import Dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
import tokenizers
from functools import partial

# os.environ['HF_HUB_OFFLINE'] = '0'
from typing import List, Dict
from neurallambda.lab.common import (
    print_model_info,
    print_grid
)
import math
import random

import neurallambda.stack_joulin as S
import neurallambda.language as L
import neurallambda.memory as M
import neurallambda.neurallambda as N
import neurallambda.lab.common as NLCommon

import neurallambda.model.transformer01 as Transformer

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


##################################################
# Hyperparams

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

DEBUG = False

all_uppercase = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split(' ')
all_lowercase = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split(' ')
hex_digits = '0 1 2 3 4 5 6 7 8 9 A B C D E F'.split(' ')
digits = '0 1 2 3 4 5 6 7 8 9'.split(' ')

# PROBLEM = 'identity'
PROBLEM = 'sort'

TRAIN_N_DATA = 1000
TRAIN_MIN_SEQ_LEN = 4
TRAIN_MAX_SEQ_LEN = 8

TEST_N_DATA = 200
TEST_MIN_SEQ_LEN = 8
TEST_MAX_SEQ_LEN = 12

EMB_DIM = 256
NUM_HEADS = 4
DIM_FEEDFORWARD = 64
NUM_LAYERS = 2  # number of unique sequential transformer layers
NUM_RECURRENCE = 4  # number of times that same block of transformers is repeated for a single forward pass
DEVICE = 'cuda'
LR = 1e-3
NUM_EPOCHS = 40
BATCH_SIZE = 128
GRAD_CLIP = None

LOSS_FN = 'cross_entropy'
# LOSS_FN = 'cross_entropy_select_from_inputs'
# LOSS_FN = 'cosine_distance'
# LOSS_FN = 'nllloss'

SELECT_FROM_INPUTS = False


##################################################
# Data (indirection, sort)

# # @@@@@@@@@@
# # Check neurallambda conversions
# BATCH_SIZE = 1
# N_ADDRESSES = 16
# EMB_DIM = 1024
# DEVICE = 'cuda'

# program = "'(1 2 3)"
# ast = L.string_to_terms(program)
# mem = M.terms_to_memory(ast)

# nl = N.string_to_neurallambda(
#     program,
#     batch_size=BATCH_SIZE,
#     n_addresses=N_ADDRESSES,
#     vec_size=EMB_DIM,
#     zero_vec_bias=0.0,
#     device=DEVICE,
# )

# recon_mem = N.neurallambda_to_mem(
#     nl,
#     nl.addresses,
#     nl.tags,
#     nl.col1,
#     nl.col2,
#     n_ixs=N_ADDRESSES,
# )

# # # Print human-readable memory
# # print()
# # M.print_mem(recon_mem[0], None, None)
# # @@@@@@@@@@

def dfs_readout(nl):
    '''Traverse neurallambda memory, "flatten" the datastructure contained within. Differentiable.

    NOTE: This is intended to aid address-invariance. But I may not need
          it. Maybe we just get loss by comparing both resultant memory
          states. This problem generalizes into Graph Isomorphism, which is too
          complex i think to use in a loss fn. We can simplify by declaring a
          known correspondence between both memory banks, address=0. We could
          also use GNNs.

    '''

    n_addresses = nl.addresses.size(1)
    stack = S.initialize(EMB_DIM, n_addresses * 2, BATCH_SIZE, DEVICE)

    start_ix = 0
    start_addr = nl.addresses[:, start_ix]
    stack = S.push(stack, start_addr)

    out = []
    for _ in range(n_addresses * 2):
        stack, at_addr = S.pop(stack)
        head_tag, col1_addr, col2_addr = N.select_address(at_addr, nl.addresses, [nl.tags, nl.col1, nl.col2])

        out.append((head_tag, col1_addr, col2_addr))

        # Recur over branches
        is_col1_addr = N.address_similarity(col1_addr, nl.addresses).sum(dim=1).clip(0, 1)
        stack = S.push_or_null_op(stack, is_col1_addr, 1 - is_col1_addr, col1_addr)

        is_col2_addr = N.address_similarity(col2_addr, nl.addresses).sum(dim=1).clip(0, 1)
        stack = S.push_or_null_op(stack, is_col2_addr, 1 - is_col2_addr, col2_addr)

        BROKEN_TEST_FIRST  # pay attn to pushing col1/col2 as addresses if they're not, IE if those cols come from a literal type

    return out


# @@@@@@@@@@
# readout = dfs_readout(nl)

# print()
# for (tag, c1, c2) in readout:
#     print(nl.vec_to_tag(tag[0]), nl.vec_to_tag(c1[0]), nl.vec_to_tag(c2[0]), '|', nl.vec_to_address(c1[0], nl.addresses[0]), nl.vec_to_address(c2[0], nl.addresses[0]))
# @@@@@@@@@@



##################################################
# Data

def generate_dataset(n_data, min_seq_len, max_seq_len, digits):
    dataset = []
    while len(dataset) < n_data:
        seq_length = random.randint(min_seq_len, max_seq_len)
        input_list = [random.choice(digits) for _ in range(seq_length)]

        if PROBLEM == 'sort':
            output_list = sorted(input_list)
        elif PROBLEM == 'identity':
            output_list = input_list
        dataset.append((input_list, output_list))
    return dataset


def nl_embeddings(s, addresses):
    ast = L.string_to_terms(s)
    mem = M.terms_to_memory(ast)
    tags = []
    col1 = []
    col2 = []
    for addr, tup in sorted(mem.items(), key=lambda x: x[0].i):
        if tup[0] == 'IntLit':
            tags.append('IntLit')
            col1.append(str(tup[1]))
            col2.append('null')
        elif tup[0] == 'Cons':
            tags.append('Cons')
            col1.append(addresses[tup[1].i])
            col2.append(addresses[tup[2].i])
        elif tup[0] == 'Empty':
            tags.append('Empty')
            col1.append('null')
            col2.append('null')
    return tags, col1, col2


def prep(raw_data):
    print('prepping')
    all_vars = all_uppercase + all_lowercase
    out = []
    for inp_list, out_list in tqdm(raw_data):
        addresses = random.sample(all_vars, len(all_vars))  # assign random addresses
        inp_s = "'(" + ' '.join(inp_list) + ")"  # s-exp, see neurallambda.lang
        inp_tags, inp_col1, inp_col2 = nl_embeddings(inp_s, addresses)

        out_s = "'(" + ' '.join(out_list) + ")"  # s-exp, see neurallambda.lang
        out_tags, out_col1, out_col2 = nl_embeddings(out_s, addresses)

        out.append({'addresses': addresses,
                    'inp_tags': inp_tags,
                    'inp_col1': inp_col1,
                    'inp_col2': inp_col2,

                    'out_tags': out_tags,
                    'out_col1': out_col1,
                    'out_col2': out_col2,
                    })

    return out


def collate_fn(tokenizer, pad_token, batch):
    ''' Tokenize and pad batches of data. '''

    keys = batch[0].keys()
    out = {k: [] for k in keys}

    max_memory_depth = 0

    # Tokenize inputs
    for sample in batch:
        max_memory_depth = max(max_memory_depth, len(sample['inp_tags']))
        for k in keys:
            toks = sample[k]
            # is_pretokenized: sentence already split into list of strings
            ids = tokenizer.encode(toks, is_pretokenized=True).ids
            out[k].append(ids)

    # Pad inputs and stack
    for k in keys:
        # pad=(padding on left, padding on right)
        tensr = torch.stack([F.pad(torch.tensor(ids),
                                   pad=(0, max_memory_depth - len(ids)),
                                   value=tokenizer.token_to_id(pad_token))
                             for ids in out[k]])
        out[k] = tensr
    return out


def build_nl_tokenizer_dataloader(
        raw_datasets: List[  # multiple datasets
            List[Dict[str, List[str]]]  # a single dataset, eg [{inputs:..., outputs:...}]
        ],
        unk_token='[UNK]',
        pad_token='[PAD]',
        data_keys: List[str] = None,
):
    '''Dataset has pretokenization applied, ie it's split into a list of str
    tokens (it has not been tokenized into ints yet).

    This helper is highly specific to toy problems, where we eliminate
    tokenization issues by pretokenizing, IE, for an input sequence of
    "abc|cba", we need to have already pretokenized it into:

    ['a', 'b', 'c', '|', 'c', 'b', 'a']

    Returns:
      DataLoader creator fn, that accepts a batch_size

    '''
    for data in raw_datasets:
        assert isinstance(data, list)
        assert isinstance(data[0], dict)
        assert 'addresses' in data[0]
        assert 'inp_tags' in data[0]
        assert 'inp_col1' in data[0]
        assert 'inp_col2' in data[0]
        assert 'out_tags' in data[0]
        assert 'out_col1' in data[0]
        assert 'out_col2' in data[0]


    # the keys to pay attention to from the dataset
    if data_keys is None:
        data_keys = list(raw_datasets[0][0].keys())  # first data split, first row
        data_keys = list(filter(lambda x: x not in {'accuracy_mask'}, data_keys))

    # Convert to HF Dataset
    hf_datasets = [Dataset.from_list(x) for x in raw_datasets]

    # Make Tokenizer from Data
    special_tokens = [unk_token, pad_token]
    # init tokenizer
    tokenizer = Tokenizer(tokenizers.models.WordLevel(vocab={},
                                                      unk_token=unk_token))
    print('training tokenizer')
    tokenizer.train_from_iterator(NLCommon.iterator(hf_datasets, data_keys, special_tokens))
    tokenizer.add_special_tokens([unk_token, pad_token])

    def create_dataloaders(batch_size: int) -> List[DataLoader]:
        dataloaders = [
            DataLoader(x, batch_size=batch_size,
                       collate_fn=partial(collate_fn, tokenizer, pad_token))
            for x in hf_datasets
        ]
        return dataloaders

    return tokenizer, create_dataloaders


##################################################
# Prep data

try:
    already_loaded
except:
    print('building data')
    train = generate_dataset(TRAIN_N_DATA, TRAIN_MIN_SEQ_LEN, TRAIN_MAX_SEQ_LEN, digits)
    test  = generate_dataset(TEST_N_DATA, TEST_MIN_SEQ_LEN, TEST_MAX_SEQ_LEN, digits)
    print('tokenizing')
    tokenizer, dataloader_fn = build_nl_tokenizer_dataloader([prep(train), prep(test)])
    already_loaded = True
    print('done building data')


# # @@@@@@@@@@
# # Check data
# from neurallambda.lab.common import print_grid
# train_dl, test_dl = dataloader_fn(3)
# for batch in train_dl:
#     for i in range(batch['addresses'].size(0)):
#         print()
#         print_grid([['addresses'] + batch['addresses'][i].tolist(),
#                     ['inp_tags'] + batch['inp_tags'][i].tolist(),
#                     ['inp_col1'] + batch['inp_col1'][i].tolist(),
#                     ['inp_col2'] + batch['inp_col2'][i].tolist(),
#                     ['out_tags'] + batch['inp_tags'][i].tolist(),
#                     ['out_col1'] + batch['inp_col1'][i].tolist(),
#                     ['out_col2'] + batch['inp_col2'][i].tolist(),
#                     ])
#     break
# # @@@@@@@@@@


##################################################
# run_epoch

def run_epoch(model, dl, optimizer, mode, device, clip=None,
              check_accuracy=False, loss_fn='cross_entropy', regularization_fn=None):
    ''' Run an epoch over a DataLoader, and optionally perform greedy sampling
    to check accuracy.

    Args:
      if loss_fn == 'cosine_distance', model must return probabilities (after sigmoid activation)
      if loss_fn == 'cross_entropy', model must return unnormalized logits (ie final output comes from Linear layer)
      if loss_fn == 'nllloss', model must return logprobs, ie `F.log_softmax(..., dim=-1)`
    '''
    assert mode in ['eval', 'train'], "mode must be either 'eval' or 'train'"
    assert loss_fn in ['cosine_distance', 'nllloss', 'cross_entropy'], "loss_fn must be 'cosine_distance', 'nllloss', or 'cross_entropy'"

    if mode == 'train':
        model.train()
    else:
        model.eval()

    epoch_loss = 0

    if check_accuracy:
        outputs = []
        total_correct = 0
        total_samples = 0

        # for collecting scores
        y_true_tags = []
        y_pred_tags = []
        y_true_col1 = []
        y_pred_col1 = []
        y_true_col2 = []
        y_pred_col2 = []

    with torch.set_grad_enabled(mode == 'train'):
        for i, batch in enumerate(dl):
            addresses_ids = batch['addresses'].to(device)
            inp_tags_ids = batch['inp_tags'].to(device)
            inp_col1_ids = batch['inp_col1'].to(device)
            inp_col2_ids = batch['inp_col2'].to(device)
            trg_tags_ids = batch['out_tags'].to(device)
            trg_col1_ids = batch['out_col1'].to(device)
            trg_col2_ids = batch['out_col2'].to(device)

            out_tags, out_col1, out_col2 = model(addresses_ids, inp_tags_ids, inp_col1_ids, inp_col2_ids)  # [batch, seq, emb]

            if loss_fn == 'cosine_distance':
                trg_tags = model.embeddings(trg_tags_ids)
                trg_col1 = model.embeddings(trg_col1_ids)
                trg_col2 = model.embeddings(trg_col2_ids)

                # trgs = torch.cat([trg_tags, trg_col1, trg_col2], dim=1) # cat along S
                # outs = torch.cat([out_tags, out_col1, out_col2], dim=1) # cat along S

                # # trgs = torch.cat([trg_col1], dim=1) # cat along S
                # # outs = torch.cat([out_col1], dim=1) # cat along S


                # sim = torch.cosine_similarity(
                #     outs.flatten(0, 1),
                #     trgs.flatten(0, 1),
                #     dim=1)
                # loss = (1 - sim).mean()
                # # loss = (sim).mean()

                loss = 0
                for t, o in zip([trg_tags, trg_col1, trg_col2],
                                [out_tags, out_col1, out_col2]):
                    # t: [B, S, D]
                    # o: [B, S, D]
                    assert o.shape[0] == t.shape[0]
                    assert o.shape[1] == t.shape[1]
                    assert o.shape[2] == t.shape[2]
                    loss += (1 - torch.cosine_similarity(o, t, dim=2).mean())

            elif loss_fn == 'cross_entropy':
                loss = 0
                for t, o in zip([trg_tags_ids, trg_col1_ids, trg_col2_ids],
                                [out_tags, out_col1, out_col2]):
                    # t: [B, S]
                    # o: [B, S, Vocab]
                    assert o.shape[0] == t.shape[0]
                    assert o.shape[1] == t.shape[1]
                    assert o.shape[2] == model.embeddings.weight.shape[0]
                    loss += F.cross_entropy(o.flatten(0, 1),
                                            t.flatten(),
                                            reduction='mean')


            elif loss_fn == 'nllloss':
                # trg_ids = torch.cat([trg_tags_ids, trg_col1_ids, trg_col2_ids], dim=1)
                # # log_probs = F.log_softmax(output, dim=-1)
                # loss = F.nll_loss(output.flatten(0, 1),
                #                   trg_ids.flatten(),
                #                   reduction='mean')
                pass

            # if regularization_fn is not None:
            #     loss += regularization_fn(model, output)

            epoch_loss += loss.item()

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            if check_accuracy:
                with torch.no_grad():
                    if loss_fn == 'cosine_distance':
                        # find the embedding closest to the output, consider
                        # that the output_id
                        out_tags_ids = torch.cosine_similarity(model.embeddings.weight.unsqueeze(0).unsqueeze(0),  # [1, 1, VOCAB, EMB_DIM]
                                                               out_tags.unsqueeze(2),  # [BATCH, TIME, 1, EMB_DIM]
                                                               dim=3).argmax(dim=2)
                        out_col1_ids = torch.cosine_similarity(model.embeddings.weight.unsqueeze(0).unsqueeze(0),  # [1, 1, VOCAB, EMB_DIM]
                                                               out_col1.unsqueeze(2),  # [BATCH, TIME, 1, EMB_DIM]
                                                               dim=3).argmax(dim=2)
                        out_col2_ids = torch.cosine_similarity(model.embeddings.weight.unsqueeze(0).unsqueeze(0),  # [1, 1, VOCAB, EMB_DIM]
                                                               out_col2.unsqueeze(2),  # [BATCH, TIME, 1, EMB_DIM]
                                                               dim=3).argmax(dim=2)
                    else:
                        out_tags_ids = out_tags.argmax(dim=2)
                        out_col1_ids = out_col1.argmax(dim=2)
                        out_col2_ids = out_col2.argmax(dim=2)

                    correct_tags = (out_tags_ids == trg_tags_ids).sum().item()
                    correct_col1 = (out_col1_ids == trg_col1_ids).sum().item()
                    correct_col2 = (out_col2_ids == trg_col2_ids).sum().item()

                    total_correct += correct_tags + correct_col1 + correct_col2
                    total_samples += out_tags_ids.numel() + out_col1_ids.numel() + out_col2_ids.numel()

                    y_true_tags.extend(trg_tags_ids.cpu().numpy().flatten().tolist())
                    y_pred_tags.extend(out_tags_ids.cpu().numpy().flatten().tolist())
                    y_true_col1.extend(trg_col1_ids.cpu().numpy().flatten().tolist())
                    y_pred_col1.extend(out_col1_ids.cpu().numpy().flatten().tolist())
                    y_true_col2.extend(trg_col2_ids.cpu().numpy().flatten().tolist())
                    y_pred_col2.extend(out_col2_ids.cpu().numpy().flatten().tolist())
                    outputs.append((
                        addresses_ids,
                        inp_tags_ids,
                        inp_col1_ids,
                        inp_col2_ids,
                        trg_tags_ids,
                        trg_col1_ids,
                        trg_col2_ids,
                        out_tags_ids,
                        out_col1_ids,
                        out_col2_ids,
                    ))

    if check_accuracy:
        accuracy = total_correct / total_samples
        # scores: (precision, recall, f1_score, support)
        f1_tags = precision_recall_fscore_support(y_true_tags, y_pred_tags, average='weighted', zero_division=torch.nan)
        f1_col1 = precision_recall_fscore_support(y_true_col1, y_pred_col1, average='weighted', zero_division=torch.nan)
        f1_col2 = precision_recall_fscore_support(y_true_col2, y_pred_col2, average='weighted', zero_division=torch.nan)

        return epoch_loss / len(dl), {
            'f1_tags': f1_tags,
            'f1_col1': f1_col1,
            'f1_col2': f1_col2,
            'accuracy': accuracy,
        }, outputs

    else:
        return epoch_loss / len(dl)



##################################################
# Baselines


def relative_positions(seq_len: int, device) -> torch.tensor:
    ''' for ALiBi '''
    x = torch.arange(seq_len, device=device)[None, :]
    y = torch.arange(seq_len, device=device)[:, None]
    return x - y


def alibi_slope(num_heads, device):
    ''' for ALiBi '''
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)], device=device)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

class FullLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(FullLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h, c = torch.zeros(batch_size, self.lstm_cell.hidden_size).to(x.device), torch.zeros(batch_size, self.lstm_cell.hidden_size).to(x.device)
        outputs = []

        for t in range(seq_len):
            h, c = self.lstm_cell(x[:, t, :], (h, c))
            outputs.append(h)

        output = torch.stack(outputs, dim=1)
        return self.layer_norm(output)

class ControlModel(nn.Module):
    def __init__(self, tokenizer, emb_dim, num_heads, num_layers, dim_feedforward, num_recurrence, attn_nonlin, dropout=0.1, model_type='transformer'):
        super(ControlModel, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.num_recurrence = num_recurrence
        self.attn_nonlin = attn_nonlin
        self.model_type = model_type
        assert model_type in {'transformer',
                              'abstractor_all', 'abstractor_first', 'abstractor_last',
                              'sorter'}

        self.embeddings = nn.Embedding(self.vocab_size, emb_dim)
        self.pos_encoding = Transformer.positional_encoding(emb_dim)

        if model_type == 'transformer':
            use_wq = True
            use_wk = True
            use_wv = True
            use_wout = True
            self.layers = nn.ModuleList([
                Transformer.DecoderLayer(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
                                         use_wq=use_wq, use_wk=use_wk, use_wv=use_wv, use_wout=use_wout)
                for _ in range(num_layers)])

        elif model_type in {'abstractor_all', 'abstractor_first', 'abstractor_last'}:
            MAX_SEQUENCE_LEN = 100
            self.symbol_embeddings = torch.randn(MAX_SEQUENCE_LEN, emb_dim)

            use_wq = True
            use_wk = True
            use_wv = True
            use_wout = True

            self.layers = nn.ModuleList([
                Transformer.DecoderLayer(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
                                         use_wq=use_wq, use_wk=use_wk, use_wv=use_wv, use_wout=use_wout)
                for _ in range(num_layers)])


        elif model_type == 'sorter':
            MAX_SEQUENCE_LEN = 100
            self.symbol_embeddings = torch.randn(MAX_SEQUENCE_LEN, emb_dim)

            use_wq = True
            use_wk = True
            use_wv = True
            use_wout = True

            self.lstm = nn.Sequential(
                FullLSTM(input_size=emb_dim, hidden_size=emb_dim, dropout=0.0),
                # FullLSTM(input_size=emb_dim, hidden_size=emb_dim, dropout=0.0)
            )

            self.layers = nn.ModuleList([
                Transformer.DecoderLayer(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
                                         use_wq=use_wq, use_wk=use_wk, use_wv=use_wv, use_wout=use_wout)
                for _ in range(num_layers)])

            self.swap = nn.Sequential(
                nn.Linear(emb_dim, 2, bias=False)
                # nn.Linear(emb_dim, 16),
                # nn.ReLU(),
                # nn.Linear(16, 2)
            )

        # self.fc_out = nn.Linear(emb_dim, self.vocab_size, bias=False)
        # self.fc_out = nn.Linear(emb_dim, emb_dim, bias=False)
        if LOSS_FN in {'cross_entropy', 'cross_entropy_select_from_inputs'}:
            self.fc_out = nn.Linear(emb_dim, self.vocab_size, bias=False)
        elif LOSS_FN == 'cosine_distance':
            self.fc_out = nn.Linear(emb_dim, emb_dim, bias=False)


    def forward(self, addresses_ids, inp_tags_ids, inp_col1_ids, inp_col2_ids):
        B, S = addresses_ids.shape
        device = addresses_ids.device

        addresses = self.embeddings(addresses_ids) * math.sqrt(self.emb_dim)
        inp_tags  = self.embeddings(inp_tags_ids) * math.sqrt(self.emb_dim)
        inp_col1  = self.embeddings(inp_col1_ids) * math.sqrt(self.emb_dim)
        inp_col2  = self.embeddings(inp_col2_ids) * math.sqrt(self.emb_dim)

        embs = torch.cat([inp_tags, inp_col1, inp_col2], dim=1)
        pos = self.pos_encoding[:S, :].to('cuda')
        pos3 = torch.cat([pos] * 3, dim=0)
        # pos3 = torch.cat([addresses] * 3, dim=1) * 1e-2
        xs = embs


        # ALiBi

        # alibi_bias_2 = -torch.arange(1, S + 1).to(device) * 0.2
        # alibi_bias_2 = alibi_bias_2.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, S, -1)
        # # alibi_bias_2 = alibi_bias_2.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, S)


        m = alibi_slope(self.num_heads, device)
        # m = 0.1
        alibi_bias = (m * relative_positions(S, device)).unsqueeze(0).expand(B, -1, -1, -1)  # [B, NUM_HEAD, S, S]
        alibi_bias = alibi_bias.repeat_interleave(3, dim=-2).repeat_interleave(3, dim=-1)

        # alibi_bias = alibi_bias + alibi_bias_2


        for i in range(self.num_recurrence):
            in_xs = xs  # save for use in sorter
            for j, layer in enumerate(self.layers):

                if self.model_type == 'transformer':
                    q = xs
                    k = xs
                    v = xs
                elif self.model_type == 'abstractor_all':
                    q = xs
                    k = xs
                    v = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)
                    v = torch.cat([v] * 3, dim=1).to(device)
                elif self.model_type == 'abstractor_first':
                    q = xs
                    k = xs
                    if j == 0:
                        v = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)
                        v = torch.cat([v] * 3, dim=1).to(device)
                    else:
                        v = xs
                elif self.model_type == 'abstractor_last':
                    q = xs
                    k = xs
                    if j == self.num_recurrence-1:
                        v = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)
                        v = torch.cat([v] * 3, dim=1).to(device)
                    else:
                        v = xs
                elif self.model_type == 'sorter':

                    # do lstm in the middle
                    if j == 1:
                        xs = self.lstm(xs)

                    q = xs
                    k = xs
                    if j == 0:
                    # if j == self.num_recurrence-1:
                        v = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)
                        v = torch.cat([v] * 3, dim=1).to(device)
                    else:
                        v = xs

                # if j == 0 and i == 0:  # only add at first layer, first recurrence
                if j == 0:  # add at first layer of each recurrence
                    q = q + pos3
                    k = k + pos3
                    v = v + pos3

                xs = layer(q, k, v, mask=None, attn_nonlin=self.attn_nonlin, alibi_bias=alibi_bias)


            # if self.model_type == 'sorter':
            if False:

                swap_ixs = self.swap(xs)

                tags_swap1 = torch.softmax(swap_ixs[:, 0:S, 0], dim=1)
                tags_swap2 = torch.softmax(swap_ixs[:, 0:S, 1], dim=1)
                col1_swap1 = torch.softmax(swap_ixs[:, S:2*S, 0], dim=1)
                col1_swap2 = torch.softmax(swap_ixs[:, S:2*S:, 1], dim=1)
                col2_swap1 = torch.softmax(swap_ixs[:, 2*S:, 0], dim=1)
                col2_swap2 = torch.softmax(swap_ixs[:, 2*S:, 1], dim=1)

                tags, col1, col2 = torch.chunk(in_xs, 3, dim=1)
                tags = swap(tags, tags_swap1, tags_swap2)
                col1 = swap(col1, col1_swap1, col1_swap2)
                col2 = swap(col2, col2_swap1, col2_swap2)

                xs = torch.cat([tags, col1, col2], dim=1)

        tags, col1, col2 = torch.chunk(xs, 3, dim=1)

        # return (
        #     self.fc_out(tags),
        #     self.fc_out(col1),
        #     self.fc_out(col2),
        # )


        ##########
        # outputs

        if LOSS_FN == 'cosine_distance':

            # tags, col1, col2 = torch.chunk(xs, 3, dim=1)

            # return (
            #     self.fc_out(tags),
            #     self.fc_out(col1),
            #     self.fc_out(col2),
            # )

            return (
                tags,
                col1,
                col2,
            )

        elif LOSS_FN == 'cross_entropy_select_from_inputs':


            # tags, col1, col2 = torch.chunk(xs, 3, dim=1)

            # Use model output to select from inputs, and then output the
            # corresponding vocabulary location.
            out_tags = einsum(
                'btd, vd -> btv',  # select vocab, to go through cross entropy
                einsum('bst, bsd -> btd',  # "softmax(QK)V"
                       einsum('bsd, btd -> bst', inp_tags, tags).softmax(dim=2),  # "softmax(QK)
                       inp_tags),
                self.embeddings.weight)

            out_col1 = einsum(
                'btd, vd -> btv',  # select vocab, to go through cross entropy
                einsum('bst, bsd -> btd',  # "softmax(QK)V"
                       einsum('bsd, btd -> bst', inp_col1, col1).softmax(dim=2),  # "softmax(QK)
                       inp_col1),
                self.embeddings.weight)

            out_col2 = einsum(
                'btd, vd -> btv',  # select vocab, to go through cross entropy
                einsum('bst, bsd -> btd',  # "softmax(QK)V"
                       einsum('bsd, btd -> bst', inp_col2, col2).softmax(dim=2),  # "softmax(QK)
                       inp_col2),
                self.embeddings.weight)

            return (out_tags, out_col1, out_col2)

        elif LOSS_FN == 'cross_entropy':
            out_tags = self.fc_out(tags)
            out_col1 = self.fc_out(col1)
            out_col2 = self.fc_out(col2)

            # # Use model output to select from inputs, and then output the
            # # corresponding vocabulary location.
            # out_tags = einsum(
            #     'btd, vd -> btv',  # select vocab, to go through cross entropy
            #     tags,
            #     self.embeddings.weight)

            # out_col1 = einsum(
            #     'btd, vd -> btv',  # select vocab, to go through cross entropy
            #     col1,
            #     self.embeddings.weight)

            # out_col2 = einsum(
            #     'btd, vd -> btv',  # select vocab, to go through cross entropy
            #     col2,
            #     self.embeddings.weight)

            return (out_tags, out_col1, out_col2)




##################################################
# Custom Model


def swap(x, swap1, swap2):
    ''' swap1 and swap2 are softmax vectors (think onehot) of rows of x that will
    be swapped. '''
    # Combine swap1 and swap2 into a single matrix
    P = torch.einsum('bx,by->bxy', swap1, swap2)
    P = P + P.transpose(1, 2)  # swap both directions
    # identity matrix to keep non-swapped data
    Id = torch.diag_embed(torch.ones_like(swap1) - swap1 - swap2)
    x_swapped = torch.einsum('bij,bjd->bid', P + Id, x)
    return x_swapped





##################################################


model = ControlModel(
    tokenizer=tokenizer,
    emb_dim=EMB_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dim_feedforward=256,
    num_recurrence=NUM_RECURRENCE,
    attn_nonlin='softmax',

    # model_type='transformer'
    # model_type='abstractor_first'
    # model_type='abstractor_last'
    # model_type='abstractor_all'
    model_type='sorter'
)


model.to(DEVICE)

# skip training embeddings
print('skipping training embeddings')
no_trains = [model.embeddings]
for no_train in no_trains:
    for p in no_train.parameters():
        p.requires_grad = False

params = [x for x in model.parameters() if x.requires_grad]
print_model_info(model)

optimizer = optim.Adam(params, lr=LR, weight_decay=0)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


# START_BLOCK_1
for epoch in range(NUM_EPOCHS):
    train_dl, val_dl = dataloader_fn(BATCH_SIZE)

    train_loss, t_log, _ = run_epoch(
        model, train_dl, optimizer, 'train', DEVICE, GRAD_CLIP,
        check_accuracy=True, loss_fn=LOSS_FN)
    tacc = t_log['accuracy']
    # scores: (precision, recall, f1_score, support)
    t_f1_tags = t_log['f1_tags'][2]
    t_f1_col1 = t_log['f1_col1'][2]
    t_f1_col2 = t_log['f1_col2'][2]


    train_losses.append(train_loss)
    train_accuracies.append(tacc)

    val_loss, v_log, _ = run_epoch(
        model, val_dl, None, 'eval', DEVICE,
        check_accuracy=True, loss_fn=LOSS_FN)
    vacc = v_log['accuracy']
    v_f1_tags = v_log['f1_tags'][2]
    v_f1_col1 = v_log['f1_col1'][2]
    v_f1_col2 = v_log['f1_col2'][2]
    val_losses.append(val_loss)
    val_accuracies.append(vacc)

    print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.9f} | Train Acc: {tacc:.3f} | Val Acc: {vacc:.3f} | TF1: {t_f1_tags:.2f}, {t_f1_col1:.2f}, {t_f1_col2:.2f} | VF1: {v_f1_tags:.2f}, {v_f1_col1:.2f}, {v_f1_col2:.2f}')

    # Log parameter histograms
    try:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
    except:
        print('couldnt log to tensorboard')
# END_BLOCK_1


# START_BLOCK_2
# DEBUG = True
print()
total_printed = 0
# for batch in val_dl:
for batch in train_dl:
    batch_size = batch['inp_tags'].size(0)
    addresses_ids = batch['addresses'].to(DEVICE)
    inp_tags_ids = batch['inp_tags'].to(DEVICE)
    inp_col1_ids = batch['inp_col1'].to(DEVICE)
    inp_col2_ids = batch['inp_col2'].to(DEVICE)
    trg_tags_ids = batch['out_tags'].to(DEVICE)
    trg_col1_ids = batch['out_col1'].to(DEVICE)
    trg_col2_ids = batch['out_col2'].to(DEVICE)

    out_tags, out_col1, out_col2 = model(addresses_ids, inp_tags_ids, inp_col1_ids, inp_col2_ids)

    if LOSS_FN == 'cross_entropy':
        out_tags_vocab, out_col1_vocab, out_col2_vocab = out_tags, out_col1, out_col2
    elif LOSS_FN == 'cosine_distance':
        out_tags_vocab = torch.cosine_similarity(model.embeddings.weight.unsqueeze(0).unsqueeze(0),  # [1, 1, VOCAB, EMB_DIM]
                                               out_tags.unsqueeze(2),  # [BATCH, TIME, 1, EMB_DIM]
                                               dim=3)
        out_col1_vocab = torch.cosine_similarity(model.embeddings.weight.unsqueeze(0).unsqueeze(0),  # [1, 1, VOCAB, EMB_DIM]
                                               out_col1.unsqueeze(2),  # [BATCH, TIME, 1, EMB_DIM]
                                               dim=3)
        out_col2_vocab = torch.cosine_similarity(model.embeddings.weight.unsqueeze(0).unsqueeze(0),  # [1, 1, VOCAB, EMB_DIM]
                                               out_col2.unsqueeze(2),  # [BATCH, TIME, 1, EMB_DIM]
                                               dim=3)

    for i in range(batch_size):
        print('----------')
        inp_tags = [tokenizer.decode([x], skip_special_tokens=False) for x in inp_tags_ids[i].tolist()]
        inp_col1 = [tokenizer.decode([x], skip_special_tokens=False) for x in inp_col1_ids[i].tolist()]
        inp_col2 = [tokenizer.decode([x], skip_special_tokens=False) for x in inp_col2_ids[i].tolist()]
        trg_tags = [tokenizer.decode([x], skip_special_tokens=False) for x in trg_tags_ids[i].tolist()]
        trg_col1 = [tokenizer.decode([x], skip_special_tokens=False) for x in trg_col1_ids[i].tolist()]
        trg_col2 = [tokenizer.decode([x], skip_special_tokens=False) for x in trg_col2_ids[i].tolist()]
        out_tags = [tokenizer.decode([x], skip_special_tokens=False) for x in out_tags_vocab[i].argmax(dim=1).tolist()]
        out_col1 = [tokenizer.decode([x], skip_special_tokens=False) for x in out_col1_vocab[i].argmax(dim=1).tolist()]
        out_col2 = [tokenizer.decode([x], skip_special_tokens=False) for x in out_col2_vocab[i].argmax(dim=1).tolist()]

        print_grid([['inp_tags'] + inp_tags,
                    ['inp_col1'] + inp_col1,
                    ['inp_col2'] + inp_col2,

                    ['trg_tags'] + trg_tags,
                    ['trg_col1'] + trg_col1,
                    ['trg_col2'] + trg_col2,

                    ['out_tags'] + out_tags,
                    ['out_col1'] + out_col1,
                    ['out_col2'] + out_col2,

                    ])

        total_printed += 1
        if total_printed > 8:
            break
    if total_printed > 8:
        break
# END_BLOCK_2
