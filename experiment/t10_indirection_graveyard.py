'''

I should delete this

'''






##################################################
##################################################
##################################################
##################################################
#
# GRAVEYARD
#

if False:


    ##################################################
    # Custom Model

    class Model(nn.Module):
        def __init__(self, tokenizer, emb_dim, num_heads, dropout=0.1):
            super(Model, self).__init__()
            self.tokenizer = tokenizer
            self.vocab_size = tokenizer.get_vocab_size()
            self.emb_dim = emb_dim
            self.num_heads = num_heads

            self.embeddings = nn.Embedding(self.vocab_size, emb_dim)

            self.leaf_proj_dim = 256

            # H = 64
            # self.project_leaf1 = nn.Linear(emb_dim, self.leaf_proj_dim * 3, bias=False)
            # self.project_leaf2 = nn.Linear(emb_dim, self.leaf_proj_dim * 3, bias=False)

            H = 64
            self.project_leaf1 = nn.Sequential(nn.Linear(emb_dim, H, bias=True), nn.ReLU(), nn.Linear(H, self.leaf_proj_dim * 3, bias=False))
            self.project_leaf2 = nn.Sequential(nn.Linear(emb_dim, H, bias=True), nn.ReLU(), nn.Linear(H, self.leaf_proj_dim * 3, bias=False))

            self.norm = nn.LayerNorm(emb_dim)
            self.fc_out = nn.Linear(emb_dim, self.vocab_size, bias=False)

        def forward(self, addresses_ids, inp_tags_ids, inp_col1_ids, inp_col2_ids):
            B, S = addresses_ids.shape
            device = addresses_ids.device
            D = self.emb_dim

            addresses = self.embeddings(addresses_ids) * math.sqrt(D)  # [B, S, D]
            inp_tags  = self.embeddings(inp_tags_ids) * math.sqrt(D)   # [B, S, D]
            inp_col1  = self.embeddings(inp_col1_ids) * math.sqrt(D)   # [B, S, D]
            inp_col2  = self.embeddings(inp_col2_ids) * math.sqrt(D)   # [B, S, D]

            int_tag = self.embeddings(torch.tensor(tokenizer.encode('IntLit').ids, device=device)).squeeze()
            cons_tag = self.embeddings(torch.tensor(tokenizer.encode('Cons').ids, device=device)).squeeze()


            for _ in range(NUM_RECURRENCE):
                intlit_gate = torch.cosine_similarity(
                    inp_tags,  # [B, S, D]
                    int_tag.unsqueeze(0).unsqueeze(0),  # [1, 1, D]
                    dim=2).unsqueeze(2)  # [B, S, 1]

                # intlit_gate = 1

                leaf1 = self.project_leaf1(inp_col1.view(B * S, D)).view(B, S, self.leaf_proj_dim * 3) * intlit_gate
                leaf2 = self.project_leaf2(inp_col2.view(B * S, D)).view(B, S, self.leaf_proj_dim * 3) * intlit_gate

                # rollup = einsum('btl, bsl -> bts', leaf1, leaf2)

                cons_gate = torch.cosine_similarity(
                    inp_tags,  # [B, S, D]
                    cons_tag.unsqueeze(0).unsqueeze(0),  # [1, 1, D]
                    dim=2).unsqueeze(2)  # [B, S, 1]

                # cons_gate = 1

                # propogate leaf info up to parent node
                address_routing1 = einsum('btd, bsd -> bts', inp_col1 * cons_gate, addresses)

                # address_lift1 = einsum('bts, bsd -> btd', address_routing1, inp_col1 * cons_gate)
                # address_routing1 = einsum('bts, bsd -> bts', address_routing1, address_lift1)

                rollup1 = einsum('bts, bsl -> btl', address_routing1, leaf1)

                # propogate leaf info up to parent node
                address_routing2 = einsum('btd, bsd -> bts', inp_col2 * cons_gate, addresses)

                # address_lift2 = einsum('bts, bsd -> btd', address_routing2, inp_col2 * cons_gate)
                # address_routing2 = einsum('bts, bsd -> bts', address_routing2, address_lift2)

                rollup2 = einsum('bts, bsl -> btl', address_routing2, leaf2)

                # parents with children rolled up
                rollup = rollup1 + rollup2

                tags_r, col1_r, col2_r = torch.chunk(rollup, 3, dim=2)

                tags_sqk = einsum('btl, bsl -> bts', tags_r, tags_r).softmax(dim=2)
                col1_sqk = einsum('btl, bsl -> bts', col1_r, col1_r).softmax(dim=2)
                col2_sqk = einsum('btl, bsl -> bts', col2_r, col2_r).softmax(dim=2)

                new_tags = einsum('bts, bsd -> btd', tags_sqk, inp_tags)
                new_col1 = einsum('bts, bsd -> btd', col1_sqk, inp_col1)
                new_col2 = einsum('bts, bsd -> btd', col2_sqk, inp_col2)

                # # roll down
                # new_tags = einsum('bts, btd -> bsd', address_routing1, new_tags)
                # new_col1 = einsum('bts, btd -> bsd', address_routing1, new_col1)
                # new_col2 = einsum('bts, btd -> bsd', address_routing1, new_col2)

                inp_tags = new_tags
                inp_col1 = new_col1
                inp_col2 = new_col2

            # new_tags = self.fc_out(new_tags.view(B * S, D)).view(B, S, self.vocab_size)
            # new_col1 = self.fc_out(new_col1.view(B * S, D)).view(B, S, self.vocab_size)
            # new_col2 = self.fc_out(new_col2.view(B * S, D)).view(B, S, self.vocab_size)

            # new_tags = self.fc_out(self.norm(new_tags.view(B * S, D))).view(B, S, self.vocab_size)
            # new_col1 = self.fc_out(self.norm(new_col1.view(B * S, D))).view(B, S, self.vocab_size)
            # new_col2 = self.fc_out(self.norm(new_col2.view(B * S, D))).view(B, S, self.vocab_size)

            new_tags = einsum('vd, bsd -> bsv', F.normalize(self.embeddings.weight, dim=1), F.normalize(new_tags, dim=2))
            new_col1 = einsum('vd, bsd -> bsv', F.normalize(self.embeddings.weight, dim=1), F.normalize(new_col1, dim=2))
            new_col2 = einsum('vd, bsd -> bsv', F.normalize(self.embeddings.weight, dim=1), F.normalize(new_col2, dim=2))

            return (
                new_tags,
                new_col1,
                new_col2,
            )


    ##################################################
    # Relation Net Version

    class RelationNetwork(nn.Module):
        def __init__(self, mlp1, mlp2, aggregation_fn=torch.sum):
            """
            Initializes the Relation Network module.

            Args:
                mlp1 (nn.Module): The first MLP to process object pairs.
                mlp2 (nn.Module): The second MLP to process the aggregated relation.
                aggregation_fn (function): The aggregation function to combine pairwise relations.
            """
            super(RelationNetwork, self).__init__()
            self.mlp1 = mlp1
            self.mlp2 = mlp2
            self.aggregation_fn = aggregation_fn

        def forward(self, x):
            """
            Forward pass of the Relation Network.

            Args:
                x (Tensor): Input tensor of shape (batch_size, num_objects, object_dim).

            Returns:
                Tensor: Output tensor of the Relation Network.
            """
            batch_size, num_objects, object_dim = x.size()

            # Create all pairs (o_i, o_j)
            pairs = torch.cat([x.unsqueeze(2).repeat(1, 1, num_objects, 1),
                               x.unsqueeze(1).repeat(1, num_objects, 1, 1)], dim=-1)

            # Reshape pairs to (batch_size * num_objects * num_objects, 2 * object_dim)
            pairs = pairs.view(batch_size * num_objects * num_objects, 2 * object_dim)

            # Pass pairs through the first MLP
            relations = self.mlp1(pairs)

            # Reshape back to (batch_size, num_objects, num_objects, relation_dim)
            relation_dim = relations.size(-1)
            relations = relations.view(batch_size, num_objects, num_objects, relation_dim)

            relations = torch.softmax(relations.sum(dim=3), dim=2)  # TODO: not even sure if this is a good idea, nor if the softmax is on the right dim
            output = einsum('bnm, bno -> bmo', relations, x)

            # # Aggregate relations
            # aggregated_relations = self.aggregation_fn(relations, dim=(1, 2))

            # # Pass the aggregated relations through the second MLP
            # output = self.mlp2(aggregated_relations)

            return output




    # START_BLOCK_3



    class AbstractorModel2(nn.Module):
        def __init__(self, tokenizer, emb_dim, num_heads, num_layers, dim_feedforward, num_recurrence, attn_nonlin, dropout=0.1):
            super(AbstractorModel2, self).__init__()
            self.tokenizer = tokenizer
            self.vocab_size = tokenizer.get_vocab_size()
            self.emb_dim = emb_dim
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.dim_feedforward = dim_feedforward
            self.num_recurrence = num_recurrence
            self.attn_nonlin = attn_nonlin

            self.embeddings = nn.Embedding(self.vocab_size, emb_dim)
            # MAX_SEQUENCE_LEN = 100
            # self.symbol_embeddings = torch.randn(MAX_SEQUENCE_LEN, emb_dim)

            self.pos_encoding = Transformer.positional_encoding(emb_dim)
            self.norm = nn.LayerNorm(emb_dim)

            self.presym = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                )

            self.sym = Flexformer(
                emb_dim,
                hidden_size=dim_feedforward,
                out_size=2,
                num_heads=num_heads,
                    use_symbols=True)

            self.layers = nn.ModuleList([Flexformer(
                emb_dim,
                hidden_size=dim_feedforward,
                out_size=emb_dim,
                num_heads=num_heads,
                use_symbols=False)
                for _ in range(num_layers)])

            if LOSS_FN in {'cross_entropy', 'cross_entropy_select_from_inputs'}:
                self.fc_out = nn.Linear(emb_dim, self.vocab_size, bias=False)
            elif LOSS_FN == 'cosine_distance':
                self.fc_out = nn.Linear(emb_dim, emb_dim, bias=False)
            self.dropout = nn.Dropout(dropout)

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
            hs = embs

            # mask = einsum('bsd, btd -> bst', inp_col1, inp_col2)
            # mask = mask.repeat(1, 3, 3)

            # v = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)
            # v = torch.cat([v] * 3, dim=1).to(device)

            for i in range(self.num_recurrence):
                in_hs = hs  # preserve the start-of-loop hs
                for j, layer in enumerate(self.layers):
                    if j == 0:
                        hs = hs + pos3

                    hs = layer(hs,
                               attn_nonlin=self.attn_nonlin,
                               use_wq=True,
                               use_wk=True,
                               use_wv=True,)

                # # hs = self.presym(hs)
                # swap_ixs = self.sym(hs,
                #                     attn_nonlin='none',
                #                     use_wq=False,
                #                     use_wk=False,
                #                     use_wv=False,)

                # tags_swap1 = torch.softmax(swap_ixs[:, 0:S, 0], dim=1)
                # tags_swap2 = torch.softmax(swap_ixs[:, 0:S, 1], dim=1)
                # col1_swap1 = torch.softmax(swap_ixs[:, S:2*S, 0], dim=1)
                # col1_swap2 = torch.softmax(swap_ixs[:, S:2*S:, 1], dim=1)
                # col2_swap1 = torch.softmax(swap_ixs[:, 2*S:, 0], dim=1)
                # col2_swap2 = torch.softmax(swap_ixs[:, 2*S:, 1], dim=1)

                # tags, col1, col2 = torch.chunk(in_hs, 3, dim=1)
                # tags = swap(tags, tags_swap1, tags_swap2)
                # col1 = swap(col1, col1_swap1, col1_swap2)
                # col2 = swap(col2, col2_swap1, col2_swap2)

                # hs = torch.cat([tags, col1, col2], dim=1)

                if DEBUG:
                    breakpoint()


                tags, col1, col2 = torch.chunk(in_hs, 3, dim=1)

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










class Flexformer(nn.Module):

    '''TODO:

    - forward gets separate qkv
    - decoder block: attn + ffnn
    - norms


    '''
    def __init__(self, embedding_size, hidden_size, out_size, use_symbols, num_heads=2):
        super(Flexformer, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_heads = num_heads
        self.use_symbols = use_symbols

        # Dimensions for multi-head attention
        self.head_dim = embedding_size // num_heads
        assert embedding_size % num_heads == 0, "embedding_size must be divisible by num_heads"

        # Linear layers for query, key, value
        self.query_linear = nn.Linear(embedding_size, embedding_size, bias=True)
        self.key_linear = nn.Linear(embedding_size, embedding_size, bias=True)

        self.V_DIM = embedding_size

        if not use_symbols:
            self.value_linear = nn.Linear(embedding_size, embedding_size, bias=True)

        else:
            # Learnable parameters for value (V) vectors in cross-attention
            MAX_VOCAB = 100
            self.value_embeddings = nn.Parameter(torch.randn(MAX_VOCAB, self.V_DIM))
            # self.value_embeddings = torch.randn(MAX_VOCAB, self.V_DIM)

        # # Output linear layer
        # self.out = nn.Linear(self.V_DIM, self.out_size)

        self.out = nn.Sequential(
            nn.Linear(self.V_DIM, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_size)
        )

    def scaled_dot_product_attention(self, Q, K, V, attn_nonlin):
        scores = torch.einsum('bhsd, bhtd -> bhst', Q, K) / (self.head_dim ** 0.5)

        # # remove self-similarity
        # B, NH, S, HD = Q.size()
        # Id = torch.eye(S, S).unsqueeze(0).unsqueeze(0).expand(B, NH, S, S).to(Q.device)
        # scores = scores * (1 - Id)
        # # breakpoint()

        match attn_nonlin:
            case 'softmax':
                attn_weights = torch.softmax(scores, dim=3)  # [B, HEAD, S, S]
            case 'none':
                attn_weights = scores
            case 'sigmoid':
                attn_weights = scores.sigmoid()
            case 'tanh':
                attn_weights = scores.tanh()

        attn_output = torch.einsum('bhst, bhtd -> bhsd', attn_weights, V)

        # if DEBUG:
        #     breakpoint()

        return attn_output

    def forward(self, x, attn_nonlin, use_wq, use_wk, use_wv):
        B, S, D = x.size()

        Q = self.query_linear(x) if use_wq else x
        K = self.key_linear(x) if use_wk else x

        # Reshape Q, K, V for multi-head attention
        Q = torch.einsum('bshd -> bhsd', Q.view(B, S, self.num_heads, self.head_dim))
        K = torch.einsum('bshd -> bhsd', K.view(B, S, self.num_heads, self.head_dim))

        # Expand the value embeddings to match the batch size and apply linear projection
        if self.use_symbols:
            self.value_embeddings = self.value_embeddings.to(x.device)
            V = self.value_embeddings[:S].unsqueeze(0).expand(B, -1, -1)  # (B, S, D)
        else:
            V = self.value_linear(x)
            # V = torch.einsum('bshd -> bhsd', K.view(B, S, self.num_heads, self.head_dim))
        V = torch.einsum('bshd -> bhsd', V.view(B, S, self.num_heads, self.V_DIM // self.num_heads))

        attn_output = self.scaled_dot_product_attention(Q, K, V, attn_nonlin)  # (B, num_heads, S, head_dim)

        # Concatenate heads
        attn_output = torch.einsum('bhsd -> bsdh', attn_output).contiguous().view(B, S, self.V_DIM)  # (B, S, V_DIM)
        o = self.out(attn_output)
        return o



##################################################
# Permuting a single item


def permute_backward(xs, from_prob, to_prob):
    '''insert the item in a sequence at `from_prob` to `to_prob`.

    the index in from_prob is expected to be later in the sequence than the
    destination to_prob, IE the selected element is moving backwards

    args:
      xs: [batch, sequence, embedding dim]
      from_prob, to_prob: [batch, sequence]

    '''
    B, S = xs.size(0), xs.size(1)
    I = torch.eye(S).unsqueeze(0).repeat(B, 1, 1)

    fcs2 = torch.cumsum(from_prob, dim=1)
    fcs1 = ((fcs2 - 1) * -1)

    tcs2 = torch.cumsum(to_prob, dim=1)
    tcs1 = ((tcs2 - 1) * -1)

    bwd_perm = (
        torch.einsum('bst, bs -> bst', I, fcs1) +  # left block
        F.pad(torch.einsum('bst, bs, bt -> bst',
                           I,
                           fcs2,
                           tcs1), [0, 0, 1, 0], value=0)[:, :-1] +  # mid block. shift I down
        torch.einsum('bs, bt -> bst', from_prob, to_prob) +  # inserted value
        torch.einsum('bst, bs -> bst', I, F.pad(tcs2, [1, 0], value=0)[:, :S])  # right block
    )

    return torch.einsum('bst, btd -> bsd', bwd_perm, xs), bwd_perm


def permute_forward(xs, from_prob, to_prob):
    '''insert the item in a sequence at `from_prob` to `to_prob`.

    the index in from_prob is expected to be earlier in the sequence than the
    destination to_prob, IE the selected element is moving forwards.

    args:
      xs: [batch, sequence, embedding dim]
      from_prob, to_prob: [batch, sequence]

    '''
    B, S = xs.size(0), xs.size(1)
    I = torch.eye(S).unsqueeze(0).repeat(B, 1, 1)

    fcs2 = torch.cumsum(from_prob, dim=1)
    fcs1 = (fcs2 - torch.ones_like(fcs2)) * -1

    tcs2 = torch.cumsum(to_prob, dim=1)
    tcs1 = (tcs2 - torch.ones_like(tcs2)) * -1

    fwd_perm = (
        torch.einsum('bst, bs -> bst', I, fcs1) +  # left block
        torch.einsum('bs, bt -> bst', to_prob, from_prob) +  # inserted value
        F.pad(torch.einsum('bst, bs, bt -> bst',
                           I,
                           F.pad(fcs2, [1, 0], value=0)[:, :S],
                           F.pad(tcs1, [1, 0], value=0)[:, :S]), [0, 0, 0, 1])[:, 1:] +  # mid block, shift I up and to the right
        torch.einsum('bst, bs -> bst', I, F.pad(tcs2, [1, 0], value=0)[:, :S])  # right block
    )
    return torch.einsum('bst, btd -> bsd', fwd_perm, xs), fwd_perm


# print()
# print('forward')
# f, fp = permute_forward(xs, from_prob, to_prob)
# print(f)
# print(fp)


# B = 2
# S = 10
# D = 3

# x = torch.arange(B * S, dtype=torch.float).reshape(B, S).unsqueeze(2).repeat(1, 1, D)
# xr = torch.roll(x, shifts=(1,), dims=(1,))
# xl = torch.roll(x, shifts=(-1,), dims=(1,))

# I = torch.eye(S).unsqueeze(0).repeat(B, 1, 1)
# Ir = torch.roll(I, shifts=(1,), dims=(1,))
# Il = torch.roll(I, shifts=(-1,), dims=(1,))

# from_prob = torch.zeros((B, S)).float()
# from_prob[:, 0] = 1.0

# to_prob = torch.zeros((B, S)).float()
# to_prob[:, 9] = 1.0


# # ar = torch.arange(S).unsqueeze(0).repeat(B, 1)
# # cs1 = ar - torch.cumsum(1 - from_prob, dim=1)
# fcs2 = torch.cumsum(from_prob, dim=1)
# fcs1 = (fcs2 - torch.ones_like(fcs2)) * -1
# fcs = torch.einsum('bi, bj -> bij', fcs1, fcs2)

# tcs2 = torch.cumsum(to_prob, dim=1)
# tcs1 = (tcs2 - torch.ones_like(tcs2)) * -1
# tcs = torch.einsum('bi, bj -> bij', tcs1, tcs2)


# # bwd
# print('backward')
# bwd_perm = (
#     torch.einsum('bst, bs -> bst', I, fcs1) +  # left block
#     F.pad(torch.einsum('bst, bs, bt -> bst',
#                        I,
#                        fcs2,
#                        tcs1), [0, 0, 1, 0], value=0)[:, :-1] +  # mid block. shift I down
#     torch.einsum('bs, bt -> bst', from_prob, to_prob) +  # inserted value
#     torch.einsum('bst, bs -> bst', I, F.pad(tcs2, [1, 0], value=0)[:, :S])  # right block
# )
# print(bwd_perm)
# print(torch.einsum('bst, btd -> bsd', bwd_perm, x))



# # fwd
# print('forward')
# fwd_perm = (
#     torch.einsum('bst, bs -> bst', I, fcs1) +  # left block
#     torch.einsum('bs, bt -> bst', to_prob, from_prob) +  # inserted value
#     F.pad(torch.einsum('bst, bs, bt -> bst',
#                        I,
#                        F.pad(fcs2, [1, 0], value=0)[:, :S],
#                        F.pad(tcs1, [1, 0], value=0)[:, :S]), [0, 0, 0, 1])[:, 1:] +  # mid block, shift I up and to the right
#     torch.einsum('bst, bs -> bst', I, F.pad(tcs2, [1, 0], value=0)[:, :S])  # right block
# )
# print(fwd_perm)
# print(torch.einsum('bst, btd -> bsd', fwd_perm, x))



# END_BLOCK_3





# START_BLOCK_4

# B = 2
# S = 10
# D = 3

# I = torch.eye(S).unsqueeze(0).repeat(B, 1, 1)


# xs = torch.arange(B * S, dtype=torch.float).reshape(B, S).unsqueeze(2).repeat(1, 1, D)

# fp = torch.zeros((B, S)).float()
# fp[:, 9] = 1.0


# tp = torch.zeros((B, S)).float()
# tp[:, 0] = 1.0


# fcs2 = torch.cumsum(tp, dim=1)
# fcs1 = ((fcs2 - 1) * -1)

# tcs2 = torch.cumsum(fp, dim=1)
# tcs1 = ((tcs2 - 1) * -1)

# # lb = torch.einsum('bst, bs -> bst', I, fcs1)
# lb = torch.diag_embed(fcs1 * tcs1)
# # mb = F.pad(torch.einsum('bst, bs, bt -> bst',
# #                        I,
# #                        fcs2,
# #                        tcs1), [0, 0, 1, 0], value=0)[:, :-1] # mid block. shift I down

# mb = (
#     # mid block, shift down for case where moving ix backward. If it's actually
#     # moving ix forward, this should become all 0s.
#     F.pad(torch.diag_embed(fcs2 * tcs1), [0, 0, 1, 0], value=0)[:, :-1] +

#     # mid block, shift I up and to the right for case where moving ix
#     # forward. If it's actually moving ix backward, this should become all 0s.
#     F.pad(torch.diag_embed(F.pad(fcs1, [1, 0], value=0)[:, :S] *
#                            F.pad(tcs2, [1, 0], value=0)[:, :S]), [0, 0, 0, 1])[:, 1:]

# )
# val = torch.einsum('bs, bt -> bst', tp, fp)
# # rb = torch.einsum('bst, bs -> bst', I, F.pad(tcs2, [1, 0], value=0)[:, :S])  # right block
# rb = torch.diag_embed(F.pad(tcs2, [1, 0], value=0)[:, :S] *
#                       F.pad(fcs2, [1, 0], value=0)[:, :S])
# bwd_perm = (lb + mb + val + rb)

# out = torch.einsum('bst, btd -> bsd', bwd_perm, xs)


# # clean up -0
# lb[torch.logical_and(lb <= 0, lb > -1e-6)] = 0
# mb[torch.logical_and(mb <= 0, mb > -1e-6)] = 0
# val[torch.logical_and(val <= 0, val > -1e-6)] = 0
# rb[torch.logical_and(rb <= 0, rb > -1e-6)] = 0

# print('----------')
# print('lb'); print(lb[0])
# print('mb'); print(mb[0])
# print('val'); print(val[0])
# print('rb'); print(rb[0])
# print('out'); print(out[0,:,0])
# # END_BLOCK_4
