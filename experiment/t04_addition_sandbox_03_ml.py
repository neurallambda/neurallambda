'''

Just testing what works well for a subnet of the main ML model

'''

##################################################
# Adder

class Add(nn.Module):
    def __init__(self, vec_size, hidden_dim):
        super(Add, self).__init__()

        self.n_symbols = 200
        self.dropout_p = 0.05

        # INIT_W = 'orthogonal'
        INIT_W = 'kaiming'
        sym_w = Weight(vec_size, self.n_symbols, INIT_W)
        self.sym_w = sym_w

        # # preload some known symbols
        # with torch.no_grad():
        #     for ix, i in enumerate(tokens + list(range(-40, 40))):
        #         self.sym_w.weight[:, ix] = project(i)

        # cos sim for multiple stacked vecs
        CosSim = CosineSimilarity(
            sym_w,
            dim=2,
            unsqueeze_inputs=[-1],    # (batch, N, vec_size, _)  # control + input
            unsqueeze_weights=[0, 0], # (_,     _, vec_size, n_symbols)
        ) # (batch, N, n_symbols)


        # cos sim for a single vec
        CosSim_1 = CosineSimilarity(
            sym_w,
            dim=1,
            unsqueeze_inputs=[-1],    # (batch, vec_size, _)
            unsqueeze_weights=[0],    # (_,     vec_size, n_symbols)
        ) # (batch, n_symbols)

        self.ff = nn.Sequential(
            Stack(dim=1), # input = (control, work)
            CosSim,
            nn.Flatten(start_dim=1, end_dim=2), # (batch, 2 * n_symbols)
            nn.Linear(self.n_symbols * 2, hidden_dim, bias=False), #(batch, work_decision + work_value)
            nn.GELU(), # prevent negative sim
            # nn.ReLU(),

            # nn.Linear(vec_size * 2, hidden_dim, bias=False), #(batch, work_decision + work_value)
            # nn.Dropout(self.dropout_p),
            # nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim, bias=True), #(batch, work_decision + work_value)
            nn.Dropout(self.dropout_p),
            nn.ReLU(),


            # # A
            # nn.Linear(hidden_dim, vec_size),
            # nn.Tanh(),

            # B

            # nn.Linear(hidden_dim, vec_size, bias=True),
            # nn.Tanh(),


            nn.Linear(hidden_dim, self.n_symbols, bias=True),
            nn.Sigmoid(),
            ReverseCosineSimilarity(CosSim_1),

        )

        self.ff = nn.Sequential(
            # Stack(dim=1),
            Elementwise(lambda x, y: x * y),
            nn.Linear(vec_size, hidden_dim),
            nn.ReLU(),

            # nn.Linear(hidden_dim, vec_size),
            # nn.Tanh(),

            nn.Linear(hidden_dim, self.n_symbols, bias=True),
            nn.Sigmoid(),
            ReverseCosineSimilarity(CosSim_1),

        )


    def forward(self, inp):
        return self.ff(inp)


add_dataset = []
for i in range(-10, 10):
    # if i in {0}:
    #     continue
    for j in range(-10, 10):
        x = (project(i), project(j))
        # y = project(i + j)
        y = project((i * j) % 10)
        # y = project((i + j) % 10)
        add_dataset.append((x, y))
train_add_dl = DataLoader(add_dataset, batch_size=BATCH_SIZE, shuffle=True)
add_model = Add(VEC_SIZE, 64)
add_model.cuda()
opt_params = list(filter(lambda p: p.requires_grad, add_model.parameters()))
optimizer = optim.Adam(opt_params, lr=1e-2)
train_losses = []
for epoch in range(NUM_EPOCHS):
    add_model.train()
    epoch_loss = 0
    for _, (src, trg) in enumerate(train_add_dl):
        optimizer.zero_grad()
        output = add_model(src)
        loss = (1 - torch.cosine_similarity(output, trg, dim=1)).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_add_dl)
    print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.5f}')

n = 5
for i in range(n-10, n):
    for j in range(n-10, n):
        pi = project(i).unsqueeze(0)
        pj = project(j).unsqueeze(0)
        out = add_model([pi, pj])
        uout = unproject(out.squeeze())
        print(f'{i} + {j} = {uout}, {torch.cosine_similarity(out.squeeze(0), project(uout), dim=0):.4f}')

BRK


##################################################
# Trash heap

        ##########
        # Operational Symbols (like push/pop/nopping the stack)

        n_op_symbols = 8
        self.n_op_symbols = n_op_symbols

        # INIT_W = 'orthogonal'
        INIT_W = 'kaiming'
        op_sym_w = Weight(self.input_dim, n_op_symbols, INIT_W)
        self.op_sym_w = op_sym_w

        # cos sim for multiple stacked vecs
        OpCosSim = CosineSimilarity(
            op_sym_w,
            dim=2,
            unsqueeze_inputs=[-1],    # (batch, N, vec_size, _)  # control + input
            unsqueeze_weights=[0, 0], # (_,     _, vec_size, n_op_symbols)
        ) # (batch, N, n_op_symbols)

        # cos sim for a single vec
        OpCosSim_1 = CosineSimilarity(
            op_sym_w,
            dim=1,
            unsqueeze_inputs=[-1],    # (batch, vec_size, _)
            unsqueeze_weights=[0],    # (_,     vec_size, n_op_symbols)
        ) # (batch, n_op_symbols)


        ##########
        # Identifying Symbols

        # INIT_W = 'orthogonal'
        INIT_W = 'kaiming'
        sym_w = Weight(self.input_dim, self.n_symbols, INIT_W)
        self.sym_w = sym_w

        # preload some known symbols
        with torch.no_grad():
            for ix, i in enumerate(tokens + list(range(-40, 40))):
                self.sym_w.weight[:, ix] = project(i)

        # cos sim for multiple stacked vecs
        CosSim = CosineSimilarity(
            sym_w,
            dim=2,
            unsqueeze_inputs=[-1],    # (batch, N, vec_size, _)  # control + input
            unsqueeze_weights=[0, 0], # (_,     _, vec_size, n_symbols)
        ) # (batch, N, n_symbols)

        # cos sim for a single vec
        CosSim_1 = CosineSimilarity(
            sym_w,
            dim=1,
            unsqueeze_inputs=[-1],    # (batch, vec_size, _)
            unsqueeze_weights=[0],    # (_,     vec_size, n_symbols)
        ) # (batch, n_symbols)
