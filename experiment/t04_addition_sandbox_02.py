'''.

Here, I'm trying to experiment with toy non-ML problems in python to understand
what needs to go into the ML version.


THOUGHTS

* Per conditional branch, stacks can be read once, popped once, and pushed
  once. Same for queues.

'''

import re
import pandas as pd
from neurallambda.util import print_grid

##########
#

class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if len(self.stack) < 1:
            return None
        return self.stack.pop()

    def peek(self):
        if len(self.stack) < 1:
            return None
        return self.stack[-1]

class Queue:
    def __init__(self):
        self.queue = []

    def put(self, item):
        self.queue.append(item)

    def get(self):
        if len(self.queue) < 1:
            return None
        return self.queue.pop(0)

    def peek(self):
        if len(self.queue) < 1:
            return None
        return self.queue[0]


##########
# Read Data

pattern = re.compile(r'\(PSH +[^\^ ^)]+\)|[^ ]+')

def parse_cell(cell):
    xs = pattern.findall(cell)
    out = []
    for x in xs:
        try:
            out.append(int(x.strip()))
        except:
            out.append(x)
    return out

def read_csv(data_path):
    df = pd.read_csv(data_path, sep="|")
    for col in df.columns:
        df[col] = df[col].apply(parse_cell)
    return df

data_path_3 = "experiment/t04_addition/mod_sum_length_3.csv"
data_path_5 = "experiment/t04_addition/mod_sum_length_5.csv"
data_path_10 = "experiment/t04_addition/mod_sum_length_10.csv"
data_path_20 = "experiment/t04_addition/mod_sum_length_20.csv"

df = read_csv(data_path_3)

Input         = df['Input']
Output        = df['Output']
PreGlobalOp   = df['PreGlobalOp']
PreWorkOp     = df['PreWorkOp']
PostGlobalOp  = df['PostGlobalOp']
PostGlobalVal = df['PostGlobalVal']
PostWorkOp    = df['PostWorkOp']
PostWorkVal   = df['PostWorkVal']

# ex = [
#     (Input[0], Output[0], PreGlobalOp[0], PreWorkOp[0], PostGlobalOp[0], PostGlobalVal[0], PostWorkOp[0], PostWorkVal[0]),
#     (Input[1], Output[1], PreGlobalOp[1], PreWorkOp[1], PostGlobalOp[1], PostGlobalVal[1], PostWorkOp[1], PostWorkVal[1])
# ]
# labels = ('Input', 'Output', 'PreGlobalOp', 'PreWorkOp', 'PostGlobalOp', 'PostGlobalVal', 'PostWorkOp', 'PostWorkVal', )
# print_grid(ex, labels)


##########

#####
# PRE

def f_typ(inp):
    if isinstance(inp, int):
        return 'INT'
    else:
        return 'NINT'

def f_pre_c_op(c_peek, inp_typ, inp):
    match (c_peek, inp_typ, inp):
        # This is cool. The global control stack can say "return left", and we
        # can check if it did. If it didn't, don't pop the control. This is
        # reminiscent of biological feedback loops (ala active
        # inference?). Actually, the < > tags aren't from input, it'd need to
        # remember it's last output, and then this implication would be
        # relevant.
        case ('RL', _, _) | ('RS', _, _) | ('RR', _, _):
            return 'POP'
    return 'NOP'

def f_pre_w_op(c_peek, inp_typ, inp):
    match (c_peek, inp_typ, inp):
        case ('RS', _, _):
            return 'POP'
        case (_, 'INT', _):
            return 'POP'
    return 'NOP'


#####
# POST

def f_post_c_op(c_peek, inp_typ, inp):
    match (c_peek, inp_typ, inp):
        case (_, _, 'S') | (_, _, 'F'):
            return 'PSH'
        case ('RL', _, _) | ('RS', _, _):
            return 'PSH'
    return 'NOP'

def f_post_c_val(c_peek, inp_typ, inp):
    match (c_peek, inp_typ, inp):
        case (_, _, 'S'):
            return 'SS'
        case (_, _, 'F'):
            return 'RL'
        case ('RL', _, _):
            return 'RS'
        case ('RS', _, _):
            return 'RR'
    return '_'

def f_post_w_op(c_peek, inp_typ, inp):
    match (c_peek, inp_typ, inp):
        case ('SS', 'INT', _):
            return 'PSH'
    return 'NOP'

def f_post_w_val(c_peek, w_peek, w_peek_typ, inp_typ, inp):
    match (c_peek, w_peek, w_peek_typ, inp_typ, inp):
        case ('SS', _, 'INT', 'INT',  _):
            return (inp + w_peek) % 10
        case ('SS', _, 'NINT', _,  _):
            return inp
    return '_'


#####
# SELECT

def f_select(c_peek, w_peek):
    match (c_peek, w_peek):
        case ('RS', _):
            return w_peek
        case ('RL', _):
            return '<'
        case ('RR', _):
            return '>'
    return '_'


# inps = ['S',  1,   2,   3,  "F", "N", "N", "N", 'N']
# exps = ['N', "N", "N", "N", 'N', "L",  6,  "R", 'N']

inps = Input[0]
exps = Output[0]

work_stack = Stack()
control_stack = Stack()

outs = []
pre_c_ops = []
pre_w_ops = []
post_c_ops = []
post_c_vals = []
post_w_ops = []
post_w_vals = []

for inp, exp in zip(inps, exps):

    # inp_typ = inp
    inp_typ = f_typ(inp)
    c_peek = control_stack.peek()
    w_peek = work_stack.peek()

    pre_inp = [c_peek, inp_typ, inp]

    ##########
    # Pre

    # control stack maybe pop
    pre_c_op = f_pre_c_op(*pre_inp)
    if pre_c_op == 'POP':
        control_stack.pop()

    # work stack maybe pop
    pre_w_op = f_pre_w_op(*pre_inp)
    if pre_w_op == 'POP':
        work_stack.pop()


    ##########
    # Post

    # control stack
    post_c_inp = [c_peek, inp_typ, inp]
    post_c_op = f_post_c_op(*post_c_inp)
    post_c_val = f_post_c_val(*post_c_inp)
    if post_c_op == 'PSH':
        control_stack.push(post_c_val)

    # work stack
    w_peek_typ = f_typ(w_peek)
    post_w_op = f_post_w_op(c_peek, inp_typ, inp)
    post_w_val = f_post_w_val(c_peek, w_peek, w_peek_typ, inp_typ, inp)
    if post_w_op == 'PSH':
        work_stack.push(post_w_val)


    ##########
    # Select out
    out = f_select(c_peek, w_peek)

    outs.append(out)
    pre_c_ops.append(pre_c_op)
    pre_w_ops.append(pre_w_op)
    post_c_ops.append(post_c_op)
    post_c_vals.append(post_c_val)
    post_w_ops.append(post_w_op)
    post_w_vals.append(post_w_val)


grid = [
    (inps, exps, outs,
     pre_c_ops,
     pre_w_ops,
     post_c_ops,
     post_c_vals,
     post_w_ops,
     post_w_vals,
     ),
]
labels = (
    'Input', 'Target', 'Output',
    'pre_c_ops',
    'pre_w_ops',
    'post_c_ops',
    'post_c_vals',
    'post_w_ops',
    'post_w_vals',
)


print_grid(grid, labels)


##################################################

f_typ = [
    (0, 'INT'),
    (1, 'INT'),
    (2, 'INT'),
    (3, 'INT'),
    (4, 'INT'),
    (5, 'INT'),
    (6, 'INT'),
    (7, 'INT'),
    (8, 'INT'),
    (9, 'INT'),
    (('DEFAULT'), 'NINT')
]

f_pre_c_op = [
    (('RL', 'ANY', 'ANY'), 'POP'),
    (('RS', 'ANY', 'ANY'), 'POP'),
    (('RR', 'ANY', 'ANY'), 'POP'),
    (('DEFAULT'), 'NOP')
]

f_pre_w_op = [
    (('RS', 'ANY', 'ANY'), 'POP'),
    (('ANY', 'INT', 'ANY'), 'POP'),
    (('DEFAULT'), 'NOP')
]

f_post_c_op = [
    (('ANY', 'ANY', 'S'), 'PSH'),
    (('ANY', 'ANY', 'F'), 'PSH'),
    (('RL', 'ANY', 'ANY'), 'PSH'),
    (('RS', 'ANY', 'ANY'), 'PSH'),
    (('DEFAULT'), 'NOP')
]

f_post_c_val = [
    (('ANY', 'ANY', 'S'), 'SS'),
    (('ANY', 'ANY', 'F'), 'RL'),
    (('RL', 'ANY', 'ANY'), 'RS'),
    (('RS', 'ANY', 'ANY'), 'RR'),
    (('DEFAULT'), '_')
]

f_post_w_op = [
    (('SS', 'INT', 'ANY'), 'PSH'),
    (('DEFAULT'), 'NOP')
]

f_post_w_val = [
    (('SS', 'ANY', 'INT', 'INT', 'ANY'), 'SUM_MOD_10'),
    (('SS', 'ANY', 'NINT', 'ANY', 'ANY'), 'INP'),
    (('DEFAULT'), '_')
]

f_select = [
    (('RS', 'ANY'), 'W_PEEK'),
    (('RL', 'ANY'), '<'),
    (('RR', 'ANY'), '>'),
    (('DEFAULT'), '_')
]



##############################
# Cos Sim experiment

import torch


'''

A template vector embedded in a larger vector yields high similarity with
another template-containing vector, especially if the remainder of the
non-template portion has small magnitude.

1/R% of vectors `a` and `b` contain the template. Play around with the scale of
the remaining vector. At 1e-3, the cos-sim is 1.000

'''



N = 256
R = 4
template = torch.randn(N // R)
a = torch.randn(N) * 1e-1
a[:N//R] = template
b = torch.randn(N) * 1e-1
b[:N//R] = template

print(torch.cosine_similarity(a, b, dim=0))



###################
#
# *  [0, 1] == Bool
# *  relu(cossim(x, y)) == Bool
# *  cs1 * cs2 == And
# *  cs1 + cs2 == Or
# *  1 - cs == Not

import neurallambda.symbol as Sym
DEVICE = 'cuda'

cs = lambda x, y: torch.cosine_similarity(x, y, dim=0)

sym_map = Sym.SymbolMapper(N, Sym.chars + Sym.nums, device=DEVICE)
project = sym_map.project
unproject = sym_map.unproject

def f(x):
    '''

    IF-THEN-EQUALS in VSA

    if x==p: then a; else b

    (p and x) or (not p and y)

    '''
    p = project('p')
    a = project('a')
    b = project('b')
    sim_p = torch.cosine_similarity(p, x, dim=0)
    return (
        sim_p * a +
        (1 - sim_p) * b
    )

for lvl in [0, 1e-3, 1e-2, 1e-1, 5e-1, 1]:
    noise = torch.randn(N, device=DEVICE) * lvl
    print('should be `a`:', unproject(f(project('p') + noise), return_sim=True))

print('should be `b`:', unproject(f(project('q')), return_sim=True))
print('should be `b`:', unproject(f(project('a')), return_sim=True))
print('should be `b`:', unproject(f(project('b')), return_sim=True))
print('\n' * 5)

##########

def g(x):
    '''

    IF-THEN-EQUALS in VSA

    if x in {1, 2, 3, 4, 5, 6, 7, 8, 9}: then a; else b

    WARNING: if p has too many superposed vectors, the false branch can win out
    easily. You can correct by biasing the cos-sim values toward 1.0. But the
    returned value will still have significant cos sim with the false branch
    still, IE, both the T and F branches will be superposed.

    '''
    BIAS = 0.4 # must be in (0, 1], 1 means no bias, <1 biases toward true branch
    ps = [project(i) for i in range(10)]
    p = torch.stack(ps).sum(dim=0)
    a = project('A')
    b = project('B')
    sim_p = torch.cosine_similarity(p, x, dim=0)
    sim_p = ((sim_p + 0j) ** BIAS).real
    return (
        sim_p * a +
        (1 - sim_p) * b
    )

for x in range(15):
    gx = g(project(x))
    y = unproject(gx, return_sim=True)
    sim_a = torch.cosine_similarity(gx, project('A'), dim=0)
    sim_b = torch.cosine_similarity(gx, project('B'), dim=0)
    if x in set(range(10)):
        print('should be `A`:', y, f'sim_A={sim_a:>.3f}, sim_B={sim_b:>.3f}')
    else:
        print('should be `B`:', y, f'sim_A={sim_a:>.3f}, sim_B={sim_b:>.3f}')
