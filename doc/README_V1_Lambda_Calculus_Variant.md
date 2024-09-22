# neurallambda - V1 lambda calculus variant

<a href="https://x.com/neurallambda">
  <img src="https://raster.shields.io/badge/follow-@neurallambda-blue.png?logo=x&color=BD2C00&labelColor=474240" alt="Follow on X" height="20">
</a>
&nbsp;&nbsp;
<a href="https://discord.gg/HRrPTQn2Uf">
  <img src="https://raster.shields.io/badge/discord-neurallambda-blue.png?logo=discord&logoColor=ffffff&color=BD2C00&labelColor=474240" alt="Join Discord" height="20">
</a>

<div align="center">
  <img src="doc/logo.png" width="150" alt="Blueberry soil pH">
</div>


# Reasoning AI, via Differentiable Lambda Calculus.

(And other NeuroSymbolic stuff: Stacks, Queues, Addressable Memory, Lists, Trees, Latches, etc.)



## **tl;dr**

If AI can compile a program in its latent space, it can Reason.

"Compile a program" will likely mean something much simpler than the following, but this full e2e-differentiable lambda calculus serves as an existence proof that this is possible, and compatible with gradient descent based AIs.

This library & research endeavor is compatible with SoTA RNNs and likely Transformers to eventually confer reasoning ability on them.

(run this demo [here](https://github.com/neurallambda/neurallambda/blob/e3c9dcd94c89640f1cf844ed4060e5369549cb68/demo/d01_neurallambda.py#L7))

<div align="center">
  <img src="doc/neurallambda demo.png" width="300" alt="Neurallambda compiling a program">
</div>

## V1 layout

```sh
demo/
    d01_neurallambda.py  # The example of e2e differentiable lambda calc

src/neurallambda/
    language.py          # the language spec of the Lambda Calculus
    memory.py            # the intermediary addressable "memory" representation
    stack.py             # a NeuralStack
    queue.py             # a NeuralQueue
    symbol.py            # project python objs to/from floating point tensors
    neurallambda.py      # the NeuralLambda and NeuralBeta classes
    transformer/stack.py # an experiment in adding a Stack to a Transformer

experiment/
    t00_neurallambda_sandbox.py      # the original Neurallambda demo
    t01_latch.py                     # a latch is like a stack with depth=1
    t03_palindrome.py                # a trivial proof that the Stack can train
    t04_addition*                    # trying to get more complex combos of Stacks to train
    t05_hyperdimensional_nand*       # a NANDish gate
    t06_rwkv_v4_sandbox.py           # ignore this, a better attempt is here: https://github.com/neurallambda/RWKV-infctx-trainer
    t07_transformer_stack_sandbox.py # Can we get a Stack into a transformer?

doc/
    TODO.md  # Coordinating the roadmap
```

## the problem

My premise all comes down to **"reasoning"**, and the lack thereof, in current AI models. I'll provide my working definition of "reasoning", but for a moment, please bear with a couple examples of reasoning failures.

**Transformers cannot reason:** | **Diffusion cannot reason:**
-- | --
![](doc/blueberries.png) | ![](doc/horse.png)
The correct response should have been `4.5 - 5.5`, and in reference to blueberries. These "multi leap" problems are tough for AI. | It's training set strongly biased its understanding of the relation between "riding" and "humans" and "horses", and it cannot navigate around that bias (even though the LLM portion recognized this inversion of the normal relationship!).

Current RNNs, SSMs, etc all also lack the ability to reason.

**AI is currently like a living textbook.** They can capture logic, performed by humans, and added into the training set, and then recombine and emit that human reasoning, and thus appear to reason, even perhaps slightly beyond their scope of training:

<div align="center">
  <img src="doc/socrates.png" width="300" alt="Is Socrates mortal?">
</div>

**But they cannot perform reasoning themselves.**

## what is reasoning?

**Reasoning** is the ability to know true things without having learned them.

It is building knowledge/predictions/retrodictions/actions atop *principles*, instead of *evidence*.

These principles are about **constraining the system** in appropriate ways, such that true things can be combined into new true things.

If you build a dataset and train an AI on projectiles in motion (arrows, cannonballs, rotten cabbage heads, etc.), you will grossly fail to infer the motions of satellites, which no longer follow parabolic curves like all your examples you were able to collect. But if you can derive a "program", say Newton's Laws, then you *can* infer things tremendously far beyond the domain of your training, say Planetary motion.

To make **reasoning** tractable for the scope of this library, I will consider forms of reasoning that are **programs**, in a computer science sense, but ported into the domain of Machine Learning (IE backproppable "programs" that live and operate within a neural net).

It is my (unproven) hope that by injecting reasoning into part (just *part*) of the architecture of LLMs, standard LLMs can begin to reason about natural language concerns (IE recipes, business advice, science, etc, and not just cold "programs"), and then beyond to other modes such as vision.


## what are programs?

I am interested especially in the Chomsky Hierarchy:

<div align="center">
  <img src="https://devopedia.org/images/article/210/7090.1571152901.jpg" width="300" alt="Chomsky Hierarchy">
</div>

There are several classes of "programs":

1. **Finite State Machines** / **FSM**: These are not turing complete, but constrain the system states to move within a graph. Examples include gas pumps and vending machines (you insert payment, make a selection, then it can dispense). Another example of the same power is a regular expression, such as this "program" that can recognize an html tag `<[^>]+>`.

2. **Push Down Automata** / **PDA**: These are FSMs, but add a universal Stack to keep track of things. An example that this can solve, which FSMs/regex cannot, is "N `a`s followed by N `b`s", eg `aaabbb`.

3. **Turing Machine**: These are machines capable of executing programs which can calculate anything calculatable. Your computer is a turing machine (in the limit of infinite memory and time). I'd also suggest that your conscious mind is Turing Complete.

Somewhere in there is the ability to execute logic programs, such as `A AND B OR (NOT C)`, or `(A IMPLIES B) AND A`.

This library ports this "machinery" into the world of fully differentiable Tensor land, to make this machinery compatible with standard neural net architectures.

An important mention is that most of this library works more like "data structures" and "machinery" than classical neural nets. This work can all operate *untrained*. IE the lambda calculus example executes programs with zero training/dataset/etc. It just works with whatever program you feed into it, like a python interpreter.


## prior art

Neural Turing Machines, Differentiable Neural Computers, Neurosymbolics, Hyperdimensional Computing / Vector Symbolic Architectures.


## tiers of "programming ability" / "reasoning ability"

1. An AI can execute programs

2. An AI can verify traits of a program

3. An AI can generate novel programs during training

4. An AI can generate novel programs post training


So far, this library provides an existence proof up to Level 1. It contains code which can execute arbitrary programs, written in a custom lisp dialect, in a fully differentiable setting. (Some tantilizing tests have proven up to Level 3, that AI can *learn* novel programs to solve toy problems via SGD, but, there are still frontiers of research here).

You can write lisp programs in a human readable form, deterministically translate them into tensors, compile/interpret the programs (ie lambda calculus's beta reduction), and then read back out of the resulting tensors a human readable result, ie the results of running your program.

Example:

```
((fn [x] '(x x x)) 42)   # In python syntax:  (lambda x: [x, x, x])(42)
--->
tensors
-- differentiable interpreter -->
tensors
--->
'(42 42 42)              # In python syntax: [42, 42, 42]
```

## where to get started with this lib?

Great question. Right now there's a libraryish portion in `neurallambda/`, and the improvements are starting in `experiments/`, but this whole repo is largely temporarily more like an Open Research project.

I'm improving the ergonomics of this lib daily, but it's research grade right now, so, make sure you have a stiff cup of coffee, and a heavy dose of forgiveness at the ready.

* **[Lambda Calc Example](https://github.com/neurallambda/neurallambda/blob/master/demo/d01_neurallambda.py)**: This should "just run". It demonstrates hand-written programs getting read into tensors, then being beta-reduced for a fixed number of steps, and then the resulting program read back out of tensor land and made human readable. Everything that goes `tensor -> tensor` is backprop friendly (ie just not the `human string -> tensor` portion, nor `tensor -> human string`).

* **[Symbols](https://github.com/neurallambda/neurallambda/blob/master/src/neurallambda/symbol.py#L76)**: Anything I want to represent in tensors gets an associated random vec. Eg to represent `"hello"`, I could use `torch.randn(256)`. For `42`, also `torch.randn(256)`. These are saved in a dictionaries, and I can `project` into them via a lookup, and `unproject` from tensor to python object via finding the symbol with the nearest `cosine_similarity` to a query vector.

* **[Stacks](https://github.com/neurallambda/neurallambda/blob/master/src/neurallambda/stack.py#L65)**: These are great for explaining some underlying concepts. That link shows how the stack's pointer is a superposition of 3 potentialities: the user `push`ed, `pop`ped, or `null_op`ed. If the caller intended a `pop`, they would pass in `should_pop == 1.0`, and `should_push, should_null_op == 0.0`. Then all 3 things happen in superposition, but only one matters. The reason for this is, this superposition technique is required to make the stack differentiable. You can't differentiate `if should_pop: ... elif: should_push...`.

* **[Substitution in lambda calc](https://github.com/neurallambda/neurallambda/blob/master/src/neurallambda/neurallambda.py#L523)**: If you're craving a headache, check out how substitution is done in the lambda calc example. I'll repeat, this is likely far too heavy handed compared to the theoretical ideal of what we'll eventually stick in an LLM, but, this is an existence proof that reasoning is possible inside a fully differentiable setting, and this `substitution` operation is the key to computation.

* **[Lighter-weight Substitution](https://github.com/neurallambda/neurallambda/blob/master/experiment/t05_hyperdimensional_nand_02_substitution.py)**: Substitution is so critical to many formalisms of "computation", that I [created this experiment](https://github.com/neurallambda/neurallambda/blob/master/experiment/t05_hyperdimensional_nand_02_substitution.py#L171) to see how simple I could get while still performing something that looks like substitution. It eschews the `neurallambda` stuff, and just uses a small symbolicy module called `NAND` which can take in arbitrary numbers of vectors, determine if they match/don't an internal parameter which conceptually gives a `True/False` analog, and then NAND them all together.

I'm sure there's more you'd like to know, help me focus my communication efforts by asking a question in an [Issue](https://github.com/neurallambda/neurallambda/issues) or collabing however you see fit! I'd love to work together.

## neurallambda, a little more in depth:

Here's a example of reading in a simple program, and doing differentiable beta reduction on it.

```python
# Human Readable

((fn [x] x) 42)   # python: (lambda x: x)(42)

# Parsed AST

Apply (Lambda (Var "x") (Var "x")) (Int 42)

# Virtual Memory. "A1" means "Address 1", pointing back into this dictionary
#   All tuples are sized 1, 2 or 3.

{
0: ("Apply", A1, A3)
1: ("Lambda", A2, A2)
2: ("Var", "x")
3: ("Int", 42)
}

# 4 Tensors: address, tag, col1, col2
#   These 4 tensors correspond to those tuples and their addresses

# The concept (not the implementation).
#   Recall, `project` sends any python object to `torch.randn(vec_size)`, and
#   keeps a lookup table so you can project back
{
project(0): (project("Apply"), project(A1), project(A3))
project(1): (project("Lambda"), project(A2), project(A2))
project(2): (project("Var"), project("x"))
project(3): (project("Int"), project(42))
}

# Actual tensor implementation is this, but `torch.stack`ed into one
# tensor. "Column 1/2" refers to the position in that tuple.

| Addres     | Tag               | Column 1     | Column 2    |
|------------+-------------------+--------------+-------------|
| project(0) | project("Apply")  | project(A1)  | project(A3) |
| project(1) | project("Lambda") | project(A2)  | project(A2) |
| project(2) | project("Var")    | project("x") |             |
| project(3) | project("Int")    | project(42)  |             |


# Beta reduction

1. That tensor above represents the "program".

2. To do `beta reduction`, we'll need to mutate that tensor. To start, we'll
   need to keep track of whether or not an expression is reduced yet. We can't
   reduce a compound term until we know if its subexpressions are reduced.

   Let's keep track of "Is Reduced" with 2 new columns, IR1 and IR2. These hold
   scalars in `[0.0, 1.0]` which are conceptually `bool`s:

| Addres     | Tag               | Column 1     | Column 2    | IR1 | IR2 |
|------------+-------------------+--------------+-------------+-----+-----|
| project(0) | project("Apply")  | project(A1)  | project(A3) |   0 |   0 |
| project(1) | project("Lambda") | project(A2)  | project(A2) |   0 |   0 |
| project(2) | project("Var")    | project("x") |             |   0 |   0 |
| project(3) | project("Int")    | project(42)  |             |   0 |   0 |

3. Now we need to start a depth first tree search, to check if nodes of the tensor-AST
   are reduced yet, and do substitution if they look like `Apply (Lambda x body) y`

   We start at address 0. It's that `Apply`, with an expression on the left
   (col1) and right (col2). We can see that IR1 and IR2 are both False, so push
   both on a Neuralstack because we'll need to visit them next.

4. Pop the stack and handle the expression at that address similarly to step 3.

5. If at any point we've come across an expression where both `IR1 ~= 1.0` and
   `IR2 ~= 1.0`, we can mark this expression also as reduced. This is done by
   updating the entire tensor-AST. Any cell that refers to this address gets its
   corresponding `IR` column changed to 1.0. Now remember, this is all happening
   in this non-binary, superpositiony way, so all of these operations are a
   little bit fuzzy.

6. Furthermore, if an expression has reduced left and right terms, and looks
   like `Apply (Lambda param body) arg`, we can do substitution. This is done by
   replacing the `Apply` term at the current address with the `body`, but also
   replacing every occurence of `param` with `arg`. Now, big caveat here. In a
   normal setting, variables are properly scoped, and you only replace `param`
   with `arg` within the `body`. This library doesn't support var scoping yet,
   so, every occurence of eg the variable `x` in the entire program gets
   replaced with the `arg`. If your program came in from a human readable
   program, via "alpha equivalence" you can make sure that every var name is
   unique so this isn't an issue in practice. The one place it becomes an issue
   is with recursion, which this lib doesn't handle yet, but there is a path to
   it.
```

## the frontier

**TL;DR:** Jam some of this work into the [RWKV](https://github.com/BlinkDL/RWKV-LM) project; a pretrained LLM that uses an RNN only, no transformer, but is competitive with same-sized transformers.

**TL;DR 2:** Please let's collaborate! **neurallambda -AT- proton -DOT- me**

This work so far proves that some of the most extreme cases of Reasoning are possible in an end-to-end differentiable setting, but I suspect that the full Lambda Calculus may be (significantly) more heavy handed than is needed.

For instance, it is known that if you have a classical Finite State Machine, and upgrade it with a simple Queue, you now have a Turing Complete machine. If a classical Neural Net can reasonbly be seen as an FSM (I suspect that this is the case), then perhaps you can have a full Reasoning Computer by just adding a Queue to your AI.

* **Step 1**: Prove out that simple structures, such as NeuralQueues, can be trained within an RNN. By "trained", I mean can the RNN learn to make use of the datasructure, and push/pop/null_op appropriately to solve a task.

* **Step 2**: Crack open a pretrained LLM, and add this in.

  * I like [RWKV](https://github.com/BlinkDL/RWKV-LM). He has released OSS friendly pretrained models from small sizes up to 7B and 13B sizes. The tiny sized versions (down to ~100M params) will be great for experimenting and iterating on. SSM models like Mamba may work great too. Transformer models... possibly, if they were side-chained with RNNs.

  * **Architectural choices**:

    * What Neurosymbolic stuff should be added? Can we get by with just a single Queue?

    * What parameters should be left frozen / unfrozen? We'll likely keep much of the original RNN frozen.

    * Can we train the programs separately from the semantic side of things? IE, worst case scenario, neurallambda stuff trains poorly in SGD settings, so can we prepopulate a vector database with "programs", possibly hand written, and leave that frozen, such that the model learns which programs to use against it's latent space to improve its performance against our loss function?

    * What dataset should we use? For the program portion, we can likely use hand-rolled synthetic datasets. Honestly, we could probably even prompt engineer an LLM to help write these programs.

    * What loss function should we use? Perhaps we read out the learned-programs and do supervised learning?

  * **Optimizations**:

    * If the "programs" remain human readable, and perhaps if not, during inference we can replace all the super heavy tensor stuff with super cheap conventional system calls.

    * All the "machinery" in this library relies basically just on **linear projections and softmax**. That's mostly it. I'm sure we could compile the wild tensor stuff before training to dramatically reduce the FLOPs needed.

  * **License**: Thus far, this work is unlicensed and I retain all rights to it, but would like to determine the ideal mode of opensourcing it.


# conclusion

There's work to be done!

I'm happy to accept Grants, GH Issues, PRs, chat with people, start a discord, maybe a youtube to explain some of these ideas, etc.

Please get involved!

**neurallambda -AT- proton -DOT- me**
