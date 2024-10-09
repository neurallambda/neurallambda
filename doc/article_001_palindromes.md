# Article 001: Neural Stacks

## Abstract

In this work we use a differentiable neural stack to learn to solve toy problems, such as recognizing palindromes.

We then test it using tokens/embeddings that were never seen during training, and the learned representation of the problem generalizes successfully.

**Example:**

```
Train Palindromes, vocabulary = {a, b, c}:
`a b c | c b a`

Test Palindromes, vocabulary = {x, y z}:
`x y z | z y x`
```


## Why? Thesis:

If a NN can (1) learn a (2) symbolic program to deterministically (in the limit of noise) manipulate (3) novel inputs, that thing is AGI.

Once this happens in a natural language setting, people will be convinced that it's indeed AGI (and I'll make the bolder claim that if it is allowed to reason about its internal states, it may be AC).

"Deterministically" here means that (but for noise) a syntactic update rule performed on symbols will result in the same outputs.

Importantly, this process must completely ignore the **semantic** meaning of any symbol. For instance, if you ask Dall-E to give you an image of a horse riding atop a human, there are 2 ways it could reason about this request:

* Syntactically: `An x atop a y`

* Semantically: pattern match the words to interpolate across things it's familiar with

If it were pattern matching, it would (and does) show a human riding atop a horse, which is wrong.

If it were reasoning symbolically, it would give you the horse atop a human that you requested.

<div align="center">
  <img src="doc/horse.png" width="200" alt="A horse riding atop a human according to Dall-E">
</div>

Another example (from among many) that the community encountered early on in LLMs was with the request, "What weighs more, a pound of lead or two pounds of feathers?" The proper pattern match is, to think you've seen this problem before, and they weigh the same. The proper syntactic answer is that they've changed the puzzle to *two pounds* of feathers, which weighs more.

Or to drive this point home, talking with non-reasoning AI frequently feels like it does bayesian updates instead of symbolic updates:

```python
x = 10
x = 10
x = 10
x = 10
x = 10
x = 0

>>> print(x)
    9.2
```

To allow symbolic/syntactic updates, we'll look to computer science

## Stacks

A palindrome is a sentence/word spelled the same way forward and backward.

`This sentence is a palindrome!emordnilap a si ecnetnes sihT`

Can't an LLM already solve this? I would contend it can do it better than an unconscious observer just vibing the correctness of a palindrome, but worse than someone reasoning about it.

```
User: Is this a palindrome?
🌴 🎡 🥑 🦄 🎹 🌋 🎪 🍣 🚀 🎨 🦜 🍕 🗿 🎩 🌈 🎯 🍩 🦖 🎭 🚲 🦖 🍩 🎯 🌈 🎩 🗿 🍕 🦜 🎨 🚀 🍣 🎪 🌋 🎹 🦄 🥑 🎡 🌴

Opus: Yes, this is a palindrome using emoji characters. The sequence of emojis is the same forwards and backwards.

User: u sure?

Opus: Yes, I'm certain. If we look at the sequence closely, we can see that the emojis are mirrored exactly in the center, with the bicycle (🚲) in the middle.
```

A surprisingly good, yet ultimately incorrect answer. And this is not cherrypicked, this was our very first try.


If you're a leet python programmer, you could check if a string was a valid palindrome with:

```python
def is_palindrome(s):
    return s == s[::-1]
```

That uses a sequence equivalence construct, and list reversal, and 2 passes over the list. Those are tough to accomplish in a differentiable fashion (required for SGD to work). But we have a neuralstack at hand so let's first rewrite this problem using a python stack, and then we'll convert it to a neuralstack later:

```python
def is_palindrome(s):
    stack = []
    midpoint_reached = False
    for char in s:
        if char == '|':
            midpoint_reached = True
        elif not midpoint_reached:
            stack.append(char)
        else:
            if not stack or char != stack.pop():
                return False
    return True
```

Now we need an item equivalence construct to check when we've reached the midpoint, and also to compare popped items from the stack against current symbols. We also need the stack.

Item equivalence is pretty simple, FFNNs can check if 2 things are equivalent, or you can just use `cosine_similarity` which is a fancy term for a normalized dot product. Easy.

For how the stack works, we can `read`, `push`, or `pop` it, and we do this using tensors:

```python
def read(ss: StackState) -> torch.Tensor:
    '''Read the top of the stack.'''
    return ss.stack[:, 0]  # [B, S, D] -> [B, D]


def push(ss: StackState, val) -> StackState:
    ''' A guaranteed push op'''
    return StackState(
        torch.cat([val.unsqueeze(1),
                   ss.stack[:, :-1]], dim=1))


def pop(ss: StackState) -> StackState:
    ''' '''
    # read off top
    B, D = ss.stack.size(0), ss.stack.size(2)
    device, dtype = ss.stack.device, ss.stack.dtype
    pop_val = read(ss)
    nss = StackState(
        torch.cat([ss.stack[:, 1:],
                   torch.zeros(B, 1, D, device=device, dtype=dtype)], dim=1))
    return nss, pop_val
```


, it has to behave in a fuzzy, probabilistic way in order to be differentiable.

In this write up, I'll talk about the `Neuralstack`, a differentiable version of the stack you forgot from compsci 101, and I'll show some additions that allow a neural net to control it, for the purposes of solving this pernicious palindrome problem.

2 files carry the main code from these experiments:

* [`simple demo`](./demo/d02_palindrome.py): simply train a neuralstack on a test problem.

* [`architecture comparisons`](./demo/d02_palindrome_comparisons.py): prove what different architectures can do
