# TODO v1 (differentiable lambda calc)

- Roadmap as of May 2024
  - [ ] E001: Get Neuralstack + Sum Sequence working. `[t04_addition](./experiments/)`.
  - [ ] E002: Integrate Neuralstack with RWKV. `[rwkv fork](https://github.com/neurallambda/RWKV-infctx-trainer)`.
  - [ ] E003: Test new transformer stacks. `[t07_transformer](./experiments/)`.
  - [ ] E004: Get Neuralqueue + Sort working
  - [ ] ESOCIAL: Social Demos: how to make this stuff more digestable + user friendly?
  - [ ] ?? ESD: Stable Diffusion? that'd prob make it click with ppl


# Tickets

STACK
- [ ] E001: Arithmetic: flat associativity (no parens)
- [ ] E001: Arithmetic: parens
- [ ] E001: Binary Arithmetic


SOCIAL DEMOS
- [ ] ESOCIAL: Write up palindrome
- [ ] ESOCIAL: Move current README to a neurallambda-specific writeup
- [ ] ESOCIAL: Tighten up current README


QUEUE
- [ ] E004: Sorting using queues


RWKV
- [ ] E002: After Sum Sequence + Sorting are solved, re-integrate into RWKV
- [ ] E002: Determine if you should use infctx-trainer or RWKV-LM or custom
- [ ] E002: 4 way test: 1) random init + vanilla RWKV
- [ ] E002: 4 way test: 2) random init + stack + RWKV
- [ ] E002: 4 way test: 3) pretrained + vanilla RWKV
- [ ] E002: 4 way test: 4) pretrained + stack + RWKV


# Miscellaneous

* Toy Problems, to test components:
  * [ ] Sum Sequence
  * [ ] RNG
  * [ ] Balanced Parens?
  * [ ] Sort inputs
  * [ ] Find and replace
  * [ ] Find and emit
  * [ ] N-back
  * [ ] "Conditioned N-back", where some condition determines N realtime
  * [ ] Function over N-window-back
  * [ ] Fn Composition


* NOISE IN STACK VIA "zero_offset" when dotpdting pointer with stack values (IE zero_offset is highly biased, and adds up without cancelling). Randn may be too expensive to generate, plus not backproppable. No fixed value works. Maybe, diminish the values toward zero, and roll them (which makes them orthogonal). Test quantitty of noise reduction of this technique.

* Dynamic Time: output null tokens if you need more time to think
* Dynamic Space: Can multiple objects be added to a queue, stack, or array, in one timestep? If so, error correction would likely help.

* NeuralX ("neural" is such a great prefix)
  * [X] Neuralstack
  * [X] Neurallatch
  * [X] Neuralqueue
  * [ ] Neural List
  * [ ] Neural Array / Addressable Storage
  * [ ] Neural Dict / Content Addressable Storage
  * [ ] Simplified Find + Replace
  * [ ] Neural Heap
  * [ ] Neural Set
  * [ ] Neural FSM
  * [ ] Neural PDA
  * [ ] Neural Tree
  * [ ] Neural Graph
  * [ ] Neural Priority Queue
  * [ ] Neural Dequeue
  * [ ] Neural Algebraic Structures (Ring, Group, Field)

* Empirical validation:
  * Test on simple RNNs
  * Build some benchmark suites to try these on
  * RNG Bot that generates pseudo random numbers via program


* Misc:
  * Improve test coverage
  * For Stack, pointer is updated with torch.roll. This wraps around, possibly superposing unrelated information. Consider nn.ZeroPad2d(0, 0, -1, 1)
  * Does my computational hierarchy map back to the Chomsky hierarchy?
  * Neuralbeta currently uses kv_loopkup and replace functions that might be better abstracted as a Neuralarray.
  * Try other machines/turing machines than lambda calc:
    * combinatory logic, eg SKI
    * a ticker tape model
    * Typed Lambda Calc!?
    * FSMs
    * PDAs
    * PDA w 2 stacks is turing complete, I think
    * PDA w queue instead of stack is also turing complete, I think
    * More: post machines, tag systems, register machines, a myriad of Cellular Automata
  * Loading memory: is there a nice differentiable way to read memory out from a bank of programs (akin to token embeddings, but, program embeddings here).
  * Make a Dequeue (double ended queue, put and pop from L+R)

* Error correcting of neuralsymbols (eg tags, integers, addresses)
  * Look into error correction, both differentiable (Hopfield net) and non-differentiable (interpolate symbol vector with value from LuT).
  * Pre-trained + frozen hopfield net? It's differentiable, and, could be linear, so, possibly some big optimizations possible.
  * Brute force: `without_grad` just replace symbols from LUT. This wouldn't work for training though.

* Dynamic computation time via ability to output "null tokens". Similar to
  "delete tokens". This works if it's simultaneously outputting hidden state.

* Optimizations:
  * [X] Based on profiling, memory looks good, cosine_similarity is slooow, predominantly because of a hypercomplex.dot_product, and hypercomplex.hadamard. Solved when I got rid of hypercomplex.
  * Get a Linear Algebra pro to help fuse things down
  * There are some memory explosions some of the cos-sims, can we optimize?

* Robustify:
  * Clean:
    * `batch_size` is passed around poorly
    * `device` is passed around poorly
    * neurallambda_to_mem currently has nl passed, along with it's weights, separately
      * and: read_col
      * and: stack pretty_print fns
    * Move symbolics stuff out of Neurallambda into a "Symbol" class?
  * Collect assumptions made throughout and test:
    * [X] hypercomplex. If this isn't paying for itself, rip it out of the repo. Solution: deleted hypercomplex.
    * [X] converting from mat form of hypercomplex back to vec form. Solution: deleted hypercomplex.
    * same vec_size across addresses, tags, col1, col2
    * This neuralstack vs others?

* For Fun:
  * Arrow composition of nn.Modules, instead of just nn.Sequential. Haskell has
    an Category/Arrow semantics that would be great, especially for composing
    multiple streams of things such as joining tuples of values into one Linear
    operation, and then splitting the result back out into say a different
    shapped tuple. (This is likely too big a distraction). The advantages of
    this are statically knowing the computation graph ahead of time, so, it
    could be easier to optimize. Another is, if devs need to write a lot of
    different architectures, it could be to beneficial to have a nice "Free" (as
    in cat theory) data structure, which was based on an Arrow.


- Avoiding "Reasoning Shortcuts"
  - Train multiple tasks. Then a shortcut that would have worked for one task,
    must also work for the second task if it is to persist, which less likely.
  - Supervise-train latent space
  - Self-supervise, IE auto encode. It's hard for a reasoning shortcut to also
    excel at reconstructing inputs.
  - Disentanglement. Is this just making sure latent concepts are orthogonal to each other?
  - energy-based methods.



# Completed Tickets

(Recent completions @ bottom)

- [X] EINIT: Initial checkin of a lot of disparate code.
- [X] EMISC: Clean up imports
- [X] EMISC: Try stack in an RNN (`experiment/t03_modulo_game.py`)
- [X] EMISC: Profile code, find some hotspots
- [X] EMISC: Time profile results: `cosine_similarity` is the big obvious slowpoke
- [X] EMISC: Memory profile results: worst offenders so far are outside the main ML loop, and don't really matter: `string_to_neurallambda`, and `neurallambda_to_mem`
- [X] EMISC: Clean up how Stack initializes and handle batch_size, dtype, and device
- [X] EMISC: README: Motivation of 4 tensors: tag tensor allows sum types, columns 1 and 2 allow product types
- [X] EMISC: README: Computational Hierarchy: Pattern-matching < Prog Execution < Prog Validation < Prog Generation
- [X] EMISC: clean up ticketing setup
- [X] EMISC: clean up TODO.md
- [X] EMISC: Add directory layout to README
- [X] EMISC: Tighten up README
- [X] E002: Stack should not be nn.Module
- [X] E002: Stack.fwd should pass in `stack`, not have as attribute
- [X] E002: Stack.initialize should be top-level, optional, module fn
- [X] E002: chase through broken tests
- [X] E002: chase through broken deps (demo/)
- [X] E002: dataset caching is not working, re-downloads and processes every time.
- [X] E002: Build MyStack in rwkv repo
- [X] E002: Test running it in RWKV!
- [X] E002: Update prompting strategy in accordance with used dataset
- [X] E002: What datasets to start using? (made `awesome-reasoning` repo)
- [X] E002: Integrate tensorboard
- [X] E001: Add new improved Palindrome RNNStack
- [X] E001: Add ray data (skipping, we can pass data straight in using `with_parameters`, or directly in training fn
- [X] E001: Add ray tune
- [X] E001: Build model comparison tooling for increased training + search
- [X] E001: Compare RNN, LSTM, Transformer
- [X] ESOCIAL: Add new `palindrome` thing to `demo/`
- [X] ESOCIAL: Comment `palindrome` thing well
- [X] E001: Repro Armand Joulin Stack
- [X] E001: Rerun neural stack work with Joulin Stack
