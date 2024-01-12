# Neurallambda

Lambda Calculus, Fully Differentiable.


## Explanation

Motivation:

<TODO>

TL;DR:

This work introduces a datatype encapsulating the lambda calculus, which is
stored in tensors. It also offers a differentiable beta-reduction function.

Further, the ideas present in this work lend themselves to other datatype
representations and other computation functions (that compute computation) that
are better than this naieve lambda calculus example.

1. Implement Lambda Calculus, wholly within a (pytorch) tensor
   representation. This Neurallambda is stored in 4 tensors.

2. Implement beta reduction using only e2e differentiable tensor operations.

3. Some QoL tools for writing programs into a Neurallambda, eg a human readable
   string like this:

```lisp
((fn [x y z] '(x y z)) 1 2 3)
```

turns into 4 tensors aka the Neurallambda.

4. Given a Neurallambda, which is 4 tensors, read back out a human-readable
   string representation of the program.

   Those 4 tensors include: 1) Memory Addresses, 2) A Type tag, 3+4) References
   to other addresses, or literal values.

Demonstration:

<TODO> : str -> terms -> mem -> neurallambda ---beta---> neurallambda -> mem -> terms -> str


## Todo

* Pedagogy:
  * Motivation of 4 tensors: tag tensor allows sum types, columns 1 and 2 allow product types
  * Computational Hierarchy: Pattern-matching < Prog Execution < Prog Validation < Prog Generation
  * References
  * Explain hypercomplex: Reals weren't training via backprop, Complex, and Quaternions were.

* NeuralX ("neural" is such a great prefix)
  * Neuralqueue, NeuralFSM, NeuralPDA, Neurallist, Neuraltree, Neuralgraph, simplified Find+Replace

* Empirical validation:
  * Test on simple RNNs
  * Build some benchmark suites to try these on

* Misc:
  * Improve test coverage
  * Does my computational hierarchy map back to the Chomsky hierarchy?
  * Try other machines/turing machines than lambda calc: combinators, a ticker tape model, FSMs, PDAs,
  * Loading memory: is there a nice differentiable way to read memory out from a bank of programs (akin to token embeddings, but, program embeddings here).

* Error correcting of neuralsymbols (eg tags, integers, addresses)
  * Pre-trained + frozen hopfield net? It's differentiable, and, could be linear, so, possibly some big optimizations possible.
  * Brute force: `without_grad` just replace symbols from LUT. This wouldn't work for training though.

* Optimizations:
  * Profile, for starters
  * Get a Linear Algebra pro to help fuse things down
  * There are some memory explosions some of the cos-sims, can we optimize?

* Robustify:
  * Clean:
    * `batch_size` is passed around poorly
    * `device` is passed around poorly
    * neurallambda_to_mem currently has nl passed, along with it's weights, separately
      * and: read_col
      * and: stack pretty_print fns
  * Collect assumptions made throughout and test:
    * hypercomplex
    * converting from mat form of hypercomplex back to vec form
    * same vec_size across addresses, tags, col1, col2


## Zip Repo

```sh
zip -r "neurallambda_$(date +"%Y-%m-%d_%H-%M-%S").zip" . -x "*__pycache__*" -x ".pytest_cache/*" -x ".env/*" -x ".git/*" -x "neurallambda*.zip"
``
