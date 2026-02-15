# microgpt.ts

A line-by-line TypeScript port of Andrej Karpathy's
[microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
— the most atomic way to train and inference a GPT, now in TypeScript.

Zero dependencies. Single file. Just `node microgpt.ts`.

## What is this?

A complete GPT implementation from scratch — autograd engine, transformer with
multi-head attention, Adam optimizer, training loop, and text generation — all
in one ~250 line file with no libraries.

It trains a character-level language model on a dataset of names, then generates
new ones:

```
step    1 / 1000 | loss 3.2326
step    2 / 1000 | loss 3.2376
...
step  999 / 1000 | loss 2.6316
step 1000 / 1000 | loss 2.2478

--- inference (new, hallucinated names) ---
sample  1: mamin
sample  2: jalela
sample  3: vishan
sample  4: elile
sample  5: carynn
sample  6: ramie
sample  7: ania
sample  8: saria
sample  9: raylan
sample 10: hamai
```

## Performance: TypeScript vs Python

Both implementations are pure scalar autograd with no BLAS, no tensors, no C
extensions — just objects and arithmetic. The only variable is the language
runtime.

| Runtime                       | Time (1000 steps) | Relative       |
| ----------------------------- | ----------------- | -------------- |
| **Node.js 25.2** (TypeScript) | **~6.3s**         | **1x**         |
| **Python 3.14** (CPython)     | **~50.2s**        | **~8x slower** |

> Benchmarked on Apple M3 Max, 128GB RAM. Average of 3 runs.

That's an **~8x speedup** from a straight port — no optimization, same
algorithm, same structure. V8's JIT is remarkably good at optimizing hot loops
with millions of small object allocations, which is essentially what scalar
autograd does.

## Run it

```bash
node microgpt.ts
```

Requires Node.js 23.6+ (native TypeScript support). Downloads the training data
automatically on first run.

## Credits

Original Python implementation by [Andrej Karpathy](https://x.com/karpathy).
This is a direct port preserving the same architecture, hyperparameters, and
structure.
