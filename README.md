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
| **Python 3.14** (CPython)     | **~50.3s**        | **~8x slower** |

> Benchmarked on Apple M3 Max, 128GB RAM. Average of 3 runs.

That's an **~8x speedup** from a straight port — no optimization, same
algorithm, same structure. JIT compilation makes a massive difference when the
workload is millions of small object allocations, which is essentially what
scalar autograd does.

## Run it

```bash
node microgpt.ts
```

Requires Node.js 23.6+ (native TypeScript support). Also works with
`bun microgpt.ts` or `deno run --allow-read --allow-write --allow-net microgpt.ts`.

Downloads the training data automatically on first run.

## Wider runtime comparison

The same TypeScript file also runs on Bun and Deno — all three produce identical
output.

| Runtime                       | Time (1000 steps) | vs Python         |
| ----------------------------- | ----------------- | ----------------- |
| **Bun 1.3.6** (JavaScriptCore)| **~3.8s**         | **~13x faster**   |
| **Node.js 25.2** (V8)         | **~6.3s**         | **~8x faster**    |
| **Deno 2.6.7** (V8)           | **~6.8s**         | **~7x faster**    |
| **Python 3.14** (CPython)     | **~50.3s**        | baseline          |

Reproduce on your machine:

```bash
./bench.sh
```

## Why is Bun 1.6x faster than Node?

Bun uses JavaScriptCore (JSC), Node uses V8 — but which part of the engine
explains the gap? We tested and eliminated the obvious suspects:

| Hypothesis | Test | Result |
| --- | --- | --- |
| V8 JIT warmup | Ran training twice in same process | 3% speedup — not it |
| Startup time | Bare `node -e '0'` vs `bun -e '0'` | 37ms diff — negligible |
| GC pressure | `node --trace-gc` | 8 pauses, ~15ms total — not it |
| TypeScript stripping | Pre-compiled to JS, ran on Node | Same speed — not it |

Then we isolated the variable with `bench-alloc.ts` — a microbenchmark that
creates 2M `Value` objects (the same autograd class microgpt uses):

| Engine | Allocs/sec | Relative |
| --- | --- | --- |
| V8 (Node) | 12.8M | 1x |
| JSC (Bun) | 20.9M | **1.63x faster** |

That 1.63x matches the 1.6x overall speedup almost exactly. The entire
performance gap comes from JSC's faster object allocation throughput on
short-lived objects.

Run it yourself:

```bash
node bench-alloc.ts
bun bench-alloc.ts
```

## Credits

Original Python implementation by [Andrej Karpathy](https://x.com/karpathy).
This is a direct port preserving the same architecture, hyperparameters, and
structure.
