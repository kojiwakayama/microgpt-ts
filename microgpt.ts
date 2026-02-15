/**
 * The most atomic way to train and inference a GPT in pure, dependency-free TypeScript.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 *
 * Ported from Andrej Karpathy's microgpt.py
 * https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";

// ---- Seeded PRNG (Mulberry32) ----

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rng = mulberry32(42);

function gauss(mean: number, std: number): number {
  const u1 = rng();
  const u2 = rng();
  return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function shuffle<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

function weightedChoice(weights: number[]): number {
  const total = weights.reduce((a, b) => a + b, 0);
  let r = rng() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r <= 0) return i;
  }
  return weights.length - 1;
}

// ---- Autograd Value ----

class Value {
  data: number;
  grad: number = 0;
  private _children: Value[];
  private _localGrads: number[];

  constructor(data: number, children: Value[] = [], localGrads: number[] = []) {
    this.data = data;
    this._children = children;
    this._localGrads = localGrads;
  }

  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return new Value(this.data + o.data, [this, o], [1, 1]);
  }

  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return new Value(this.data * o.data, [this, o], [o.data, this.data]);
  }

  pow(n: number): Value {
    return new Value(this.data ** n, [this], [n * this.data ** (n - 1)]);
  }

  log(): Value {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }

  exp(): Value {
    return new Value(Math.exp(this.data), [this], [Math.exp(this.data)]);
  }

  relu(): Value {
    return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]);
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.add(o.neg());
  }

  div(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.mul(o.pow(-1));
  }

  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    function buildTopo(v: Value): void {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._children) {
          buildTopo(child);
        }
        topo.push(v);
      }
    }

    buildTopo(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      for (let j = 0; j < v._children.length; j++) {
        v._children[j].grad += v._localGrads[j] * v.grad;
      }
    }
  }
}

// ---- Model helpers ----

function linear(x: Value[], w: Value[][]): Value[] {
  return w.map((wo) =>
    wo.reduce((acc, wi, i) => acc.add(wi.mul(x[i])), new Value(0))
  );
}

function softmax(logits: Value[]): Value[] {
  const maxVal = Math.max(...logits.map((v) => v.data));
  const exps = logits.map((v) => v.sub(maxVal).exp());
  const total = exps.reduce((a, b) => a.add(b));
  return exps.map((e) => e.div(total));
}

function rmsnorm(x: Value[]): Value[] {
  const ms = x
    .reduce((acc, xi) => acc.add(xi.mul(xi)), new Value(0))
    .div(x.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
}

// ---- Main ----

async function main() {
  // Data loading
  if (!existsSync("input.txt")) {
    const url =
      "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt";
    const res = await fetch(url);
    writeFileSync("input.txt", await res.text());
  }

  const docs = readFileSync("input.txt", "utf-8")
    .trim()
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  shuffle(docs);
  console.log(`num docs: ${docs.length}`);

  const uchars = [...new Set(docs.join(""))].sort();
  const BOS = uchars.length;
  const vocabSize = uchars.length + 1;
  console.log(`vocab size: ${vocabSize}`);

  // Hyperparameters
  const N_EMBD = 16;
  const N_HEAD = 4;
  const N_LAYER = 1;
  const BLOCK_SIZE = 16;
  const HEAD_DIM = N_EMBD / N_HEAD;

  // Model initialization
  function matrix(nout: number, nin: number, std = 0.08): Value[][] {
    return Array.from(
      { length: nout },
      () => Array.from({ length: nin }, () => new Value(gauss(0, std))),
    );
  }

  const stateDict: Record<string, Value[][]> = {
    wte: matrix(vocabSize, N_EMBD),
    wpe: matrix(BLOCK_SIZE, N_EMBD),
    lm_head: matrix(vocabSize, N_EMBD),
  };

  for (let i = 0; i < N_LAYER; i++) {
    stateDict[`layer${i}.attn_wq`] = matrix(N_EMBD, N_EMBD);
    stateDict[`layer${i}.attn_wk`] = matrix(N_EMBD, N_EMBD);
    stateDict[`layer${i}.attn_wv`] = matrix(N_EMBD, N_EMBD);
    stateDict[`layer${i}.attn_wo`] = matrix(N_EMBD, N_EMBD);
    stateDict[`layer${i}.mlp_fc1`] = matrix(4 * N_EMBD, N_EMBD);
    stateDict[`layer${i}.mlp_fc2`] = matrix(N_EMBD, 4 * N_EMBD);
  }

  const params = Object.values(stateDict).flatMap((mat) =>
    mat.flatMap((row) => row)
  );
  console.log(`num params: ${params.length}`);

  // GPT forward pass
  function gpt(
    tokenId: number,
    posId: number,
    keys: Value[][][],
    values: Value[][][],
  ): Value[] {
    const tokEmb = stateDict["wte"][tokenId];
    const posEmb = stateDict["wpe"][posId];
    let x = tokEmb.map((t, i) => t.add(posEmb[i]));
    x = rmsnorm(x);

    for (let li = 0; li < N_LAYER; li++) {
      let xResidual = x;
      x = rmsnorm(x);
      const q = linear(x, stateDict[`layer${li}.attn_wq`]);
      const k = linear(x, stateDict[`layer${li}.attn_wk`]);
      const v = linear(x, stateDict[`layer${li}.attn_wv`]);
      keys[li].push(k);
      values[li].push(v);

      const xAttn: Value[] = [];
      for (let h = 0; h < N_HEAD; h++) {
        const hs = h * HEAD_DIM;
        const qH = q.slice(hs, hs + HEAD_DIM);
        const kH = keys[li].map((ki) => ki.slice(hs, hs + HEAD_DIM));
        const vH = values[li].map((vi) => vi.slice(hs, hs + HEAD_DIM));

        const attnLogits = kH.map((kHt) =>
          qH
            .reduce((acc, qj, j) => acc.add(qj.mul(kHt[j])), new Value(0))
            .mul(1 / HEAD_DIM ** 0.5)
        );
        const attnWeights = softmax(attnLogits);

        for (let j = 0; j < HEAD_DIM; j++) {
          xAttn.push(
            attnWeights.reduce(
              (acc, wt, t) => acc.add(wt.mul(vH[t][j])),
              new Value(0),
            ),
          );
        }
      }

      x = linear(xAttn, stateDict[`layer${li}.attn_wo`]);
      x = x.map((a, i) => a.add(xResidual[i]));

      xResidual = x;
      x = rmsnorm(x);
      x = linear(x, stateDict[`layer${li}.mlp_fc1`]);
      x = x.map((xi) => xi.relu());
      x = linear(x, stateDict[`layer${li}.mlp_fc2`]);
      x = x.map((a, i) => a.add(xResidual[i]));
    }

    return linear(x, stateDict["lm_head"]);
  }

  // Training
  const learningRate = 0.01;
  const beta1 = 0.85;
  const beta2 = 0.99;
  const epsAdam = 1e-8;
  const m = new Float64Array(params.length);
  const v = new Float64Array(params.length);
  const numSteps = 1000;

  for (let step = 0; step < numSteps; step++) {
    const doc = docs[step % docs.length];
    const tokens = [BOS, ...doc.split("").map((ch) => uchars.indexOf(ch)), BOS];
    const n = Math.min(BLOCK_SIZE, tokens.length - 1);

    const keys: Value[][][] = Array.from({ length: N_LAYER }, () => []);
    const vals: Value[][][] = Array.from({ length: N_LAYER }, () => []);
    const losses: Value[] = [];

    for (let posId = 0; posId < n; posId++) {
      const tokenId = tokens[posId];
      const targetId = tokens[posId + 1];
      const logits = gpt(tokenId, posId, keys, vals);
      const probs = softmax(logits);
      losses.push(probs[targetId].log().neg());
    }

    const loss = losses.reduce((a, b) => a.add(b)).mul(1 / n);
    loss.backward();

    const lrT = learningRate * (1 - step / numSteps);
    for (let i = 0; i < params.length; i++) {
      const p = params[i];
      m[i] = beta1 * m[i] + (1 - beta1) * p.grad;
      v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2;
      const mHat = m[i] / (1 - beta1 ** (step + 1));
      const vHat = v[i] / (1 - beta2 ** (step + 1));
      p.data -= lrT * mHat / (vHat ** 0.5 + epsAdam);
      p.grad = 0;
    }

    console.log(
      `step ${String(step + 1).padStart(4)} / ${numSteps} | loss ${
        loss.data.toFixed(4)
      }`,
    );
  }

  // Inference
  const temperature = 0.5;
  console.log("\n--- inference (new, hallucinated names) ---");

  for (let sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
    const keys: Value[][][] = Array.from({ length: N_LAYER }, () => []);
    const vals: Value[][][] = Array.from({ length: N_LAYER }, () => []);
    let tokenId = BOS;
    const sample: string[] = [];

    for (let posId = 0; posId < BLOCK_SIZE; posId++) {
      const logits = gpt(tokenId, posId, keys, vals);
      const probs = softmax(logits.map((l) => l.mul(1 / temperature)));
      tokenId = weightedChoice(probs.map((p) => p.data));
      if (tokenId === BOS) break;
      sample.push(uchars[tokenId]);
    }

    console.log(
      `sample ${String(sampleIdx + 1).padStart(2)}: ${sample.join("")}`,
    );
  }
}

main();
