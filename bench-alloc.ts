/**
 * Microbenchmark: isolates object allocation + method dispatch
 * to test whether JSC (Bun) vs V8 (Node) differ on this specific pattern.
 *
 * Mimics the Value autograd pattern from microgpt.ts:
 * - Creates millions of small objects with array children
 * - Chains method calls (.add, .mul)
 * - All objects are short-lived (GC pressure)
 */

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

  add(other: Value): Value {
    return new Value(this.data + other.data, [this, other], [1, 1]);
  }

  mul(other: Value): Value {
    return new Value(this.data * other.data, [this, other], [other.data, this.data]);
  }
}

// --- Benchmark: chain of add/mul creating ~10M Value objects ---
const ITERS = 1_000_000;

const t0 = performance.now();

let acc = new Value(1);
const b = new Value(0.5);

for (let i = 0; i < ITERS; i++) {
  acc = acc.add(b).mul(b);  // 2 allocations per iter = 2M Value objects
}

const t1 = performance.now();
const elapsed = (t1 - t0) / 1000;
const allocsPerSec = ((ITERS * 2) / elapsed / 1e6).toFixed(1);

console.log(`${ITERS * 2} Value allocations in ${elapsed.toFixed(3)}s`);
console.log(`${allocsPerSec}M allocs/sec`);
console.log(`result: ${acc.data}`); // prevent dead code elimination
