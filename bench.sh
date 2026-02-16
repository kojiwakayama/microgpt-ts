#!/bin/sh
set -eu

RUNS=3
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Portable sub-second timer (macOS date lacks %N)
now() { perl -MTime::HiRes=time -e 'printf "%.3f\n", time'; }

found=0
node_ok=0
results=""
fastest=999999

bench_one() {
  name=$1
  shift
  total=0
  i=1
  while [ "$i" -le "$RUNS" ]; do
    t0=$(now)
    "$@" > /dev/null 2>&1
    t1=$(now)
    elapsed=$(printf '%s\n' "$t1 - $t0" | bc)
    total=$(printf '%s\n' "$total + $elapsed" | bc)
    printf "  %s run %d: %ss\n" "$name" "$i" "$elapsed"
    i=$((i + 1))
  done
  avg=$(printf 'scale=2; %s / %s\n' "$total" "$RUNS" | bc)
  printf "  %s avg: %ss\n\n" "$name" "$avg"
  results="${results}${name} ${avg}
"
  is_faster=$(printf '%s < %s\n' "$avg" "$fastest" | bc -l)
  if [ "$is_faster" = "1" ]; then
    fastest=$avg
  fi
}

# Detect runtimes
if command -v bun > /dev/null 2>&1; then
  echo "Found: bun $(bun --version)"
  found=1
fi

if command -v node > /dev/null 2>&1; then
  if node -e 'const [a,b]=process.versions.node.split(".").map(Number);process.exit(a>23||(a===23&&b>=6)?0:1)' 2>/dev/null; then
    echo "Found: node $(node --version)"
    found=1
    node_ok=1
  else
    echo "Skip:  node $(node --version) (needs 23.6+ for TypeScript)"
  fi
fi

if command -v deno > /dev/null 2>&1; then
  echo "Found: deno $(deno --version | head -1)"
  found=1
fi

if command -v python3 > /dev/null 2>&1 && [ -f microgpt.py ]; then
  echo "Found: $(python3 --version)"
  found=1
fi

if [ "$found" -eq 0 ]; then
  echo "No supported runtimes found." >&2
  exit 1
fi

echo ""
echo "Running $RUNS iterations per runtime..."
echo ""

# Ensure training data exists
if [ ! -f input.txt ]; then
  echo "Downloading training data..."
  curl -sL https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt -o input.txt
fi

# Run benchmarks
if command -v bun > /dev/null 2>&1; then
  bench_one "bun" bun microgpt.ts
fi

if [ "$node_ok" -eq 1 ]; then
  bench_one "node" node microgpt.ts
fi

if command -v deno > /dev/null 2>&1; then
  bench_one "deno" deno run --allow-read --allow-write --allow-net microgpt.ts
fi

if command -v python3 > /dev/null 2>&1 && [ -f microgpt.py ]; then
  bench_one "python3" python3 microgpt.py
fi

# Summary
echo "=============================="
echo "  Runtime Benchmark Summary"
echo "=============================="
printf "%-14s %10s %10s\n" "Runtime" "Avg (s)" "Relative"
echo "--------------------------------------"

echo "$results" | while IFS=' ' read -r name avg; do
  [ -z "$name" ] && continue
  ratio=$(printf 'scale=1; %s / %s\n' "$avg" "$fastest" | bc)
  if [ "$ratio" = "1.0" ]; then
    label="1x (fastest)"
  else
    label="${ratio}x slower"
  fi
  printf "%-14s %10ss %10s\n" "$name" "$avg" "$label"
done
