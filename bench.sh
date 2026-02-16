#!/usr/bin/env bash
set -euo pipefail

RUNS=3

# Collect available runtimes
declare -a RUNTIMES=()
declare -A RUNTIME_CMD

if command -v bun &>/dev/null; then
  RUNTIMES+=(bun)
  RUNTIME_CMD[bun]="bun microgpt.ts"
  echo "Found: bun $(bun --version)"
fi

if command -v node &>/dev/null; then
  RUNTIMES+=(node)
  RUNTIME_CMD[node]="node microgpt.ts"
  echo "Found: node $(node --version)"
fi

if command -v deno &>/dev/null; then
  RUNTIMES+=(deno)
  RUNTIME_CMD[deno]="deno run --allow-read --allow-write --allow-net microgpt.ts"
  echo "Found: deno $(deno --version | head -1)"
fi

if command -v python3 &>/dev/null && [ -f microgpt.py ]; then
  RUNTIMES+=(python3)
  RUNTIME_CMD[python3]="python3 microgpt.py"
  echo "Found: $(python3 --version)"
fi

if [ ${#RUNTIMES[@]} -eq 0 ]; then
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

declare -A RESULTS

for rt in "${RUNTIMES[@]}"; do
  total=0
  times=""
  for i in $(seq 1 $RUNS); do
    elapsed=$( { TIMEFORMAT='%R'; time ${RUNTIME_CMD[$rt]} > /dev/null 2>&1; } 2>&1 )
    total=$(echo "$total + $elapsed" | bc)
    times+="${elapsed}s "
    echo "  $rt run $i: ${elapsed}s"
  done
  avg=$(echo "scale=2; $total / $RUNS" | bc)
  RESULTS[$rt]=$avg
  echo "  $rt avg: ${avg}s"
  echo ""
done

# Print summary table
echo "=============================="
echo "  Runtime Benchmark Summary"
echo "=============================="
printf "%-14s %10s %10s\n" "Runtime" "Avg (s)" "Relative"
echo "--------------------------------------"

# Find fastest
fastest=999999
for rt in "${RUNTIMES[@]}"; do
  val=${RESULTS[$rt]}
  if (( $(echo "$val < $fastest" | bc -l) )); then
    fastest=$val
  fi
done

for rt in "${RUNTIMES[@]}"; do
  val=${RESULTS[$rt]}
  ratio=$(echo "scale=1; $val / $fastest" | bc)
  if [ "$ratio" = "1.0" ]; then
    label="1x (fastest)"
  else
    label="${ratio}x slower"
  fi
  printf "%-14s %10ss %10s\n" "$rt" "$val" "$label"
done
