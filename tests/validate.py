#!/usr/bin/env python3
"""Validate a small-graph canonicalizer against nauty's isomorphism classes.

Usage: python3 tests/validate.py --cpu build/nauty-canonical-cpu
       python3 tests/validate.py --cpu build/nauty-canonical-cpu --cuda build/nauty-canonical-cuda

Only the Python standard library is needed.  The oracle tools are geng and
labelg from nauty.  The canonicalizer may choose a different canonical form
from nauty: we compare class partitions, and independently check every witness.
"""

import argparse
import itertools
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
UNLABELLED_COUNTS = (1, 1, 2, 4, 11, 34, 156, 1044, 12346)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def encode_graph6(rows):
    """Encode an adjacency matrix independently of the implementation."""
    n = len(rows)
    require(0 <= n <= 32, "graph6 test encoder supports orders 0 through 32")
    for i, row in enumerate(rows):
        require(isinstance(row, int) and 0 <= row < (1 << n), "invalid adjacency row")
        require(not (row >> i) & 1, "self-loop in test graph")
        for j in range(i):
            require(((row >> j) & 1) == ((rows[j] >> i) & 1), "asymmetric test graph")
    bits = [(rows[i] >> j) & 1 for j in range(1, n) for i in range(j)]
    bits.extend([0] * (-len(bits) % 6))
    data = []
    for start in range(0, len(bits), 6):
        value = 0
        for bit in bits[start:start + 6]:
            value = (value << 1) | bit
        data.append(chr(value + 63))
    return chr(n + 63) + "".join(data)


def decode_graph6(text):
    """Strictly decode the short graph6 form used by this prototype."""
    require(isinstance(text, str) and text, "missing graph6 string")
    require(all(63 <= ord(char) <= 126 for char in text), "invalid graph6 character")
    n = ord(text[0]) - 63
    require(n <= 32, f"unsupported graph6 order {n}")
    bit_count = n * (n - 1) // 2
    require(len(text) == 1 + (bit_count + 5) // 6, "incorrect graph6 length")
    bits = [(ord(char) - 63 >> shift) & 1
            for char in text[1:] for shift in range(5, -1, -1)]
    require(not any(bits[bit_count:]), "nonzero graph6 padding")
    rows = [0] * n
    position = 0
    for j in range(1, n):
        for i in range(j):
            if bits[position]:
                rows[i] |= 1 << j
                rows[j] |= 1 << i
            position += 1
    return tuple(rows)


def graph(n, edges):
    rows = [0] * n
    for u, v in edges:
        require(0 <= u < n and 0 <= v < n and u != v, "invalid fixture edge")
        rows[u] |= 1 << v
        rows[v] |= 1 << u
    return tuple(rows)


def relabel(rows, order):
    """The new vertex i is old vertex order[i], matching the CLI witness."""
    return tuple(sum(((rows[u] >> v) & 1) << j for j, v in enumerate(order))
                 for u in order)


def run(command, payload=None, timeout=180, allowed_codes=(0,)):
    result = subprocess.run(command, input=payload, capture_output=True,
                            text=True, timeout=timeout, check=False)
    require(result.returncode in allowed_codes,
            f"{command[0]} exited {result.returncode}: {result.stderr[-4000:]}")
    return result


def oracle_labels(inputs, options):
    result = run([options.labelg, "-q"], "\n".join(inputs) + "\n", options.timeout)
    labels = result.stdout.splitlines()
    require(len(labels) == len(inputs), "labelg returned the wrong number of graphs")
    for label in labels:
        decode_graph6(label)
    return labels


def invoke(binary, inputs, options, max_nodes=None, incomplete=False):
    if max_nodes is None:
        max_nodes = options.max_nodes
    command = [binary, "--max-nodes", str(max_nodes),
               "--batch-size", str(options.batch_size)]
    result = run(command, "\n".join(inputs) + "\n", options.timeout,
                 allowed_codes=(2,) if incomplete else (0,))
    lines = result.stdout.splitlines()
    require(len(lines) == len(inputs),
            f"{binary}: expected {len(inputs)} JSON records, got {len(lines)}")
    records = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise AssertionError(f"{binary}: invalid JSON at row {index}: {line!r}") from error
        require(isinstance(record, dict), f"{binary}: row {index} is not an object")
        require(record.get("status") in ("ok", "invalid_input", "node_limit"),
                f"{binary}: unrecognized status at row {index}: {record}")
        require(type(record.get("n")) is int, f"{binary}: n is not an integer at row {index}")
        require(type(record.get("nodes")) is int and record["nodes"] >= 0,
                f"{binary}: invalid nodes at row {index}")
        require(record["nodes"] <= max_nodes,
                f"{binary}: node budget exceeded at row {index}: {record}")
        if record["status"] != "ok":
            require(record.get("graph6") is None and record.get("order") == [],
                    f"{binary}: incomplete row {index} exposes a canonical result: {record}")
        records.append(record)
    return records


def check_success(source, record, context):
    require(record["status"] == "ok", f"{context}: incomplete result {record}")
    rows = decode_graph6(source)
    n = len(rows)
    require(record["n"] == n, f"{context}: incorrect vertex count")
    order = record.get("order")
    require(isinstance(order, list) and all(type(v) is int for v in order)
            and sorted(order) == list(range(n)), f"{context}: invalid permutation witness")
    output = decode_graph6(record.get("graph6"))
    require(output == relabel(rows, order),
            f"{context}: graph6 does not describe the graph induced by order")
    require(encode_graph6(output) == record["graph6"], f"{context}: nonstandard graph6 encoding")


def validate_class_partition(inputs, labels, records, context):
    """Prove a bijection of observed oracle classes and candidate forms."""
    oracle_to_candidate = {}
    candidate_to_oracle = {}
    for index, (source, oracle, record) in enumerate(zip(inputs, labels, records)):
        check_success(source, record, f"{context}, row {index}")
        canonical = record["graph6"]
        require(oracle_to_candidate.setdefault(oracle, canonical) == canonical,
                f"{context}: isomorphic inputs received different forms at row {index}")
        require(candidate_to_oracle.setdefault(canonical, oracle) == oracle,
                f"{context}: non-isomorphic inputs collided at row {index}")
    return len(oracle_to_candidate)


def validate_dataset(name, inputs, expected_classes, options):
    labels = oracle_labels(inputs, options)
    require(len(set(labels)) == expected_classes,
            f"{name}: unexpected nauty class count {len(set(labels))}, expected {expected_classes}")
    cpu = invoke(options.cpu, inputs, options)
    classes = validate_class_partition(inputs, labels, cpu, name + " CPU")
    if options.cuda:
        gpu = invoke(options.cuda, inputs, options)
        validate_class_partition(inputs, labels, gpu, name + " CUDA")
        for index, (host, device) in enumerate(zip(cpu, gpu)):
            require(host["graph6"] == device["graph6"],
                    f"{name}: CPU/CUDA canonical forms differ at row {index}")
    print(f"PASS {name}: {len(inputs):,} inputs, {classes:,} isomorphism classes", flush=True)


def with_permutations(inputs, rng, count):
    variants = []
    for source in inputs:
        rows = decode_graph6(source)
        variants.append(source)
        for _ in range(count):
            order = list(range(len(rows)))
            rng.shuffle(order)
            variants.append(encode_graph6(relabel(rows, order)))
    return variants


def unlabelled_graphs(options):
    inputs = [encode_graph6(())]  # geng starts at order 1.
    for n in range(1, options.max_order + 1):
        result = run([options.geng, "-q", str(n)], timeout=options.timeout)
        generated = result.stdout.splitlines()
        require(len(generated) == UNLABELLED_COUNTS[n],
                f"geng order {n}: expected {UNLABELLED_COUNTS[n]} graphs, got {len(generated)}")
        require(all(len(decode_graph6(item)) == n for item in generated), "geng order mismatch")
        inputs.extend(generated)
    return inputs


def labelled_graphs():
    inputs = []
    for n in range(6):
        edges = [(i, j) for j in range(1, n) for i in range(j)]
        for mask in range(1 << len(edges)):
            inputs.append(encode_graph6(graph(n, [edge for bit, edge in enumerate(edges)
                                                  if (mask >> bit) & 1])))
    require(len(inputs) == 1100, "incorrect labelled test population")
    return inputs


def symmetric_fixtures():
    fixtures = []
    for n in (0, 1, 2, 7, 16, 32):
        fixtures.append(graph(n, []))
        fixtures.append(graph(n, itertools.combinations(range(n), 2)))
    for n in (3, 4, 5, 6, 8, 16, 31, 32):
        fixtures.append(graph(n, [(i, (i + 1) % n) for i in range(n)]))
    for n in (2, 3, 5, 8, 31, 32):
        fixtures.append(graph(n, [(i, i + 1) for i in range(n - 1)]))
    for left, right in ((1, 31), (2, 30), (8, 8), (16, 16)):
        fixtures.append(graph(left + right, itertools.product(range(left), range(left, left + right))))
    for dimension in (3, 4, 5):
        fixtures.append(graph(1 << dimension,
                              [(v, v ^ (1 << bit)) for v in range(1 << dimension)
                               for bit in range(dimension)]))
    # Petersen, the 4-by-4 rook graph, and the Shrikhande graph.
    fixtures.append(graph(10, [(i, (i + 1) % 5) for i in range(5)]
                          + [(i + 5, (i + 2) % 5 + 5) for i in range(5)]
                          + [(i, i + 5) for i in range(5)]))
    fixtures.append(graph(16, [(u, v) for u in range(16) for v in range(u)
                               if u // 4 == v // 4 or u % 4 == v % 4]))
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))
    fixtures.append(graph(16, [(4 * x + y, 4 * ((x + dx) % 4) + (y + dy) % 4)
                               for x in range(4) for y in range(4) for dx, dy in steps]))
    for size in (4, 6):
        fixtures.append(graph(2 * size, [(offset + i, offset + (i + 1) % size)
                                        for offset in (0, size) for i in range(size)]))
    return list(dict.fromkeys(map(encode_graph6, fixtures)))


def validate_limits_and_invalid(options):
    cycle = encode_graph6(graph(6, [(i, (i + 1) % 6) for i in range(6)]))
    for binary in filter(None, (options.cpu, options.cuda)):
        for budget in (0, 1):
            records = invoke(binary, [cycle, cycle], options, max_nodes=budget, incomplete=True)
            require(all(record["status"] == "node_limit" for record in records),
                    f"{binary}: C6 unexpectedly completed with node budget {budget}")
            require(all(record["n"] == 6 for record in records), "node-limit record lost graph order")
        # Interleave valid rows to check processing continues after invalid input.
        inputs = ["@", "!", "A", "?x", "@?", "_", "`", "@"]
        records = invoke(binary, inputs, options, incomplete=True)
        for index, (source, record) in enumerate(zip(inputs, records)):
            if source == "@":
                check_success(source, record, f"{binary} mixed-input row {index}")
            else:
                require(record["status"] == "invalid_input",
                        f"{binary}: malformed graph6 {source!r} was not rejected")
    print("PASS node budgets and mixed valid/invalid input: incomplete rows expose no canonical form",
          flush=True)


def find_nauty_tool(name):
    installed = Path("/opt/homebrew/opt/nauty/bin") / name
    return str(installed) if installed.is_file() else shutil.which(name) or name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", default=str(ROOT / "build" / "nauty-canonical-cpu"))
    parser.add_argument("--cuda", help="CUDA executable; omitted means CUDA is explicitly untested")
    parser.add_argument("--geng", default=find_nauty_tool("geng"))
    parser.add_argument("--labelg", default=find_nauty_tool("labelg"))
    parser.add_argument("--max-order", type=int, default=7, choices=range(1, 9))
    parser.add_argument("--permutations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-nodes", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180,
                        help="timeout in seconds for each subprocess (default: 180)")
    options = parser.parse_args()
    if options.permutations < 1 or options.max_nodes < 1 or options.batch_size < 1 or options.timeout <= 0:
        parser.error("permutations, max-nodes, batch-size, and timeout must be positive")
    return options


def main():
    options = parse_args()
    if not options.cuda:
        print("CUDA NOT TESTED: no --cuda executable supplied", flush=True)
    # Fixed wire examples prevent a mutually consistent encoder/decoder bug.
    require(encode_graph6(()) == "?", "empty graph6 encoding")
    require(encode_graph6(graph(1, [])) == "@", "one-vertex graph6 encoding")
    require(encode_graph6(graph(2, [(0, 1)])) == "A_", "edge graph6 encoding")
    require(encode_graph6(graph(3, [(0, 1), (0, 2), (1, 2)])) == "Bw", "triangle graph6 encoding")
    require(encode_graph6(graph(5, [(0, 2), (0, 4), (1, 3), (3, 4)])) == "DQc",
            "asymmetric graph6 wire example")
    rng = random.Random(options.seed)
    inputs = unlabelled_graphs(options)
    validate_dataset(f"all unlabelled orders 0..{options.max_order}, seeded relabellings",
                     with_permutations(inputs, rng, options.permutations),
                     sum(UNLABELLED_COUNTS[:options.max_order + 1]), options)
    validate_dataset("all labelled orders 0..5", labelled_graphs(),
                     sum(UNLABELLED_COUNTS[:6]), options)
    fixtures = symmetric_fixtures()
    fixture_classes = len(set(oracle_labels(fixtures, options)))
    validate_dataset("symmetric and regular fixtures through order 32",
                     with_permutations(fixtures, rng, options.permutations),
                     fixture_classes, options)
    validate_limits_and_invalid(options)
    print("PASS all CPU correctness checks" + (" and CUDA parity checks" if options.cuda else ""), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
