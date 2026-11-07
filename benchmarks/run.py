#!/usr/bin/env python3
"""Generate reproducible corpora and benchmark canonical labelling serially.

Run from the project root after building the benchmark targets:
    python3 benchmarks/run.py --cpu build/nauty-benchmark-cpu
    python3 benchmarks/run.py --cpu build/nauty-benchmark-cpu --cuda build/nauty-benchmark-cuda

The C++ executables time canonicalization of already parsed graphs.  This
driver does not time graph generation, parsing, process startup, or JSON I/O.
CPU measurements use one thread, not the whole machine.  CUDA measurements,
when explicitly requested, include transfers and report kernel time separately.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "geng8", "random32-p010", "random32-p050", "cycles32", "hypercubes32",
    "rook-shrikhande16", "clique-empty32",
)


def helpers():
    spec = importlib.util.spec_from_file_location("canonical_validation", ROOT / "tests" / "validate.py")
    module = importlib.util.module_from_spec(spec)
    # Avoid writing a __pycache__ beside the correctness test.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def find_tool(name):
    installed = Path("/opt/homebrew/opt/nauty/bin") / name
    return str(installed) if installed.is_file() else shutil.which(name) or name


def generate_corpus(name, options, helper):
    rng = random.Random(f"{options.seed}:{name}")
    if name == "geng8":
        result = subprocess.run([options.geng, "-q", "8"], capture_output=True,
                                text=True, check=True, timeout=options.timeout)
        lines = result.stdout.splitlines()
        if len(lines) != 12346:
            raise ValueError(f"geng 8 produced {len(lines)} graphs, expected 12346")
        description = "Every unlabelled simple graph of order 8, in geng output order"
    elif name.startswith("random32"):
        probability = 0.1 if name.endswith("p010") else 0.5
        lines = [helper.encode_graph6(helper.graph(
            32, [(i, j) for j in range(1, 32) for i in range(j) if rng.random() < probability]
        )) for _ in range(options.random_count)]
        description = f"{options.random_count} independent G(32, {probability}) labelled graphs"
    else:
        if name == "cycles32":
            bases = [helper.graph(32, [(i, (i + 1) % 32) for i in range(32)])]
            description = "Seeded relabellings of the cycle C32"
        elif name == "hypercubes32":
            bases = [helper.graph(32, [(v, v ^ (1 << bit)) for v in range(32) for bit in range(5)])]
            description = "Seeded relabellings of the 5-dimensional hypercube"
        elif name == "rook-shrikhande16":
            rook = helper.graph(16, [(u, v) for u in range(16) for v in range(u)
                                     if u // 4 == v // 4 or u % 4 == v % 4])
            steps = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))
            shrikhande = helper.graph(16, [(4 * x + y, 4 * ((x + dx) % 4) + (y + dy) % 4)
                                           for x in range(4) for y in range(4) for dx, dy in steps])
            bases = [rook, shrikhande]
            description = "Alternating seeded relabellings of the 4x4 rook and Shrikhande graphs"
        else:
            bases = [helper.graph(32, []), helper.graph(32, itertools.combinations(range(32), 2))]
            description = "Alternating empty and complete graphs of order 32"
        lines = []
        for index in range(options.symmetric_count):
            rows = bases[index % len(bases)]
            order = list(range(len(rows)))
            rng.shuffle(order)
            lines.append(helper.encode_graph6(helper.relabel(rows, order)))
    orders = sorted({len(helper.decode_graph6(line)) for line in lines})
    payload = "\n".join(lines) + "\n"
    return payload, {"name": name, "description": description, "graphs": len(lines),
                     "orders": orders, "seed": options.seed,
                     "sha256": hashlib.sha256(payload.encode("ascii")).hexdigest()}


def command_for(backend, options, repeats, warmup, loops):
    command = [options.cuda] if backend == "cuda" else [options.cpu, "--backend", backend]
    command += ["--repeats", str(repeats), "--warmup", str(warmup),
                "--loops", str(loops), "--max-nodes", str(options.max_nodes)]
    if backend == "cuda":
        command += ["--batch-size", str(options.batch_size)]
    return command


def check_summary(summary, corpus, backend, repeats, warmup, loops, options):
    if not isinstance(summary, dict):
        raise ValueError("benchmark output must be one JSON object")
    if summary.get("status") != "ok":
        raise ValueError(f"benchmark status is {summary.get('status')!r}, not 'ok'")
    if summary.get("backend") != backend:
        raise ValueError("benchmark reported a different backend")
    if summary.get("repeats") != repeats or summary.get("warmup") != warmup:
        raise ValueError("benchmark reported a different repeat or warmup count")
    if summary.get("max_nodes") != options.max_nodes:
        raise ValueError("benchmark reported a different node budget")
    if backend != "cuda" and summary.get("threads") != 1:
        raise ValueError("CPU benchmark must use one thread")
    if backend == "cuda" and summary.get("batch_size") != options.batch_size:
        raise ValueError("CUDA benchmark reported a different batch size")
    if summary.get("graphs") != corpus["graphs"] or summary.get("completed") != corpus["graphs"]:
        raise ValueError("benchmark did not complete the entire base corpus")
    if summary.get("loops") != loops:
        raise ValueError("benchmark reported a different loop count")
    if not isinstance(summary.get("checksum"), str) or not re.fullmatch(r"[0-9a-f]{16}", summary["checksum"]):
        raise ValueError("benchmark checksum must be 16 lowercase hexadecimal digits")
    operations = corpus["graphs"] * loops
    if "operations" in summary and summary["operations"] != operations:
        raise ValueError("benchmark operation count differs from graphs * loops")
    trials = summary.get("trials_seconds")
    if not isinstance(trials, list) or len(trials) != repeats:
        raise ValueError("benchmark returned an incorrect number of timed trials")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in trials):
        raise ValueError("benchmark returned nonpositive or nonfinite trial times")
    median = summary.get("median_seconds")
    if type(median) not in (int, float) or not math.isfinite(median) or median <= 0:
        raise ValueError("benchmark returned an invalid median")
    if not math.isclose(median, statistics.median(trials), rel_tol=1e-5, abs_tol=1e-12):
        raise ValueError("reported median does not match the trial times")
    if backend == "cuda" or "kernel_trials_seconds" in summary:
        kernel = summary.get("kernel_trials_seconds")
        if not isinstance(kernel, list) or len(kernel) != repeats:
            raise ValueError("incorrect number of CUDA kernel trials")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in kernel):
            raise ValueError("invalid CUDA kernel times")
        if not math.isclose(summary.get("kernel_median_seconds", -1), statistics.median(kernel),
                            rel_tol=1e-5, abs_tol=1e-12):
            raise ValueError("reported kernel median does not match the kernel trials")


def invoke(command, payload, corpus, backend, repeats, warmup, loops, options, deadline):
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        return {"status": "budget_exhausted", "command": command,
                "error": "overall benchmark time limit reached"}
    try:
        result = subprocess.run(command, input=payload, capture_output=True, text=True,
                                timeout=min(options.timeout, remaining), check=False)
    except subprocess.TimeoutExpired as error:
        return {"status": "timeout", "command": command,
                "error": f"subprocess exceeded {error.timeout:.1f} seconds"}
    except OSError as error:
        return {"status": "execution_error", "command": command, "error": str(error)}
    record = {"command": command, "returncode": result.returncode,
              "stderr": result.stderr[-16000:]}
    try:
        summary = json.loads(result.stdout)
        record["summary"] = summary
    except json.JSONDecodeError:
        record.update(status="benchmark_error", error="benchmark did not return a JSON summary",
                      stdout=result.stdout[-16000:])
        return record
    if result.returncode != 0:
        status = summary.get("status") if isinstance(summary, dict) else None
        record.update(status="incomplete" if status == "incomplete" else "benchmark_error",
                      error=f"benchmark exited {result.returncode}; timings are not comparable")
        return record
    try:
        check_summary(summary, corpus, backend, repeats, warmup, loops, options)
    except (ValueError, TypeError) as error:
        record.update(status="invalid_summary", error=str(error))
        return record
    record["status"] = "ok"
    return record


def measure(backend, payload, corpus, options, deadline, reference_checksum=None):
    measurement = {"dataset": corpus["name"], "backend": backend,
                   "graphs": corpus["graphs"], "corpus_sha256": corpus["sha256"]}
    if backend == "cuda":
        measurement["cpu_reference_checksum"] = reference_checksum
        measurement["cpu_checksum_verified"] = False
        if reference_checksum is None:
            measurement.update(status="cpu_reference_unavailable",
                               error="complete prototype CPU result required before CUDA timing")
            return measurement
    pilot = invoke(command_for(backend, options, 1, options.warmup, 1), payload, corpus,
                   backend, 1, options.warmup, 1, options, deadline)
    measurement["calibration"] = pilot
    if pilot["status"] != "ok":
        measurement.update(status=pilot["status"], error="calibration: " + pilot.get("error", "failed"))
        return measurement
    pilot_checksum = pilot["summary"]["checksum"]
    if backend == "cuda" and pilot_checksum != reference_checksum:
        measurement.update(status="checksum_mismatch",
                           error="CUDA calibration checksum differs from prototype CPU; timings rejected")
        return measurement
    pilot_seconds = pilot["summary"]["median_seconds"]
    loops = min(options.max_loops, max(1, math.ceil(1.15 * options.target_seconds / pilot_seconds)))
    measurement["loops"] = loops
    trial = invoke(command_for(backend, options, options.repeats, options.warmup, loops),
                   payload, corpus, backend, options.repeats, options.warmup, loops, options, deadline)
    measurement["run"] = trial
    measurement["status"] = trial["status"]
    if trial["status"] != "ok":
        measurement["error"] = trial.get("error", "measurement failed")
        return measurement
    summary = trial["summary"]
    if summary["checksum"] != pilot_checksum:
        measurement.update(status="checksum_mismatch",
                           error="timed checksum differs from calibration; timings rejected")
        return measurement
    measurement["checksum"] = summary["checksum"]
    if backend == "cuda":
        measurement["cpu_checksum_verified"] = True
    operations = corpus["graphs"] * loops
    measurement.update(
        operations=operations,
        median_seconds=summary["median_seconds"],
        microseconds_per_graph=1e6 * summary["median_seconds"] / operations,
        graphs_per_second=operations / summary["median_seconds"],
        trial_microseconds_per_graph=[1e6 * value / operations for value in summary["trials_seconds"]],
        below_target=summary["median_seconds"] < options.target_seconds,
    )
    if "kernel_median_seconds" in summary:
        measurement["kernel_microseconds_per_graph"] = 1e6 * summary["kernel_median_seconds"] / operations
    return measurement


def number(value, precision=3):
    return "-" if value is None else f"{value:.{precision}f}"


def write_results(report, destination):
    cpu = {(row["dataset"], row["backend"]): row["microseconds_per_graph"]
           for row in report["measurements"] if row["backend"] != "cuda" and row["status"] == "ok"}
    for row in report["measurements"]:
        if row["status"] == "ok" and (row["backend"] != "cuda" or row.get("cpu_checksum_verified")):
            for backend in ("prototype", "nauty", "traces"):
                baseline = cpu.get((row["dataset"], backend))
                if baseline is not None:
                    row[f"speedup_vs_{backend}_cpu"] = baseline / row["microseconds_per_graph"]
    (destination / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = ["dataset", "backend", "status", "graphs", "loops", "operations", "median_seconds",
               "microseconds_per_graph", "graphs_per_second", "speedup_vs_prototype_cpu",
               "speedup_vs_nauty_cpu", "speedup_vs_traces_cpu",
               "kernel_microseconds_per_graph", "checksum", "cpu_checksum_verified", "below_target", "error"]
    with (destination / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["measurements"])
    lines = [
        "Canonical labelling timings: single-threaded CPU baselines",
        "Workstation timings; background processes were not controlled.",
        "Core timers exclude generation, graph6 parsing, startup, and JSON I/O.",
        "CUDA primary times include H2D + kernel + D2H; kernel-only times are separate.",
        "CUDA comparisons require a matching prototype CPU canonical-result checksum.",
        "Loop counts may differ by backend; all rates use graphs * loops.",
        f"CUDA: {report['cuda_status']}",
        "",
        f"{'dataset':<22} {'backend':<10} {'status':<17} {'loops':>7} {'ms/trial':>10} {'us/graph':>11} {'vs proto':>9}",
    ]
    for row in report["measurements"]:
        milliseconds = row.get("median_seconds")
        if milliseconds is not None:
            milliseconds *= 1000
        lines.append(f"{row['dataset']:<22} {row['backend']:<10} {row['status']:<17} "
                     f"{str(row.get('loops', '-')):>7} {number(milliseconds):>10} "
                     f"{number(row.get('microseconds_per_graph')):>11} "
                     f"{number(row.get('speedup_vs_prototype_cpu'), 2):>9}")
        if row.get("error"):
            lines.append("  " + row["error"])
        if "kernel_microseconds_per_graph" in row:
            lines.append(f"  CUDA kernel only: {row['kernel_microseconds_per_graph']:.3f} us/graph; "
                         f"transfer-inclusive speedup vs nauty CPU: {number(row.get('speedup_vs_nauty_cpu'), 2)}; "
                         f"vs Traces CPU: {number(row.get('speedup_vs_traces_cpu'), 2)}.")
        if row.get("below_target"):
            lines.append("  Trial duration fell below the calibration target; interpret small differences cautiously.")
    (destination / "summary.txt").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def binary_info(value):
    if not value:
        return None
    path = Path(shutil.which(value) or value).resolve()
    result = {"path": str(path)}
    try:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        result["error"] = str(error)
    return result


def load_average():
    try:
        return list(os.getloadavg())
    except (AttributeError, OSError):
        return None


def source_info():
    paths = {ROOT / "CMakeLists.txt", Path(__file__).resolve(), ROOT / "tests" / "validate.py"}
    for directory in ("src", "include"):
        paths.update(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and path.suffix in (".c", ".cpp", ".cu", ".h", ".hpp", ".cuh"))
    result = {"description": "Current source snapshot; binary hashes identify the measured executables",
              "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(paths)}}
    for name, arguments in (("git_head", ["rev-parse", "HEAD"]),
                            ("git_status", ["status", "--short"])):
        try:
            process = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True,
                                     text=True, timeout=5, check=False)
            if process.returncode == 0:
                result[name] = process.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def build_info(executable):
    if not executable:
        return None
    binary = Path(shutil.which(executable) or executable).resolve()
    build = next((path for path in (binary.parent, binary.parent.parent)
                  if (path / "CMakeCache.txt").is_file()), None)
    if build is None:
        return {"status": "unavailable", "reason": "no CMakeCache.txt beside executable or its parent"}
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        match = re.match(r"([^/#][^:]*):[^=]*=(.*)", line)
        if match:
            key, value = match.groups()
            if key.startswith(("CMAKE_CXX_", "CMAKE_CUDA_", "CMAKE_EXE_LINKER_FLAGS", "CMAKE_OSX_",
                               "CMAKE_GENERATOR", "CMAKE_CACHE_", "NAUTY_")) or key == "CMAKE_BUILD_TYPE":
                if not key.endswith("-ADVANCED"):
                    cache[key] = value
    compiler = {}
    for language in ("CXX", "CUDA"):
        for path in sorted((build / "CMakeFiles").glob(f"*/CMake{language}Compiler.cmake")):
            for key in ("COMPILER", "COMPILER_ID", "COMPILER_VERSION", "COMPILER_TARGET", "STANDARD_COMPUTED_DEFAULT"):
                name = f"CMAKE_{language}_{key}"
                match = re.search(r"set\(" + name + r' "([^"\n]*)"\)', path.read_text())
                if match:
                    compiler[name] = match.group(1)
    flags = build / "CMakeFiles" / (binary.name + ".dir") / "flags.make"
    result = {"status": "available", "directory": str(build), "cmake_cache": cache,
              "compiler": compiler}
    if flags.is_file():
        result["target_flags"] = flags.read_text()
    commands = build / "compile_commands.json"
    if commands.is_file():
        try:
            result["compile_commands"] = [entry for entry in json.loads(commands.read_text())
                                          if binary.name + ".dir" in entry.get("command", "")
                                          or binary.name + ".dir" in " ".join(entry.get("arguments", []))]
        except (ValueError, TypeError):
            result["compile_commands_error"] = "could not parse compile_commands.json"
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", default=str(ROOT / "build" / "nauty-benchmark-cpu"))
    parser.add_argument("--cuda", help="optional CUDA timing executable; no GPU estimate is made when omitted")
    parser.add_argument("--geng", default=find_tool("geng"))
    parser.add_argument("--output-dir", type=Path, help="default: benchmarks/results/<UTC timestamp>")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--backends", nargs="+", choices=("prototype", "nauty", "traces"),
                        default=["prototype", "nauty", "traces"])
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--random-count", type=int, default=1024)
    parser.add_argument("--symmetric-count", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--target-seconds", type=float, default=0.1)
    parser.add_argument("--max-loops", type=int, default=100000)
    parser.add_argument("--timeout", type=float, default=20, help="maximum seconds per subprocess")
    parser.add_argument("--max-total-seconds", type=float, default=110,
                        help="overall timing budget, including calibration (default: 110)")
    options = parser.parse_args()
    if options.cuda and "prototype" not in options.backends:
        parser.error("--cuda requires prototype in --backends for CPU/CUDA result verification")
    for name in ("random_count", "symmetric_count", "repeats", "max_nodes", "batch_size", "max_loops"):
        if getattr(options, name) < 1:
            parser.error(name.replace("_", "-") + " must be positive")
    if options.warmup < 1:
        parser.error("warmup must be at least 1")
    for name in ("target_seconds", "timeout", "max_total_seconds"):
        if not math.isfinite(getattr(options, name)) or getattr(options, name) <= 0:
            parser.error(name.replace("_", "-") + " must be finite and positive")
    return options


def main():
    options = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = options.output_dir or ROOT / "benchmarks" / "results" / timestamp
    destination.mkdir(parents=True, exist_ok=False)
    corpus_dir = destination / "corpora"
    corpus_dir.mkdir()
    report = {
        "schema_version": 1, "created_utc": timestamp,
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "logical_cpus": os.cpu_count(), "python": sys.version},
        "protocol": {"cpu_threads": 1, "repeats": options.repeats, "warmup": options.warmup,
                     "target_seconds": options.target_seconds, "max_nodes": options.max_nodes,
                     "batch_size": options.batch_size, "max_total_seconds": options.max_total_seconds,
                     "seed": options.seed,
                     "scope": "C++ core timers; no generation, parsing, startup, or JSON I/O",
                     "cuda_primary_scope": "host H2D + kernel + D2H; kernel-only reported separately",
                     "cuda_correctness": "base-corpus FNV-1a checksum of n and canonical rows must match prototype CPU",
                     "normalization": "median trial seconds / (base corpus graphs * loops)"},
        "binaries": {"cpu": binary_info(options.cpu), "cuda": binary_info(options.cuda),
                     "geng": binary_info(options.geng)},
        "sources": source_info(),
        "builds": {"cpu": build_info(options.cpu), "cuda": build_info(options.cuda)},
        "corpora": [], "measurements": [],
        "cuda_status": "requested, not measured yet" if options.cuda else "NOT MEASURED (no --cuda executable supplied)",
    }
    print(f"Results: {destination}", flush=True)
    print("CPU measurements are single-threaded; other CPU/GPU workloads should be idle.", flush=True)
    if not options.cuda:
        print("CUDA NOT MEASURED: no --cuda executable supplied; no GPU speedup will be estimated.", flush=True)
    helper = helpers()
    corpora = []
    for name in dict.fromkeys(options.datasets):
        payload, manifest = generate_corpus(name, options, helper)
        path = corpus_dir / (name + ".g6")
        path.write_text(payload, encoding="ascii")
        manifest["file"] = str(path.relative_to(destination))
        report["corpora"].append(manifest)
        corpora.append((payload, manifest))
    (destination / "corpora.json").write_text(json.dumps(report["corpora"], indent=2) + "\n")
    report["host"]["load_average_before"] = load_average()
    started = time.perf_counter()
    deadline = started + options.max_total_seconds
    backends = list(dict.fromkeys(options.backends)) + (["cuda"] if options.cuda else [])
    for payload, manifest in corpora:
        for backend in backends:
            print(f"Timing {manifest['name']} / {backend} ...", flush=True)
            reference = next((row["checksum"] for row in report["measurements"]
                              if row["dataset"] == manifest["name"] and row["backend"] == "prototype"
                              and row["status"] == "ok"), None)
            measurement = measure(backend, payload, manifest, options, deadline, reference)
            report["measurements"].append(measurement)
            print(f"  {measurement['status']}: "
                  + (f"{measurement['microseconds_per_graph']:.3f} us/graph, loops={measurement['loops']}"
                     if measurement["status"] == "ok" else measurement.get("error", "failed")), flush=True)
            if options.cuda:
                cuda_rows = [row for row in report["measurements"] if row["backend"] == "cuda"]
                report["cuda_status"] = (
                    "INCOMPLETE OR UNAVAILABLE" if any(row["status"] != "ok" for row in cuda_rows)
                    else "MEASURED" if len(cuda_rows) == len(corpora)
                    else "requested, measurements pending"
                )
            report["elapsed_timing_seconds"] = time.perf_counter() - started
            write_results(report, destination)
    report["host"]["load_average_after"] = load_average()
    print("\n" + write_results(report, destination), flush=True)
    return int(any(row["status"] != "ok" for row in report["measurements"]))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, AssertionError, subprocess.SubprocessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
