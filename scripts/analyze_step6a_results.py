#!/usr/bin/env python3
"""
Step 6A: Analyze and compare benchmark results from remote and local runs.

This script:
  1. Loads both remote and local benchmark JSON files
  2. Generates comparison tables
  3. Creates visualizations (if matplotlib available)
  4. Produces a summary report

Usage:
  python scripts/analyze_step6a_results.py

  # Or specify custom files
  python scripts/analyze_step6a_results.py \
    --remote benchmark_results/step6a_remote_comparison.json \
    --local benchmark_results/step6a_local_comparison.json \
    --output STEP_6A_RESULTS.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not available. Visualizations will be skipped.")
    print("Install with: pip install matplotlib")


# =============================================================================
# Helper Functions
# =============================================================================

def load_json(filepath: Path) -> Optional[Dict]:
    """Load JSON file."""
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def format_table_row(cols, widths):
    """Format a table row with specified column widths."""
    formatted = []
    for col, width in zip(cols, widths):
        if isinstance(col, float):
            formatted.append(f"{col:>{width}.1f}")
        else:
            formatted.append(f"{col:<{width}}")
    return " | ".join(formatted)


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_single_image_latency(remote_data: Dict, local_data: Dict) -> str:
    """Analyze single-image latency across backends and locations."""
    output = []
    output.append("## Single-Image Latency Comparison")
    output.append("")
    
    # Remote results
    if remote_data:
        output.append("### Remote (from Mac, includes network latency)")
        output.append("")
        output.append("| Backend | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) |")
        output.append("|---------|-----------|-------------|----------|----------|----------|----------|")
        
        for backend_name, backend_data in remote_data.get("backends", {}).items():
            if "single_image" in backend_data:
                si = backend_data["single_image"]
                if "client_mean_ms" in si:
                    output.append(
                        f"| {backend_name:<15} | "
                        f"{si['client_mean_ms']:>9.1f} | "
                        f"{si.get('client_median_ms', 0):>11.1f} | "
                        f"{si.get('client_p95_ms', 0):>8.1f} | "
                        f"{si.get('client_p99_ms', 0):>8.1f} | "
                        f"{si.get('client_min_ms', 0):>8.1f} | "
                        f"{si.get('client_max_ms', 0):>8.1f} |"
                    )
        output.append("")
    
    # Local results
    if local_data:
        output.append("### Local (from within instance, minimal overhead)")
        output.append("")
        output.append("| Backend | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) |")
        output.append("|---------|-----------|-------------|----------|----------|----------|----------|")
        
        for backend_name, backend_data in local_data.get("backends", {}).items():
            if "single_image" in backend_data:
                si = backend_data["single_image"]
                if "client_mean_ms" in si:
                    output.append(
                        f"| {backend_name:<15} | "
                        f"{si['client_mean_ms']:>9.2f} | "
                        f"{si.get('client_median_ms', 0):>11.2f} | "
                        f"{si.get('client_p95_ms', 0):>8.2f} | "
                        f"{si.get('client_p99_ms', 0):>8.2f} | "
                        f"{si.get('client_min_ms', 0):>8.2f} | "
                        f"{si.get('client_max_ms', 0):>8.2f} |"
                    )
        output.append("")
    
    # Calculate network overhead
    if remote_data and local_data:
        output.append("### Network Overhead (Remote - Local)")
        output.append("")
        output.append("| Backend | Remote Mean (ms) | Local Mean (ms) | Network Overhead (ms) |")
        output.append("|---------|------------------|-----------------|----------------------|")
        
        for backend_name in remote_data.get("backends", {}).keys():
            remote_si = remote_data["backends"][backend_name].get("single_image", {})
            local_si = local_data["backends"].get(backend_name, {}).get("single_image", {})
            
            if "client_mean_ms" in remote_si and "client_mean_ms" in local_si:
                remote_mean = remote_si["client_mean_ms"]
                local_mean = local_si["client_mean_ms"]
                overhead = remote_mean - local_mean
                
                output.append(
                    f"| {backend_name:<15} | "
                    f"{remote_mean:>16.1f} | "
                    f"{local_mean:>15.2f} | "
                    f"{overhead:>20.1f} |"
                )
        output.append("")
    
    return "\n".join(output)


def analyze_server_side_metrics(remote_data: Dict, local_data: Dict) -> str:
    """Analyze server-side GPU compute time."""
    output = []
    output.append("## Server-Side GPU Compute Time")
    output.append("")
    output.append("*From Triton `/metrics` endpoint - pure GPU compute, no network overhead*")
    output.append("")
    
    # Try local first (more reliable), fall back to remote
    data = local_data or remote_data
    
    if not data:
        output.append("No data available.")
        return "\n".join(output)
    
    output.append("| Backend | GPU Compute (ms) | Total Request (ms) | Overhead (ms) |")
    output.append("|---------|------------------|--------------------|-----------------|")
    
    onnx_compute = None
    results = []
    
    for backend_name, backend_data in data.get("backends", {}).items():
        if "single_image" in backend_data:
            si = backend_data["single_image"]
            if "server_compute_ms" in si:
                compute = si["server_compute_ms"]
                request = si.get("server_request_ms", 0)
                overhead = request - compute
                
                if "onnx" in backend_name.lower() and "trt" not in backend_name.lower():
                    onnx_compute = compute
                
                results.append((backend_name, compute, request, overhead))
    
    # Sort by compute time
    results.sort(key=lambda x: x[1])
    
    for backend_name, compute, request, overhead in results:
        output.append(
            f"| {backend_name:<15} | "
            f"{compute:>16.2f} | "
            f"{request:>18.2f} | "
            f"{overhead:>15.2f} |"
        )
    
    output.append("")
    
    # Calculate speedups
    if onnx_compute and len(results) > 1:
        output.append("### Speedup vs ONNX CUDA EP")
        output.append("")
        output.append("| Backend | GPU Compute (ms) | Speedup |")
        output.append("|---------|------------------|---------|")
        
        for backend_name, compute, _, _ in results:
            speedup = onnx_compute / compute
            speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "baseline"
            output.append(
                f"| {backend_name:<15} | "
                f"{compute:>16.2f} | "
                f"{speedup_str:>7} |"
            )
        output.append("")
    
    return "\n".join(output)


def analyze_batch_throughput(remote_data: Dict, local_data: Dict) -> str:
    """Analyze batch processing throughput."""
    output = []
    output.append("## Batch Processing Throughput")
    output.append("")
    
    data = local_data or remote_data
    
    if not data:
        output.append("No data available.")
        return "\n".join(output)
    
    # Get all batch sizes
    batch_sizes = set()
    for backend_data in data.get("backends", {}).values():
        if "batch" in backend_data:
            for key in backend_data["batch"].keys():
                if key.startswith("batch_"):
                    batch_sizes.add(int(key.split("_")[1]))
    
    batch_sizes = sorted(batch_sizes)
    
    for batch_size in batch_sizes:
        output.append(f"### Batch Size: {batch_size}")
        output.append("")
        output.append("| Backend | Latency (ms) | Throughput (img/s) |")
        output.append("|---------|--------------|-------------------|")
        
        results = []
        for backend_name, backend_data in data.get("backends", {}).items():
            batch_key = f"batch_{batch_size}"
            if "batch" in backend_data and batch_key in backend_data["batch"]:
                batch_info = backend_data["batch"][batch_key]
                latency = batch_info.get("mean_ms", 0)
                throughput = batch_info.get("throughput_imgs_per_sec", 0)
                results.append((backend_name, latency, throughput))
        
        # Sort by throughput (descending)
        results.sort(key=lambda x: x[2], reverse=True)
        
        for backend_name, latency, throughput in results:
            output.append(
                f"| {backend_name:<15} | "
                f"{latency:>12.1f} | "
                f"{throughput:>17.1f} |"
            )
        
        output.append("")
    
    return "\n".join(output)


def generate_winner_declaration(local_data: Dict) -> str:
    """Declare the winner based on metrics."""
    output = []
    output.append("## Winner Declaration 🏆")
    output.append("")
    
    if not local_data:
        output.append("Insufficient data for winner declaration.")
        return "\n".join(output)
    
    # Collect metrics
    backends = {}
    for backend_name, backend_data in local_data.get("backends", {}).items():
        backends[backend_name] = {
            "client_mean": backend_data.get("single_image", {}).get("client_mean_ms", float('inf')),
            "server_compute": backend_data.get("single_image", {}).get("server_compute_ms", float('inf')),
            "batch32_throughput": backend_data.get("batch", {}).get("batch_32", {}).get("throughput_imgs_per_sec", 0),
        }
    
    # Find winners
    best_latency = min(backends.items(), key=lambda x: x[1]["client_mean"])
    best_compute = min(backends.items(), key=lambda x: x[1]["server_compute"])
    best_throughput = max(backends.items(), key=lambda x: x[1]["batch32_throughput"])
    
    output.append("### Category Winners")
    output.append("")
    output.append(f"- **Lowest Latency (single-image):** {best_latency[0]} ({best_latency[1]['client_mean']:.2f}ms)")
    output.append(f"- **Fastest GPU Compute:** {best_compute[0]} ({best_compute[1]['server_compute']:.2f}ms)")
    output.append(f"- **Highest Throughput (batch-32):** {best_throughput[0]} ({best_throughput[1]['batch32_throughput']:.1f} img/s)")
    output.append("")
    
    # Overall recommendation
    output.append("### Overall Recommendation")
    output.append("")
    
    # Count wins
    winner_counts = {}
    for backend_name in backends.keys():
        wins = 0
        if backend_name == best_latency[0]:
            wins += 1
        if backend_name == best_compute[0]:
            wins += 1
        if backend_name == best_throughput[0]:
            wins += 1
        winner_counts[backend_name] = wins
    
    overall_winner = max(winner_counts.items(), key=lambda x: x[1])
    
    if overall_winner[1] >= 2:
        output.append(f"**Winner: {overall_winner[0]}** (wins {overall_winner[1]}/3 categories)")
        output.append("")
        output.append("This backend offers the best overall performance for production deployment.")
    else:
        output.append("**Tie:** No clear winner across all categories.")
        output.append("")
        output.append("**Recommendation:** Choose based on your workload:")
        output.append(f"- Single-image latency priority: {best_latency[0]}")
        output.append(f"- Batch processing priority: {best_throughput[0]}")
    
    output.append("")
    
    return "\n".join(output)


def create_visualizations(local_data: Dict, output_dir: Path):
    """Create comparison visualizations."""
    if not MATPLOTLIB_AVAILABLE or not local_data:
        return
    
    backends = local_data.get("backends", {})
    if not backends:
        return
    
    backend_names = list(backends.keys())
    
    # Figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Step 6A: Backend Comparison', fontsize=16, fontweight='bold')
    
    # 1. Single-image latency
    latencies = []
    for name in backend_names:
        latency = backends[name].get("single_image", {}).get("client_mean_ms", 0)
        latencies.append(latency)
    
    axes[0].bar(backend_names, latencies, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('Single-Image Latency\n(Lower is Better)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Server-side GPU compute
    compute_times = []
    for name in backend_names:
        compute = backends[name].get("single_image", {}).get("server_compute_ms", 0)
        compute_times.append(compute if compute > 0 else None)
    
    # Filter out None values
    valid_compute = [(name, compute) for name, compute in zip(backend_names, compute_times) if compute is not None]
    if valid_compute:
        names, times = zip(*valid_compute)
        axes[1].bar(names, times, color=['#e74c3c', '#2ecc71'])
        axes[1].set_ylabel('GPU Compute Time (ms)')
        axes[1].set_title('Server-Side GPU Compute\n(Lower is Better)')
        axes[1].tick_params(axis='x', rotation=45)
    
    # 3. Batch-32 throughput
    throughputs = []
    for name in backend_names:
        throughput = backends[name].get("batch", {}).get("batch_32", {}).get("throughput_imgs_per_sec", 0)
        throughputs.append(throughput)
    
    axes[2].bar(backend_names, throughputs, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[2].set_ylabel('Throughput (images/sec)')
    axes[2].set_title('Batch-32 Throughput\n(Higher is Better)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / "step6a_comparison_chart.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")


# =============================================================================
# Main Report Generator
# =============================================================================

def generate_report(remote_data: Dict, local_data: Dict, output_path: Path):
    """Generate comprehensive comparison report."""
    
    output = []
    
    # Header
    output.append("# Step 6A: 3-Way Backend Comparison Results")
    output.append("")
    output.append("**Backends Tested:**")
    output.append("1. PyTorch FastAPI (Step 3)")
    output.append("2. Triton ONNX CUDA EP (Step 4/5A)")
    output.append("3. Triton TensorRT EP (Step 5B)")
    output.append("")
    
    # Configuration
    data = local_data or remote_data
    if data and "config" in data:
        config = data["config"]
        output.append("**Configuration:**")
        output.append(f"- Location: {config.get('location', config.get('host', 'Unknown'))}")
        output.append(f"- Iterations: {config.get('iterations', 'N/A')}")
        output.append(f"- Batch sizes: {config.get('batch_sizes', 'N/A')}")
        output.append(f"- Timestamp: {config.get('timestamp', 'N/A')}")
        output.append("")
    
    output.append("---")
    output.append("")
    
    # Winner declaration (use local data if available)
    if local_data:
        output.append(generate_winner_declaration(local_data))
        output.append("---")
        output.append("")
    
    # Detailed analysis
    output.append(analyze_single_image_latency(remote_data, local_data))
    output.append("---")
    output.append("")
    
    output.append(analyze_server_side_metrics(remote_data, local_data))
    output.append("---")
    output.append("")
    
    output.append(analyze_batch_throughput(remote_data, local_data))
    output.append("---")
    output.append("")
    
    # Key findings
    output.append("## Key Findings")
    output.append("")
    output.append("### Network Latency Impact")
    if remote_data and local_data:
        output.append("- Network overhead from Mac → Vast.ai: ~200-400ms")
        output.append("- This affects all backends equally, so relative comparison is valid")
        output.append("- Local benchmarks show true backend performance")
    else:
        output.append("- Run both remote and local benchmarks for complete analysis")
    output.append("")
    
    output.append("### Backend Characteristics")
    output.append("")
    output.append("**PyTorch FastAPI:**")
    output.append("- Simplest deployment")
    output.append("- No special configuration needed")
    output.append("- Good for single-GPU scenarios")
    output.append("")
    output.append("**Triton ONNX CUDA EP:**")
    output.append("- Production-ready features (versioning, metrics)")
    output.append("- Dynamic batching support")
    output.append("- Baseline ONNX performance")
    output.append("")
    output.append("**Triton TensorRT EP:**")
    output.append("- Best GPU utilization (FP16, kernel fusion)")
    output.append("- 2-5 min first-load compilation")
    output.append("- Cached engines for subsequent starts")
    output.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write("\n".join(output))
    
    print(f"\nReport saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze Step 6A benchmark results")
    parser.add_argument("--remote", type=str, 
                       default="benchmark_results/step6a_remote_comparison.json",
                       help="Path to remote benchmark JSON")
    parser.add_argument("--local", type=str,
                       default="benchmark_results/step6a_local_comparison.json",
                       help="Path to local benchmark JSON")
    parser.add_argument("--output", type=str,
                       default="STEP_6A_RESULTS.md",
                       help="Output markdown file")
    parser.add_argument("--visualizations", action="store_true",
                       help="Generate visualization charts (requires matplotlib)")
    
    args = parser.parse_args()
    
    # Load data
    remote_path = Path(args.remote)
    local_path = Path(args.local)
    output_path = Path(args.output)
    
    print("="*70)
    print("STEP 6A: Results Analysis")
    print("="*70)
    print()
    
    remote_data = load_json(remote_path)
    local_data = load_json(local_path)
    
    if not remote_data and not local_data:
        print("ERROR: No benchmark data found.")
        print(f"Expected files:")
        print(f"  - {remote_path}")
        print(f"  - {local_path}")
        print()
        print("Run benchmarks first:")
        print("  python scripts/benchmark_all_three.py ...")
        print("  python scripts/benchmark_all_three_local.py ...")
        sys.exit(1)
    
    if remote_data:
        print(f"✓ Loaded remote results: {remote_path}")
    if local_data:
        print(f"✓ Loaded local results: {local_path}")
    
    print()
    
    # Generate report
    generate_report(remote_data, local_data, output_path)
    
    # Generate visualizations
    if args.visualizations or MATPLOTLIB_AVAILABLE:
        if local_data:
            create_visualizations(local_data, output_path.parent)
    
    print()
    print("="*70)
    print("Analysis complete!")
    print("="*70)
    print()
    print(f"Review results in: {output_path}")


if __name__ == "__main__":
    main()
