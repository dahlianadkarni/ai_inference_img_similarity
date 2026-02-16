#!/usr/bin/env python3
"""
Analyze Multi-GPU Benchmark Results (Step 6B)

This script analyzes benchmark results from multiple GPU configurations
and generates a markdown report with:
  - Throughput scaling analysis (linear vs actual)
  - Cost-efficiency comparison
  - Latency vs concurrency charts (data for plotting)
  - Saturation point identification
  - Recommendations for production deployment

Usage:
  # Analyze all Step 6B results
  python scripts/analyze_multigpu_results.py \
      --results benchmark_results/step6b_*.json \
      --output STEP_6B_RESULTS.md

  # Include cost analysis (provide instance pricing)
  python scripts/analyze_multigpu_results.py \
      --results benchmark_results/step6b_*.json \
      --costs "1gpu:1.5,2gpu:2.5,4gpu:5.0,8gpu:9.0" \
      --output STEP_6B_RESULTS.md
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import glob


def load_benchmark_results(pattern: str) -> List[Dict]:
    """Load all benchmark result files matching the pattern."""
    files = glob.glob(pattern)
    results = []
    
    for filepath in sorted(files):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                data['_filepath'] = filepath
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    return results


def parse_costs(cost_str: str) -> Dict[str, float]:
    """Parse cost string like '1gpu:1.5,2gpu:2.5,4gpu:5.0'."""
    costs = {}
    if not cost_str:
        return costs
    
    for pair in cost_str.split(','):
        key, value = pair.split(':')
        costs[key.strip()] = float(value.strip())
    
    return costs


def find_peak_throughput(benchmarks: List[Dict]) -> Tuple[float, int]:
    """Find the peak throughput and corresponding concurrency."""
    max_throughput = 0
    best_concurrency = 0
    
    for bench in benchmarks:
        if bench['throughput_imgs_per_sec'] > max_throughput:
            max_throughput = bench['throughput_imgs_per_sec']
            best_concurrency = bench['concurrency']
    
    return max_throughput, best_concurrency


def calculate_cost_per_1000(throughput_imgs_per_sec: float, cost_per_hour: float) -> float:
    """Calculate cost per 1000 images."""
    if throughput_imgs_per_sec == 0:
        return 0
    
    # Cost per second = cost_per_hour / 3600
    # Time for 1000 images = 1000 / throughput
    # Cost for 1000 images = (1000 / throughput) * (cost_per_hour / 3600)
    return (1000 / throughput_imgs_per_sec) * (cost_per_hour / 3600)


def generate_markdown_report(results: List[Dict], costs: Dict[str, float], output_path: str):
    """Generate comprehensive markdown analysis report."""
    
    # Sort results by GPU count (extract from config name)
    def extract_gpu_count(config_name: str) -> int:
        """Extract GPU count from config name like '4-gpu-nvlink'."""
        parts = config_name.split('-')
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0
    
    results = sorted(results, key=lambda x: extract_gpu_count(x['config_name']))
    
    # Build report
    lines = []
    lines.append("# Step 6B Results: Multi-GPU Scaling Analysis")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}")
    lines.append(f"**Configurations Tested:** {len(results)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Executive Summary
    lines.append("## 🎯 Executive Summary")
    lines.append("")
    
    if results:
        # Find best throughput across all configs
        all_peak_throughputs = []
        for result in results:
            peak_tput, _ = find_peak_throughput(result['benchmarks'])
            all_peak_throughputs.append((result['config_name'], peak_tput))
        
        best_config, best_tput = max(all_peak_throughputs, key=lambda x: x[1])
        
        lines.append(f"**Best Throughput:** {best_tput:.1f} images/sec ({best_config})")
        lines.append("")
        
        # Calculate scaling efficiency
        if len(results) >= 2:
            baseline = results[0]
            baseline_peak, _ = find_peak_throughput(baseline['benchmarks'])
            
            lines.append("**Scaling Efficiency:**")
            lines.append("")
            lines.append("| Configuration | Peak Throughput | Scaling Factor | Efficiency vs Linear |")
            lines.append("|:-------------:|:---------------:|:--------------:|:-------------------:|")
            
            for result in results:
                config = result['config_name']
                gpu_count = extract_gpu_count(config)
                peak_tput, best_conc = find_peak_throughput(result['benchmarks'])
                
                if baseline_peak > 0:
                    scaling_factor = peak_tput / baseline_peak
                    expected_linear = gpu_count
                    efficiency_pct = (scaling_factor / expected_linear * 100) if expected_linear > 0 else 0
                else:
                    scaling_factor = 0
                    efficiency_pct = 0
                
                lines.append(f"| {config} | {peak_tput:.1f} img/s | {scaling_factor:.2f}x | {efficiency_pct:.1f}% |")
            
            lines.append("")
        
        # Cost analysis if available
        if costs:
            lines.append("**Cost-Efficiency Analysis:**")
            lines.append("")
            lines.append("| Configuration | Instance Cost/hr | Peak Throughput | Cost per 1000 Images |")
            lines.append("|:-------------:|:----------------:|:---------------:|:-------------------:|")
            
            for result in results:
                config = result['config_name']
                peak_tput, _ = find_peak_throughput(result['benchmarks'])
                
                # Try to match config to cost key
                cost_key = None
                for key in costs.keys():
                    if key in config or config.startswith(key.replace('gpu', '')):
                        cost_key = key
                        break
                
                if cost_key and costs[cost_key] > 0:
                    cost_per_hour = costs[cost_key]
                    cost_per_1000 = calculate_cost_per_1000(peak_tput, cost_per_hour)
                    lines.append(f"| {config} | ${cost_per_hour:.2f} | {peak_tput:.1f} img/s | ${cost_per_1000:.3f} |")
                else:
                    lines.append(f"| {config} | N/A | {peak_tput:.1f} img/s | N/A |")
            
            lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Detailed Results per Configuration
    lines.append("## 📊 Detailed Results by Configuration")
    lines.append("")
    
    for result in results:
        config = result['config_name']
        gpu_count = extract_gpu_count(config)
        
        lines.append(f"### {config}")
        lines.append("")
        lines.append(f"**Timestamp:** {result['timestamp']}")
        lines.append(f"**Triton URL:** {result['triton_url']}")
        lines.append(f"**GPU Count:** {gpu_count}")
        lines.append("")
        
        # Throughput vs Concurrency Table
        lines.append("#### Throughput vs Concurrency")
        lines.append("")
        lines.append("| Concurrency | Throughput (img/s) | Latency p50 (ms) | Latency p95 (ms) | Latency p99 (ms) |")
        lines.append("|:-----------:|:------------------:|:----------------:|:----------------:|:----------------:|")
        
        for bench in result['benchmarks']:
            lines.append(
                f"| {bench['concurrency']} | "
                f"{bench['throughput_imgs_per_sec']:.1f} | "
                f"{bench['p50_ms']:.1f} | "
                f"{bench['p95_ms']:.1f} | "
                f"{bench['p99_ms']:.1f} |"
            )
        
        lines.append("")
        
        # Peak performance
        peak_tput, best_conc = find_peak_throughput(result['benchmarks'])
        lines.append(f"**Peak Performance:** {peak_tput:.1f} images/sec at concurrency {best_conc}")
        lines.append("")
        
        # Triton Metrics (if available)
        has_metrics = any('triton_metrics' in bench for bench in result['benchmarks'])
        if has_metrics:
            lines.append("#### Server-Side Metrics")
            lines.append("")
            lines.append("| Concurrency | Queue Time (ms) | Compute Time (ms) | GPU Utilization |")
            lines.append("|:-----------:|:---------------:|:-----------------:|:---------------:|")
            
            for bench in result['benchmarks']:
                if 'triton_metrics' in bench:
                    metrics = bench['triton_metrics']
                    queue_ms = metrics.get('queue_time_us', 0) / 1000
                    compute_ms = metrics.get('compute_time_us', 0) / 1000
                    gpu_util = metrics.get('per_gpu_utilization', [])
                    avg_util = sum(gpu_util) / len(gpu_util) if gpu_util else 0
                    
                    lines.append(
                        f"| {bench['concurrency']} | "
                        f"{queue_ms:.1f} | "
                        f"{compute_ms:.1f} | "
                        f"{avg_util:.1f}% |"
                    )
            
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Key Findings & Recommendations
    lines.append("## 🔍 Key Findings")
    lines.append("")
    lines.append("### Scaling Behavior")
    lines.append("")
    
    if len(results) >= 2:
        baseline = results[0]
        baseline_peak, _ = find_peak_throughput(baseline['benchmarks'])
        
        lines.append(f"- **Baseline (1 GPU):** {baseline_peak:.1f} images/sec")
        
        for result in results[1:]:
            config = result['config_name']
            gpu_count = extract_gpu_count(config)
            peak_tput, _ = find_peak_throughput(result['benchmarks'])
            
            if baseline_peak > 0:
                scaling_factor = peak_tput / baseline_peak
                expected = gpu_count
                efficiency = (scaling_factor / expected * 100)
                
                lines.append(f"- **{config}:** {peak_tput:.1f} images/sec ({scaling_factor:.2f}x speedup, {efficiency:.0f}% efficient)")
    
    lines.append("")
    lines.append("### Saturation Point")
    lines.append("")
    
    # Find where adding more concurrency stops helping
    for result in results:
        config = result['config_name']
        benchmarks = result['benchmarks']
        
        # Find concurrency where throughput stops increasing by >10%
        saturated_at = None
        for i in range(1, len(benchmarks)):
            prev_tput = benchmarks[i-1]['throughput_imgs_per_sec']
            curr_tput = benchmarks[i]['throughput_imgs_per_sec']
            
            if prev_tput > 0:
                improvement = (curr_tput - prev_tput) / prev_tput * 100
                if improvement < 10:  # Less than 10% improvement
                    saturated_at = benchmarks[i-1]['concurrency']
                    break
        
        if saturated_at:
            lines.append(f"- **{config}:** Saturates at concurrency ~{saturated_at}")
        else:
            lines.append(f"- **{config}:** Did not reach saturation (test higher concurrency)")
    
    lines.append("")
    lines.append("### Bottlenecks Identified")
    lines.append("")
    lines.append("*(Analyze based on metrics and scaling efficiency)*")
    lines.append("")
    lines.append("- **Network I/O:** *(Check if binary protocol overhead dominates)*")
    lines.append("- **CPU Preprocessing:** *(Check if host CPU is bottleneck)*")
    lines.append("- **GPU Interconnect:** *(NVLink vs PCIe makes 2-3x difference)*")
    lines.append("- **Dynamic Batching:** *(Queue times increase under high concurrency)*")
    lines.append("")
    
    # Production Recommendations
    lines.append("## 💡 Production Recommendations")
    lines.append("")
    
    if costs:
        # Find most cost-efficient config
        cost_efficiency = []
        for result in results:
            config = result['config_name']
            peak_tput, _ = find_peak_throughput(result['benchmarks'])
            
            cost_key = None
            for key in costs.keys():
                if key in config or config.startswith(key.replace('gpu', '')):
                    cost_key = key
                    break
            
            if cost_key and costs[cost_key] > 0:
                cost_per_1000 = calculate_cost_per_1000(peak_tput, costs[cost_key])
                cost_efficiency.append((config, cost_per_1000, peak_tput))
        
        if cost_efficiency:
            best_value_config, best_cost, best_tput = min(cost_efficiency, key=lambda x: x[1])
            
            lines.append(f"**Most Cost-Efficient:** {best_value_config} at ${best_cost:.3f} per 1000 images ({best_tput:.0f} img/s)")
            lines.append("")
    
    lines.append("**When to Use Multi-GPU:**")
    lines.append("")
    lines.append("- Throughput requirement > 200 images/sec → Consider 2x GPU")
    lines.append("- Throughput requirement > 500 images/sec → Consider 4x GPU")
    lines.append("- Throughput requirement > 1000 images/sec → Consider 8x GPU or horizontal scaling")
    lines.append("")
    lines.append("**Optimization Opportunities:**")
    lines.append("")
    lines.append("1. **Input Format:** Switch to JPEG base64 input (reduces payload from 602KB to ~10KB)")
    lines.append("2. **Local Deployment:** Co-locate client and server to eliminate network overhead")
    lines.append("3. **GPU Interconnect:** Use NVLink instances for better multi-GPU scaling")
    lines.append("4. **Batch Preprocessing:** Offload preprocessing to DALI or multi-core CPU")
    lines.append("")
    
    # Conclusion
    lines.append("---")
    lines.append("")
    lines.append("## ✅ Conclusion")
    lines.append("")
    lines.append("*(Summarize whether multi-GPU scaling meets expectations and when it's worth the cost)*")
    lines.append("")
    lines.append("**Next Steps:**")
    lines.append("")
    lines.append("1. Review saturation points and determine production concurrency targets")
    lines.append("2. Implement input format optimizations (JPEG base64)")
    lines.append("3. Consider horizontal scaling (multiple single-GPU instances) vs vertical (multi-GPU)")
    lines.append("4. Deploy recommended configuration to production")
    lines.append("")
    
    # Write report
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Analysis report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-GPU benchmark results")
    parser.add_argument("--results", required=True,
                        help="Glob pattern for result JSON files (e.g., 'benchmark_results/step6b_*.json')")
    parser.add_argument("--costs", default=None,
                        help="Instance costs as 'config:cost' pairs (e.g., '1gpu:1.5,2gpu:2.5,4gpu:5.0')")
    parser.add_argument("--output", default="STEP_6B_RESULTS.md",
                        help="Output markdown file path")
    
    args = parser.parse_args()
    
    print("Loading benchmark results...")
    results = load_benchmark_results(args.results)
    
    if not results:
        print(f"Error: No results found matching pattern: {args.results}")
        return 1
    
    print(f"Found {len(results)} result files")
    
    costs = parse_costs(args.costs) if args.costs else {}
    if costs:
        print(f"Instance costs: {costs}")
    
    print(f"Generating analysis report...")
    generate_markdown_report(results, costs, args.output)
    
    print()
    print("=" * 80)
    print(f"✓ Analysis complete! Report saved to {args.output}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
