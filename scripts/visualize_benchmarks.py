#!/usr/bin/env python3
"""
Generate visualization comparing PyTorch vs Triton benchmark results.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_results(pytorch_file: str, triton_file: str):
    """Load benchmark results from JSON files."""
    with open(pytorch_file) as f:
        pytorch_data = json.load(f)
    with open(triton_file) as f:
        triton_data = json.load(f)
    return pytorch_data, triton_data


def create_comparison_charts(pytorch_data, triton_data, output_dir: Path):
    """Create comparison visualizations."""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PyTorch vs Triton Inference Server\nOpenCLIP ViT-B-32 on RTX A4000', 
                 fontsize=16, fontweight='bold')
    
    # 1. Single-image latency comparison
    ax1 = axes[0, 0]
    metrics = ['Mean', 'p50', 'p95', 'p99']
    pytorch_single = [
        pytorch_data['pytorch']['single']['mean_ms'],
        pytorch_data['pytorch']['single']['p50_ms'],
        pytorch_data['pytorch']['single']['p95_ms'],
        pytorch_data['pytorch']['single']['p99_ms'],
    ]
    triton_single = [
        triton_data['triton']['single']['mean_ms'],
        triton_data['triton']['single']['p50_ms'],
        triton_data['triton']['single']['p95_ms'],
        triton_data['triton']['single']['p99_ms'],
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, pytorch_single, width, label='PyTorch', color='#2E86AB')
    bars2 = ax1.bar(x + width/2, triton_single, width, label='Triton', color='#A23B72')
    
    ax1.set_ylabel('Latency (ms)', fontweight='bold')
    ax1.set_title('Single-Image Latency', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
    
    # 2. Batch throughput comparison
    ax2 = axes[0, 1]
    batch_sizes = [4, 8, 16, 32]
    pytorch_throughput = []
    triton_throughput = []
    
    for bs in batch_sizes:
        key = f'batch_{bs}'
        pytorch_mean = pytorch_data['pytorch'][key]['mean_ms']
        triton_mean = triton_data['triton'][key]['mean_ms']
        pytorch_throughput.append(bs / (pytorch_mean / 1000))
        triton_throughput.append(bs / (triton_mean / 1000))
    
    x = np.arange(len(batch_sizes))
    bars1 = ax2.bar(x - width/2, pytorch_throughput, width, label='PyTorch', color='#2E86AB')
    bars2 = ax2.bar(x + width/2, triton_throughput, width, label='Triton', color='#A23B72')
    
    ax2.set_ylabel('Images/second', fontweight='bold')
    ax2.set_title('Batch Processing Throughput', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Batch {bs}' for bs in batch_sizes])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    # 3. Concurrent throughput
    ax3 = axes[1, 0]
    backends = ['PyTorch', 'Triton']
    throughput = [
        pytorch_data['pytorch']['concurrent']['throughput_rps'],
        triton_data['triton']['concurrent']['throughput_rps']
    ]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax3.bar(backends, throughput, color=colors, width=0.6)
    ax3.set_ylabel('Requests/second', fontweight='bold')
    ax3.set_title('Concurrent Throughput (16 workers, 200 requests)', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels and speedup
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}\nreq/s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add speedup annotation
    speedup = throughput[0] / throughput[1]
    ax3.text(0.5, max(throughput) * 0.5, 
             f'PyTorch is\n{speedup:.1f}x faster',
             ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Cost comparison
    ax4 = axes[1, 1]
    # For 10,000 images
    pytorch_time_min = 10000 / pytorch_data['pytorch']['concurrent']['throughput_rps'] / 60
    triton_time_min = 10000 / triton_data['triton']['concurrent']['throughput_rps'] / 60
    
    gpu_cost_per_hour = 0.20  # $0.20/hour for RTX 3070
    pytorch_cost = (pytorch_time_min / 60) * gpu_cost_per_hour
    triton_cost = (triton_time_min / 60) * gpu_cost_per_hour
    
    backends = ['PyTorch', 'Triton']
    costs = [pytorch_cost, triton_cost]
    times = [pytorch_time_min, triton_time_min]
    
    bars = ax4.bar(backends, costs, color=colors, width=0.6)
    ax4.set_ylabel('Cost (USD)', fontweight='bold')
    ax4.set_title('Cost to Process 10,000 Images\n(@ $0.20/hour GPU)', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.3f}\n({times[i]:.1f} min)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add savings annotation
    savings = (1 - pytorch_cost / triton_cost) * 100
    ax4.text(0.5, max(costs) * 0.5,
             f'Save {savings:.0f}%\nwith PyTorch',
             ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'comparison_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved to: {output_file}")
    
    # Also save as PDF
    output_pdf = output_dir / 'comparison_chart.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ PDF saved to: {output_pdf}")
    
    plt.close()


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'benchmark_results'
    
    # Find the most recent result files
    pytorch_file = results_dir / 'pytorch_rtxa4000_20260214_090742.json'
    triton_file = results_dir / 'triton_rtxa4000_20260214_083702.json'
    
    if not pytorch_file.exists() or not triton_file.exists():
        print("✗ Benchmark result files not found!")
        print(f"  Looking for: {pytorch_file}")
        print(f"  Looking for: {triton_file}")
        return 1
    
    print("Loading benchmark results...")
    pytorch_data, triton_data = load_results(str(pytorch_file), str(triton_file))
    
    print("Creating comparison charts...")
    create_comparison_charts(pytorch_data, triton_data, results_dir)
    
    print("\n" + "="*60)
    print("✓ Visualization complete!")
    print("="*60)
    print(f"\nView results:")
    print(f"  - Image: {results_dir}/comparison_chart.png")
    print(f"  - PDF:   {results_dir}/comparison_chart.pdf")
    print(f"  - Data:  {project_root}/BENCHMARK_RESULTS.md")
    print(f"  - Summary: {project_root}/BENCHMARK_SUMMARY.md")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
