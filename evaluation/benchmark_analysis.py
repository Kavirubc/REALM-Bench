"""
Analysis and Visualization for Compensation Benchmark Results

This module provides analysis tools for benchmark results, including:
- Statistical comparisons
- Visualization generation
- Performance analysis
"""

import json
import os
from typing import Dict, List, Any, Optional
import statistics
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib/pandas not available. Visualization disabled.")


class BenchmarkAnalyzer:
    """Analyze benchmark results and generate comparisons"""
    
    def __init__(self, results_file: str = None):
        self.results_file = results_file
        self.data = None
        if results_file:
            self.load_results(results_file)
    
    def load_results(self, filepath: str):
        """Load benchmark results from JSON file"""
        with open(filepath, 'r') as f:
            self.data = json.load(f)
    
    def compare_frameworks(
        self,
        metric: str = "success_rate",
        group_by: str = "failure_rate"
    ) -> Dict[str, Any]:
        """Compare frameworks on a specific metric"""
        if not self.data:
            raise ValueError("No results loaded")
        
        frameworks = {}
        
        for summary in self.data["summaries"]:
            fw = summary["config"]["framework"]
            group_value = summary["config"].get(group_by, "all")
            
            if fw not in frameworks:
                frameworks[fw] = {}
            
            if group_value not in frameworks[fw]:
                frameworks[fw][group_value] = []
            
            frameworks[fw][group_value].append(summary.get(metric, 0))
        
        # Calculate statistics
        comparison = {}
        for fw, groups in frameworks.items():
            comparison[fw] = {}
            for group, values in groups.items():
                comparison[fw][group] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return comparison
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate a text report of benchmark results"""
        if not self.data:
            raise ValueError("No results loaded")
        
        report_lines = [
            "=" * 80,
            "Compensation Benchmark Results",
            "=" * 80,
            f"Timestamp: {self.data.get('timestamp', 'unknown')}",
            f"Tasks: {', '.join(self.data['config']['tasks'])}",
            f"Frameworks: {', '.join(self.data['config']['frameworks'])}",
            f"Failure Rates: {', '.join(map(str, self.data['config']['failure_rates']))}",
            f"Runs per config: {self.data['config']['num_runs']}",
            "",
            "=" * 80,
            "Summary Statistics",
            "=" * 80,
        ]
        
        # Compare frameworks
        for metric in ["success_rate", "compensation_success_rate", "avg_execution_time"]:
            report_lines.append(f"\n{metric.replace('_', ' ').title()}:")
            report_lines.append("-" * 80)
            
            comparison = self.compare_frameworks(metric=metric)
            for fw, groups in comparison.items():
                report_lines.append(f"\n  {fw}:")
                for group, stats in groups.items():
                    report_lines.append(
                        f"    {group}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                        f"(n={stats['count']})"
                    )
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report
    
    def plot_comparison(
        self,
        metric: str = "success_rate",
        group_by: str = "failure_rate",
        output_file: str = None
    ):
        """Generate comparison plots"""
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("matplotlib/pandas required for visualization")
        
        comparison = self.compare_frameworks(metric=metric, group_by=group_by)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for fw, groups in comparison.items():
            x_values = sorted([float(k) for k in groups.keys() if k != "all"])
            y_means = [groups[str(x)]["mean"] for x in x_values]
            y_stds = [groups[str(x)]["std"] for x in x_values]
            
            ax.errorbar(x_values, y_means, yerr=y_stds, label=fw, marker='o', capsize=5)
        
        ax.set_xlabel(group_by.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f"{metric.replace('_', ' ').title()} by {group_by.replace('_', ' ').title()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def export_to_csv(self, output_file: str):
        """Export results to CSV for external analysis"""
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("pandas required for CSV export")
        
        rows = []
        for summary in self.data["summaries"]:
            row = {
                "task": summary["config"]["task_id"],
                "framework": summary["config"]["framework"],
                "failure_rate": summary["config"]["failure_rate"],
                "failure_mode": summary["config"]["failure_mode"],
                "success_rate": summary["success_rate"],
                "compensation_trigger_rate": summary["compensation_trigger_rate"],
                "compensation_success_rate": summary["compensation_success_rate"],
                "avg_execution_time": summary["avg_execution_time"],
                "avg_goal_satisfaction": summary["avg_goal_satisfaction"],
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)


def main():
    """CLI for benchmark analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze compensation benchmark results")
    parser.add_argument("results_file", help="Path to benchmark results JSON file")
    parser.add_argument("--report", help="Generate text report", action="store_true")
    parser.add_argument("--plot", help="Generate comparison plots", action="store_true")
    parser.add_argument("--csv", help="Export to CSV", type=str)
    parser.add_argument("--output-dir", default="analysis_results", help="Output directory")
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer(args.results_file)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.report:
        report_file = os.path.join(args.output_dir, "report.txt")
        report = analyzer.generate_report(report_file)
        print(report)
    
    if args.plot:
        for metric in ["success_rate", "compensation_success_rate", "avg_execution_time"]:
            plot_file = os.path.join(args.output_dir, f"{metric}.png")
            analyzer.plot_comparison(metric=metric, output_file=plot_file)
            print(f"Generated plot: {plot_file}")
    
    if args.csv:
        csv_file = os.path.join(args.output_dir, args.csv)
        analyzer.export_to_csv(csv_file)
        print(f"Exported CSV: {csv_file}")


if __name__ == "__main__":
    main()
