import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
COLORS = {'ppo': '#2E86AB', 'dqn': '#A23B72'}


def find_latest_json(algo: str, results_dir: Path) -> Path:
    """Find the most recent JSON file for the given algorithm"""
    files = list(results_dir.glob(f"{algo}_*.json"))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def load_training_json(path: Path) -> dict:
    """Load training results from JSON file"""
    if not path or not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_ppo_training(data: dict, output_dir: Path, window: int = 50):
    """Plot PPO training visualization"""
    if not data:
        return
    
    episode_rewards = data.get('episode_rewards', [])
    eval_rewards = data.get('eval_rewards', [])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Episode Rewards (Training)
    ax = axes[0, 0]
    if episode_rewards:
        ax.plot(episode_rewards, alpha=0.3, color=COLORS['ppo'], linewidth=0.5, label='Raw')
        if len(episode_rewards) > window:
            ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(episode_rewards)), ma, 
                   color=COLORS['ppo'], linewidth=2, label=f'{window}-ep MA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('PPO - Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 2. Evaluation Rewards
    ax = axes[0, 1]
    if eval_rewards:
        eval_x = np.linspace(0, len(episode_rewards), len(eval_rewards))
        ax.plot(eval_x, eval_rewards, color=COLORS['ppo'], linewidth=2, 
                marker='o', markersize=3, label='Eval')
        if len(eval_rewards) > window:
            eval_ma = np.convolve(eval_rewards, np.ones(window)/window, mode='valid')
            ax.plot(eval_x[window-1:], eval_ma, color='green', 
                   linewidth=2, linestyle='--', label=f'{window}-ep MA')
        best = max(eval_rewards)
        ax.axhline(y=best, color='red', linestyle=':', alpha=0.7, label=f'Best: {best:.1f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Eval Reward')
    ax.set_title('PPO - Evaluation Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Reward Distribution
    ax = axes[1, 0]
    if episode_rewards:
        ax.hist(episode_rewards, bins=30, color=COLORS['ppo'], alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(episode_rewards), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(episode_rewards):.1f}')
        ax.axvline(x=np.median(episode_rewards), color='orange', linestyle='--', 
                  label=f'Median: {np.median(episode_rewards):.1f}')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('PPO - Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Stats Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""PPO Training Statistics
{'='*40}

Training Config:
  Episodes: {data.get('episodes', 'N/A')}
  Users: {data.get('users', 'N/A')}
  Difficulty: {data.get('difficulty', 'N/A')}
  Timestamp: {data.get('timestamp', 'N/A')[:19]}

Episode Rewards:
  Count: {len(episode_rewards)}
  Mean: {np.mean(episode_rewards):.2f}
  Std: {np.std(episode_rewards):.2f}
  Min: {np.min(episode_rewards):.2f}
  Max: {np.max(episode_rewards):.2f}
  Final: {episode_rewards[-1]:.2f}"""

    if eval_rewards:
        stats_text += f"""
Evaluation Rewards:
  Count: {len(eval_rewards)}
  Final: {eval_rewards[-1]:.2f}
  Best: {max(eval_rewards):.2f}
  Mean (last 10): {np.mean(eval_rewards[-10:]):.2f}
  Mean (all): {np.mean(eval_rewards):.2f}"""

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('PPO Training Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'ppo_training_viz.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_dqn_training(data: dict, output_dir: Path, window: int = 50):
    """Plot DQN training visualization"""
    if not data:
        return
    
    episode_rewards = data.get('episode_rewards', [])
    eval_rewards = data.get('eval_rewards', [])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Episode Rewards (Training)
    ax = axes[0, 0]
    if episode_rewards:
        ax.plot(episode_rewards, alpha=0.3, color=COLORS['dqn'], linewidth=0.5, label='Raw')
        if len(episode_rewards) > window:
            ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(episode_rewards)), ma, 
                   color=COLORS['dqn'], linewidth=2, label=f'{window}-ep MA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('DQN - Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 2. Evaluation Rewards
    ax = axes[0, 1]
    if eval_rewards:
        eval_x = np.linspace(0, len(episode_rewards), len(eval_rewards))
        ax.plot(eval_x, eval_rewards, color=COLORS['dqn'], linewidth=2, 
                marker='o', markersize=3, label='Eval')
        if len(eval_rewards) > window:
            eval_ma = np.convolve(eval_rewards, np.ones(window)/window, mode='valid')
            ax.plot(eval_x[window-1:], eval_ma, color='green', 
                   linewidth=2, linestyle='--', label=f'{window}-ep MA')
        best = max(eval_rewards)
        ax.axhline(y=best, color='red', linestyle=':', alpha=0.7, label=f'Best: {best:.1f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Eval Reward')
    ax.set_title('DQN - Evaluation Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Reward Distribution
    ax = axes[1, 0]
    if episode_rewards:
        ax.hist(episode_rewards, bins=30, color=COLORS['dqn'], alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(episode_rewards), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(episode_rewards):.1f}')
        ax.axvline(x=np.median(episode_rewards), color='orange', linestyle='--', 
                  label=f'Median: {np.median(episode_rewards):.1f}')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('DQN - Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Stats Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""DQN Training Statistics
{'='*40}

Training Config:
  Episodes: {data.get('episodes', 'N/A')}
  Users: {data.get('users', 'N/A')}
  Difficulty: {data.get('difficulty', 'N/A')}
  Timestamp: {data.get('timestamp', 'N/A')[:19]}

Episode Rewards:
  Count: {len(episode_rewards)}
  Mean: {np.mean(episode_rewards):.2f}
  Std: {np.std(episode_rewards):.2f}
  Min: {np.min(episode_rewards):.2f}
  Max: {np.max(episode_rewards):.2f}
  Final: {episode_rewards[-1]:.2f}"""

    if eval_rewards:
        stats_text += f"""
Evaluation Rewards:
  Count: {len(eval_rewards)}
  Final: {eval_rewards[-1]:.2f}
  Best: {max(eval_rewards):.2f}
  Mean (last 10): {np.mean(eval_rewards[-10:]):.2f}
  Mean (all): {np.mean(eval_rewards):.2f}"""

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('DQN Training Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'dqn_training_viz.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_comparison(ppo_data: dict, dqn_data: dict, output_dir: Path):
    """Plot PPO vs DQN comparison"""
    if not ppo_data or not dqn_data:
        print("Missing data for comparison")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    ppo_eval = ppo_data.get('eval_rewards', [])
    dqn_eval = dqn_data.get('eval_rewards', [])
    
    # 1. Final Eval Reward
    ax = axes[0, 0]
    final_ppo = ppo_eval[-1] if ppo_eval else 0
    final_dqn = dqn_eval[-1] if dqn_eval else 0
    bars = ax.bar(['PPO', 'DQN'], [final_ppo, final_dqn], color=[COLORS['ppo'], COLORS['dqn']])
    ax.set_ylabel('Final Eval Reward')
    ax.set_title('Final Eval Reward')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, v in zip(bars, [final_ppo, final_dqn]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{v:.1f}', ha='center', va='bottom', fontsize=11)
    
    # 2. Best Eval Reward
    ax = axes[0, 1]
    best_ppo = max(ppo_eval) if ppo_eval else 0
    best_dqn = max(dqn_eval) if dqn_eval else 0
    bars = ax.bar(['PPO', 'DQN'], [best_ppo, best_dqn], color=[COLORS['ppo'], COLORS['dqn']])
    ax.set_ylabel('Best Eval Reward')
    ax.set_title('Best Eval Reward')
    for bar, v in zip(bars, [best_ppo, best_dqn]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{v:.1f}', ha='center', va='bottom', fontsize=11)
    
    # 3. Avg Eval Reward (Last 10)
    ax = axes[0, 2]
    avg_ppo = np.mean(ppo_eval[-10:]) if len(ppo_eval) >= 10 else np.mean(ppo_eval)
    avg_dqn = np.mean(dqn_eval[-10:]) if len(dqn_eval) >= 10 else np.mean(dqn_eval)
    bars = ax.bar(['PPO', 'DQN'], [avg_ppo, avg_dqn], color=[COLORS['ppo'], COLORS['dqn']])
    ax.set_ylabel('Avg Eval Reward (Last 10)')
    ax.set_title('Average Eval (Last 10)')
    for bar, v in zip(bars, [avg_ppo, avg_dqn]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{v:.1f}', ha='center', va='bottom', fontsize=11)
    
    # 4. Eval Reward Over Time (Combined)
    ax = axes[1, 0]
    if ppo_eval:
        ppo_x = np.linspace(0, ppo_data.get('episodes', 100), len(ppo_eval))
        ax.plot(ppo_x, ppo_eval, color=COLORS['ppo'], linewidth=2, 
                marker='o', markersize=2, label='PPO', alpha=0.8)
    if dqn_eval:
        dqn_x = np.linspace(0, dqn_data.get('episodes', 100), len(dqn_eval))
        ax.plot(dqn_x, dqn_eval, color=COLORS['dqn'], linewidth=2, 
                marker='s', markersize=2, label='DQN', alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Eval Reward')
    ax.set_title('Eval Rewards Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 5. Moving Average Comparison
    ax = axes[1, 1]
    window = 10
    if len(ppo_eval) > window:
        ppo_ma = np.convolve(ppo_eval, np.ones(window)/window, mode='valid')
        ppo_ma_x = np.linspace(0, ppo_data.get('episodes', 100), len(ppo_ma))
        ax.plot(ppo_ma_x, ppo_ma, color=COLORS['ppo'], linewidth=2, label='PPO MA')
    if len(dqn_eval) > window:
        dqn_ma = np.convolve(dqn_eval, np.ones(window)/window, mode='valid')
        dqn_ma_x = np.linspace(0, dqn_data.get('episodes', 100), len(dqn_ma))
        ax.plot(dqn_ma_x, dqn_ma, color=COLORS['dqn'], linewidth=2, label='DQN MA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Eval Reward (MA)')
    ax.set_title(f'{window}-Episode Moving Average')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 6. Summary Stats Table
    ax = axes[1, 2]
    ax.axis('off')
    
    ppo_ep = ppo_data.get('episode_rewards', [])
    dqn_ep = dqn_data.get('episode_rewards', [])
    
    summary = f"""Comparison Summary
{'='*45}

Metric              | PPO      | DQN
--------------------|----------|----------
Episodes            | {ppo_data.get('episodes', 0):>6}   | {dqn_data.get('episodes', 0):>6}
Final Eval          | {final_ppo:>7.2f} | {final_dqn:>7.2f}
Best Eval           | {best_ppo:>7.2f} | {best_dqn:>7.2f}
Avg Eval (last 10)  | {avg_ppo:>7.2f} | {avg_dqn:>7.2f}
Avg Eval (all)      | {np.mean(ppo_eval):>7.2f} | {np.mean(dqn_eval):>7.2f}
--------------------|----------|----------
Episode Mean        | {np.mean(ppo_ep):>7.2f} | {np.mean(dqn_ep):>7.2f}
Episode Std         | {np.std(ppo_ep):>7.2f} | {np.std(dqn_ep):>7.2f}
Episode Min         | {np.min(ppo_ep):>7.2f} | {np.min(dqn_ep):>7.2f}
Episode Max         | {np.max(ppo_ep):>7.2f} | {np.max(dqn_ep):>7.2f}

Winner: {'PPO' if final_ppo > final_dqn else 'DQN'} (Final Eval)
Best:  {'PPO' if best_ppo > best_dqn else 'DQN'} (Best Eval)
    """
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('PPO vs DQN Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'comparison_viz.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def run_benchmark(ppo_path: str = None, dqn_path: str = None, num_episodes: int = 50):
    """Run benchmark to get correction rate and other metrics"""
    from simulation_enhanced import benchmark
    from rl_ppo_agent import PPOPPOAgent
    from rl_agent import DQNAgent
    import random
    random.seed(42)
    
    results = {}
    
    if ppo_path:
        agent = PPOPPOAgent(state_size=18, action_size=3)
        agent.load(ppo_path)
        results['ppo'] = benchmark(agent, num_episodes=num_episodes)
    
    if dqn_path:
        agent = DQNAgent(state_size=18, action_size=3)
        agent.load(dqn_path)
        results['dqn'] = benchmark(agent, num_episodes=num_episodes)
    
    return results


def plot_benchmark(benchmark_results: dict, output_dir: Path):
    """Plot benchmark results"""
    if not benchmark_results:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    algos = list(benchmark_results.keys())
    colors = [COLORS.get(a.lower()) for a in algos]
    
    # Correction Rate
    ax = axes[0]
    vals = [benchmark_results[a].get('correction_rate', 0) * 100 for a in algos]
    bars = ax.bar(algos, vals, color=colors)
    ax.set_ylabel('Correction Rate (%)')
    ax.set_title('Correction Rate')
    ax.set_ylim(0, 100)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{v:.1f}%', ha='center', va='bottom', fontsize=11)
    
    # Avg Episode Reward
    ax = axes[1]
    vals = [benchmark_results[a].get('avg_episode_reward', 0) for a in algos]
    bars = ax.bar(algos, vals, color=colors)
    ax.set_ylabel('Avg Episode Reward')
    ax.set_title('Average Episode Reward')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{v:.1f}', ha='center', va='bottom', fontsize=11)
    
    # Total Alerts
    ax = axes[2]
    vals = [benchmark_results[a].get('total_alerts', 0) for a in algos]
    bars = ax.bar(algos, vals, color=colors)
    ax.set_ylabel('Total Alerts')
    ax.set_title('Total Alerts')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{int(v)}', ha='center', va='bottom', fontsize=11)
    
    plt.suptitle('Benchmark Results (50 episodes)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'benchmark_viz.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')
    
    # Print benchmark summary
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    for algo, metrics in benchmark_results.items():
        print(f"\n{algo.upper()}:")
        print(f"  Correction Rate: {metrics.get('correction_rate', 0)*100:.1f}%")
        print(f"  Avg Episode Reward: {metrics.get('avg_episode_reward', 0):.2f}")
        print(f"  Total Alerts: {int(metrics.get('total_alerts', 0))}")
        print(f"  Successful Corrections: {int(metrics.get('total_corrections', 0))}")


def print_summary(ppo_data: dict, dqn_data: dict):
    """Print training summary to console"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for algo, data in [('PPO', ppo_data), ('DQN', dqn_data)]:
        if not data:
            continue
        
        print(f"\n{algo}:")
        print(f"  Episodes: {data.get('episodes')}")
        print(f"  Difficulty: {data.get('difficulty')}")
        print(f"  Timestamp: {data.get('timestamp', 'N/A')[:19]}")
        
        ep_rewards = data.get('episode_rewards', [])
        eval_rewards = data.get('eval_rewards', [])
        
        print(f"\n  Episode Rewards:")
        print(f"    Count: {len(ep_rewards)}")
        print(f"    Mean: {np.mean(ep_rewards):.2f}")
        print(f"    Std: {np.std(ep_rewards):.2f}")
        print(f"    Min: {np.min(ep_rewards):.2f}")
        print(f"    Max: {np.max(ep_rewards):.2f}")
        print(f"    Final: {ep_rewards[-1]:.2f}")
        
        if eval_rewards:
            print(f"\n  Eval Rewards:")
            print(f"    Count: {len(eval_rewards)}")
            print(f"    Final: {eval_rewards[-1]:.2f}")
            print(f"    Best: {max(eval_rewards):.2f}")
            print(f"    Avg (last 10): {np.mean(eval_rewards[-10:]):.2f}")
            print(f"    Avg (all): {np.mean(eval_rewards):.2f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--ppo-file', type=str, default=None,
                       help='PPO JSON file (default: auto-detect latest)')
    parser.add_argument('--dqn-file', type=str, default=None,
                       help='DQN JSON file (default: auto-detect latest)')
    parser.add_argument('--output-dir', default='results',
                       help='Results directory (default: results)')
    parser.add_argument('--window', type=int, default=50,
                       help='Moving average window (default: 50)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark comparison')
    parser.add_argument('--benchmark-episodes', type=int, default=50,
                       help='Number of episodes for benchmark (default: 50)')
    args = parser.parse_args()
    
    global RESULTS_DIR
    RESULTS_DIR = Path(args.output_dir)
    
    print("="*60)
    print("VISUALIZATION TOOL")
    print("="*60)
    print(f"Results directory: {RESULTS_DIR}")
    
    # Find JSON files
    ppo_path = args.ppo_file if args.ppo_file else find_latest_json('ppo', RESULTS_DIR)
    dqn_path = args.dqn_file if args.dqn_file else find_latest_json('dqn', RESULTS_DIR)
    
    print(f"PPO file: {ppo_path}")
    print(f"DQN file: {dqn_path}")
    
    # Load data
    ppo_data = load_training_json(ppo_path) if ppo_path else None
    dqn_data = load_training_json(dqn_path) if dqn_path else None
    
    if not ppo_data and not dqn_data:
        print("ERROR: No training data found!")
        return
    
    # Generate visualizations
    if ppo_data:
        plot_ppo_training(ppo_data, RESULTS_DIR, args.window)
    
    if dqn_data:
        plot_dqn_training(dqn_data, RESULTS_DIR, args.window)
    
    if ppo_data and dqn_data:
        plot_comparison(ppo_data, dqn_data, RESULTS_DIR)
    
    # Run benchmark if requested
    if args.benchmark:
        ppo_model = RESULTS_DIR / 'ppo_best.pth'
        dqn_model = RESULTS_DIR / 'dqn_best.pth'
        
        benchmark_results = run_benchmark(
            str(ppo_model) if ppo_model.exists() else None,
            str(dqn_model) if dqn_model.exists() else None,
            args.benchmark_episodes
        )
        if benchmark_results:
            plot_benchmark(benchmark_results, RESULTS_DIR)
    
    # Print summary
    print_summary(ppo_data, dqn_data)
    
    print(f"\n{'='*60}")
    print("Done! Generated visualizations:")
    print(f"  - ppo_training_viz.png")
    print(f"  - dqn_training_viz.png")
    print(f"  - comparison_viz.png")
    if args.benchmark:
        print(f"  - benchmark_viz.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()