import argparse
import csv
import os


def _read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as reader:
        return list(csv.DictReader(reader))


def _best_rows(rows):
    latest = {}
    for row in rows:
        key = (row['train_direction'], row['zero_target'], row['method'])
        latest[key] = row
    return [latest[key] for key in sorted(latest.keys())]


def _fmt(value):
    try:
        return '%.2f' % (100.0 * float(value))
    except (TypeError, ValueError):
        return value


def _direction_label(direction):
    mapping = {
        'hdfs_to_hpc': 'HDFS->HPC',
        'hdfs_to_hpc_sr065': 'HDFS->HPC_sr065',
        'hpc_to_hdfs': 'HPC_sr065->HDFS',
        'hdfs30_hpc065_known_mix': 'HDFS30+HPC_sr065',
    }
    return mapping.get(direction, direction)


def write_markdown(rows, path):
    lines = [
        '| Train Direction | Zero-shot Target | Method | Precision | Recall | F1 | Threshold Source |',
        '|---|---|---|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            '| %s | %s | %s | %s | %s | %s | %s |' % (
                _direction_label(row['train_direction']),
                row['zero_target'],
                row['method'],
                _fmt(row['precision']),
                _fmt(row['recall']),
                _fmt(row['f1']),
                row['threshold_source'],
            )
        )
    with open(path, 'w', encoding='utf-8') as writer:
        writer.write('\n'.join(lines) + '\n')


def write_latex(rows, path):
    lines = [
        r'\begin{tabular}{lllrrrr}',
        r'\toprule',
        r'Train Direction & Target & Method & Precision & Recall & F1 & Threshold Source \\',
        r'\midrule',
    ]
    for row in rows:
        lines.append(
            r'%s & %s & %s & %s & %s & %s & %s \\' % (
                _direction_label(row['train_direction']).replace('->', r'$\rightarrow$'),
                row['zero_target'],
                row['method'],
                _fmt(row['precision']),
                _fmt(row['recall']),
                _fmt(row['f1']),
                row['threshold_source'],
            )
        )
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    with open(path, 'w', encoding='utf-8') as writer:
        writer.write('\n'.join(lines) + '\n')


def write_plot(rows, path):
    try:
        import matplotlib
    except ImportError:
        print('matplotlib is not installed; skip %s' % path)
        return

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        return
    targets = sorted(set(row['target_key'] for row in rows))
    methods = sorted(set(row['method'] for row in rows))
    fig, ax = plt.subplots(figsize=(max(6, len(targets) * 2.4), 4.0))
    x = np.arange(len(targets))
    width = 0.8 / max(1, len(methods))
    colors = ['#4f6fd7', '#f05a68', '#9a6a3a', '#2f9a72', '#5d5d66']
    for idx, method in enumerate(methods):
        values = []
        for target in targets:
            matches = [row for row in rows if row['target_key'] == target and row['method'] == method]
            values.append(float(matches[0]['f1']) if matches else 0.0)
        offset = (idx - (len(methods) - 1) / 2.0) * width
        bars = ax.bar(x + offset, values, width, label=method, color=colors[idx % len(colors)], alpha=0.75)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                '%.2f' % value,
                ha='center',
                va='bottom',
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=15 if any('/' in item for item in targets) else 0, ha='right')
    ax.set_ylim(0, min(1.05, max(0.25, max(float(row['f1']) for row in rows) + 0.12)))
    ax.set_ylabel('F1-score')
    ax.legend(frameon=True, ncol=min(3, len(methods)), loc='upper center', bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_metric_plot(rows, path):
    try:
        import matplotlib
    except ImportError:
        print('matplotlib is not installed; skip %s' % path)
        return

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        return
    methods = sorted(set(row['method'] for row in rows))
    targets = sorted(set(row['target_key'] for row in rows))
    metrics = [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1-score')]
    colors = ['#4f6fd7', '#f05a68', '#b98a54']

    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(max(6.0, len(targets) * max(3.2, len(methods) * 0.9)), 3.8),
        sharey=True,
    )
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        subset = [row for row in rows if row['target_key'] == target]
        x = np.arange(len(methods))
        width = 0.22
        for metric_idx, (metric_key, metric_label) in enumerate(metrics):
            values = []
            for method in methods:
                matches = [row for row in subset if row['method'] == method]
                values.append(float(matches[0][metric_key]) if matches else 0.0)
            offset = (metric_idx - 1) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=metric_label if ax is axes[0] else None,
                color=colors[metric_idx],
                alpha=0.72,
                edgecolor=colors[metric_idx],
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(1.02, bar.get_height() + 0.015),
                    '%.2f' % value,
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    color=colors[metric_idx],
                )
        ax.set_title(target)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.set_ylim(0, 1.08)
        ax.grid(axis='y', alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('Score')
    fig.legend(frameon=True, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='outputs/experiments/zeroshot/results_zeroshot_raw.csv')
    parser.add_argument('--output_dir', default='outputs/experiments/zeroshot')
    parser.add_argument('--plot_direction', default='hdfs_to_hpc_sr065')
    args = parser.parse_args()

    rows = _best_rows(_read_rows(args.input))
    os.makedirs(args.output_dir, exist_ok=True)
    write_markdown(rows, os.path.join(args.output_dir, 'results_zeroshot_table.md'))
    write_latex(rows, os.path.join(args.output_dir, 'results_zeroshot_table.tex'))

    plot_rows = [dict(row, target_key=row['zero_target']) for row in rows if row['train_direction'] == args.plot_direction]
    if not plot_rows:
        plot_rows = [dict(row, target_key='%s/%s' % (row['train_direction'], row['zero_target'])) for row in rows]
    write_plot(plot_rows, os.path.join(args.output_dir, 'fig_zeroshot_f1_bar.png'))
    write_metric_plot(plot_rows, os.path.join(args.output_dir, 'fig_zeroshot_precision_recall_f1_bar.png'))
    print('Wrote %d summarized zero-shot rows to %s' % (len(rows), args.output_dir))


if __name__ == '__main__':
    main()
