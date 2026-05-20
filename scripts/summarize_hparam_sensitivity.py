import argparse
import csv
import os


def read_rows(path):
    with open(path, 'r', encoding='utf-8') as reader:
        return list(csv.DictReader(reader))


def write_plot(rows, output_dir):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_specs = {
        'num_experts': ('fig_f1_vs_num_experts.png', 'Number of experts K'),
        'topk': ('fig_f1_vs_topk.png', 'Top-k routing'),
        'proto_weight': ('fig_f1_vs_proto_weight.png', 'Prototype loss weight'),
    }
    for hp_name, (filename, xlabel) in plot_specs.items():
        hp_rows = [row for row in rows if row['hyperparam_name'] == hp_name]
        if not hp_rows:
            continue
        hp_rows.sort(key=lambda row: float(row['hyperparam_value']))
        x = [float(row['hyperparam_value']) for row in hp_rows]
        y = [float(row['f1']) for row in hp_rows]
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.plot(x, y, marker='o', color='#4f6fd7', linewidth=2)
        for xv, yv in zip(x, y):
            ax.text(xv, yv + 0.4, '%.2f' % yv, ha='center', va='bottom', fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('F1-score')
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close(fig)


def write_summary(rows, path):
    groups = {}
    for row in rows:
        groups.setdefault(row['hyperparam_name'], []).append(row)
    lines = [
        '| Hyperparameter | Default | Tested Values | Best Value | Best F1 |',
        '|---|---:|---|---:|---:|',
    ]
    defaults = {
        'num_experts': '4',
        'topk': '2',
        'proto_weight': '0.1',
    }
    for hp_name, hp_rows in sorted(groups.items()):
        hp_rows.sort(key=lambda row: float(row['hyperparam_value']))
        best = max(hp_rows, key=lambda row: float(row['f1']))
        lines.append('| %s | %s | %s | %s | %.2f |' % (
            hp_name,
            defaults.get(hp_name, ''),
            '/'.join(row['hyperparam_value'] for row in hp_rows),
            best['hyperparam_value'],
            float(best['f1']),
        ))
    with open(path, 'w', encoding='utf-8') as writer:
        writer.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='outputs/experiments/hparams/hparam_sensitivity_raw.csv')
    parser.add_argument('--output_dir', default='outputs/experiments/hparams')
    args = parser.parse_args()
    rows = read_rows(args.input)
    os.makedirs(args.output_dir, exist_ok=True)
    write_plot(rows, args.output_dir)
    write_summary(rows, os.path.join(args.output_dir, 'hparam_sensitivity_summary.md'))
    print('Wrote hyperparameter sensitivity summary to %s' % args.output_dir)


if __name__ == '__main__':
    main()
