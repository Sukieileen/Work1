import argparse
import csv
import os


def read_best_row(path):
    with open(path, 'r', encoding='utf-8') as reader:
        rows = list(csv.DictReader(reader))
    if not rows:
        raise ValueError('No rows in %s' % path)
    selected = [row for row in rows if row.get('selected_for_best') == '1' and row.get('phase') == 'phase_c']
    if not selected:
        selected = [row for row in rows if row.get('phase') == 'phase_c']
    if not selected:
        selected = rows
    return selected[-1]


def load_manifest(path):
    with open(path, 'r', encoding='utf-8') as reader:
        return list(csv.DictReader(reader))


def write_rows(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)


def write_markdown(path, rows):
    lines = [
        '| Direction | Variant | Precision | Recall | F1 | Delta F1 vs A0 | Delta F1 vs Prev |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append('| %s | %s %s | %.2f | %.2f | %.2f | %.2f | %.2f |' % (
            row['direction'],
            row['variant_id'],
            row['variant_name'],
            float(row['precision']),
            float(row['recall']),
            float(row['f1']),
            float(row['delta_f1_vs_a0']),
            float(row['delta_f1_vs_previous']),
        ))
    with open(path, 'w', encoding='utf-8') as writer:
        writer.write('\n'.join(lines) + '\n')


def write_latex(path, rows):
    lines = [
        r'\begin{tabular}{llrrrrr}',
        r'\toprule',
        r'Direction & Variant & Precision & Recall & F1 & $\Delta$F1 vs A0 & $\Delta$F1 vs Prev \\',
        r'\midrule',
    ]
    for row in rows:
        lines.append(r'%s & %s %s & %.2f & %.2f & %.2f & %.2f & %.2f \\' % (
            row['direction'],
            row['variant_id'],
            row['variant_name'],
            float(row['precision']),
            float(row['recall']),
            float(row['f1']),
            float(row['delta_f1_vs_a0']),
            float(row['delta_f1_vs_previous']),
        ))
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    with open(path, 'w', encoding='utf-8') as writer:
        writer.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output_dir', default='outputs/experiments/ablation')
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    rows = []
    by_direction = {}
    for item in manifest:
        best = read_best_row(item['metrics_file'])
        row = dict(item)
        row.update({
            'precision': float(best['test_precision']),
            'recall': float(best['test_recall']),
            'f1': float(best['test_f1']),
            'threshold': float(best['selected_threshold']),
            'checkpoint': item.get('checkpoint', ''),
        })
        rows.append(row)
        by_direction.setdefault(row['direction'], []).append(row)

    final_rows = []
    for direction, direction_rows in by_direction.items():
        direction_rows.sort(key=lambda row: row['variant_id'])
        base_f1 = direction_rows[0]['f1']
        previous_f1 = None
        for row in direction_rows:
            row['delta_f1_vs_a0'] = row['f1'] - base_f1
            row['delta_f1_vs_previous'] = 0.0 if previous_f1 is None else row['f1'] - previous_f1
            previous_f1 = row['f1']
            final_rows.append(row)

    fieldnames = [
        'direction', 'variant_id', 'variant_name', 'use_moe', 'use_normality_anchor',
        'router_use_distance', 'prototype_scale', 'prototype_loss_weight',
        'prototype_sep_weight', 'precision', 'recall', 'f1', 'threshold',
        'delta_f1_vs_a0', 'delta_f1_vs_previous', 'checkpoint', 'metrics_file',
    ]
    write_rows(os.path.join(args.output_dir, 'component_ablation_raw.csv'), final_rows, fieldnames)
    write_markdown(os.path.join(args.output_dir, 'component_ablation_table.md'), final_rows)
    write_latex(os.path.join(args.output_dir, 'component_ablation_table.tex'), final_rows)
    print('Wrote component ablation summary to %s' % args.output_dir)


if __name__ == '__main__':
    main()
