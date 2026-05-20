import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.extend([".", ".."])

from CONSTANTS import PROJECT_ROOT, seed
from approaches.supervised_protocol import (
    MetaLog,
    build_arg_parser,
    build_training_tinsts,
    iterate_batches,
    lstm_hiddens,
    load_checkpoint_state_dict,
    load_state_with_expanded_embedding,
    move_tinst_to_runtime_device,
    num_layer,
    prepare_protocol_context,
)


def _read_components(dataset_name):
    path = os.path.join(PROJECT_ROOT, 'datasets', dataset_name, 'block_sources.txt')
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            block_id, rest = line.split(':', 1)
            component = rest.split(':', 1)[0]
            mapping[block_id] = component
    return mapping


def _split_dataset(context, split_name):
    if split_name.startswith('source'):
        return context['direction'].source_dataset
    return context['direction'].target_dataset


def _build_model(context, args):
    metalog = MetaLog(
        context['vocab'],
        num_layer,
        lstm_hiddens,
        context['label2id'],
        backbone=args.backbone,
        dropout=args.dropout,
        mamba_state=args.mamba_state,
        mamba_conv=args.mamba_conv,
        mamba_expand=args.mamba_expand,
        mamba_variant=args.mamba_variant,
        use_moe=args.use_moe,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_bottleneck_dim=args.moe_bottleneck_dim,
        moe_temperature=args.moe_temperature,
        moe_gate_dropout=args.moe_gate_dropout,
        moe_balance_loss_weight=args.calibration_balance_loss_weight,
        moe_diversity_loss_weight=args.calibration_diversity_loss_weight,
        moe_z_loss_weight=args.moe_z_loss_weight,
        use_normality_anchor=args.use_normality_anchor,
        prototype_scale=args.prototype_scale,
        prototype_loss_weight=args.prototype_loss_weight,
        prototype_sep_weight=args.prototype_sep_weight,
        prototype_margin_global=args.prototype_margin_global,
        prototype_margin_expert=args.prototype_margin_expert,
        prototype_target_normal_only=args.prototype_target_normal_only,
        router_use_distance=args.router_use_distance,
        router_distance_mode=args.router_distance_mode,
        router_distance_scale=args.router_distance_scale,
        use_global_prototype=args.use_global_prototype,
        prototype_diversity_margin=args.prototype_diversity_margin,
    )
    load_state_with_expanded_embedding(metalog.model, load_checkpoint_state_dict(args.checkpoint))
    metalog.model.eval()
    return metalog


def _write_rows(path, rows, fieldnames):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)


def collect_rows(context, args):
    metalog = _build_model(context, args)
    rows = []
    component_maps = {
        context['direction'].source_dataset: _read_components(context['direction'].source_dataset),
        context['direction'].target_dataset: _read_components(context['direction'].target_dataset),
    }
    rng = np.random.RandomState(seed)
    for split_name in args.splits:
        instances = context[split_name]
        dataset = _split_dataset(context, split_name)
        component_map = component_maps.get(dataset, {})
        for batch in iterate_batches(instances, args.batch_size, rng, shuffle=False):
            tinst = build_training_tinsts(batch, context['vocab'])
            move_tinst_to_runtime_device(tinst)
            metalog.model(tinst.inputs)
            cache = metalog.model.proj._last_cache
            routing_probs = cache['routing_probs'].detach().cpu().numpy()
            routing_mask = cache['routing_mask'].detach().cpu().numpy()
            base_expert_distance = cache.get('base_expert_distance')
            if base_expert_distance is not None:
                base_expert_distance = base_expert_distance.detach().cpu().numpy()
            for idx, inst in enumerate(batch):
                top1 = int(np.argmax(routing_probs[idx]))
                topk = np.where(routing_mask[idx] > 0)[0].astype(int).tolist()
                rows.append({
                    'sample_id': inst.id,
                    'direction': args.direction,
                    'split': split_name,
                    'dataset': dataset,
                    'component': component_map.get(str(inst.id), dataset),
                    'label': inst.label,
                    'top1_expert': top1,
                    'topk_experts': ' '.join(str(item) for item in topk),
                    'top1_base_distance': (
                        float(np.sqrt(base_expert_distance[idx, top1] + 1e-9))
                        if base_expert_distance is not None else 0.0
                    ),
                    'routing_probs': json.dumps([float(value) for value in routing_probs[idx]]),
                })
    return rows


def summarize(rows, num_experts):
    grouped = defaultdict(lambda: {
        'total': 0,
        'top1': np.zeros(num_experts, dtype=np.int64),
        'topk': np.zeros(num_experts, dtype=np.int64),
    })
    for row in rows:
        key = (row['direction'], row['split'], row['dataset'], row['component'], row['label'])
        bucket = grouped[key]
        bucket['total'] += 1
        bucket['top1'][int(row['top1_expert'])] += 1
        for expert in row['topk_experts'].split():
            if expert:
                bucket['topk'][int(expert)] += 1

    summary_rows = []
    for key, bucket in sorted(grouped.items()):
        direction, split_name, dataset, component, label = key
        total = max(bucket['total'], 1)
        for expert in range(num_experts):
            summary_rows.append({
                'direction': direction,
                'split': split_name,
                'dataset': dataset,
                'component': component,
                'label': label,
                'expert': expert,
                'total_samples': bucket['total'],
                'top1_count': int(bucket['top1'][expert]),
                'top1_rate': bucket['top1'][expert] / float(total),
                'topk_count': int(bucket['topk'][expert]),
                'topk_load': bucket['topk'][expert] / float(total),
            })
    return summary_rows


def write_plot(summary_rows, output_path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = [row for row in summary_rows if row['component'] == row['dataset']]
    if not rows:
        rows = summary_rows
    labels = sorted(set(row['label'] for row in rows))
    experts = sorted(set(int(row['expert']) for row in rows))
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(1, len(labels))
    x = np.arange(len(experts))
    colors = {'Normal': '#4f6fd7', 'Anomalous': '#f05a68'}
    for label_idx, label in enumerate(labels):
        values = []
        for expert in experts:
            matches = [row for row in rows if row['label'] == label and int(row['expert']) == expert]
            values.append(float(np.mean([float(row['topk_load']) for row in matches])) if matches else 0.0)
        offset = (label_idx - (len(labels) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=label, color=colors.get(label), alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([str(expert) for expert in experts])
    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Top-k load ratio')
    ax.legend(frameon=True)
    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _aggregate_matrix(summary_rows, row_key, col_key, value_key='topk_load', split_name='target_test'):
    matrix = defaultdict(lambda: defaultdict(list))
    row_labels = []
    col_labels = []
    for row in summary_rows:
        if row['split'] != split_name:
            continue
        row_value = row[row_key]
        col_value = row[col_key]
        if row_value not in row_labels:
            row_labels.append(row_value)
        if col_value not in col_labels:
            col_labels.append(col_value)
        matrix[row_value][col_value].append(float(row[value_key]))

    values = np.asarray([
        [float(np.mean(matrix[row_label][col_label])) if matrix[row_label][col_label] else 0.0 for col_label in col_labels]
        for row_label in row_labels
    ], dtype=np.float64)
    return row_labels, col_labels, values


def _write_heatmap(values, row_labels, col_labels, title, xlabel, ylabel, output_path, cmap='Blues'):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.9), max(3.5, len(row_labels) * 0.6)))
    im = ax.imshow(values, aspect='auto', cmap=cmap)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels([str(label) for label in col_labels], rotation=45, ha='right')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels([str(label) for label in row_labels])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_heatmaps(summary_rows, output_dir, direction):
    label_rows, label_cols, label_values = _aggregate_matrix(
        summary_rows,
        row_key='label',
        col_key='expert',
        split_name='target_test',
    )
    if len(label_rows) > 0 and len(label_cols) > 0:
        _write_heatmap(
            label_values,
            label_rows,
            label_cols,
            title='Expert load by label (%s)' % direction,
            xlabel='Expert ID',
            ylabel='Label',
            output_path=os.path.join(output_dir, 'fig_expert_label_heatmap_%s.png' % direction),
        )

    component_rows, component_cols, component_values = _aggregate_matrix(
        summary_rows,
        row_key='component',
        col_key='expert',
        split_name='target_test',
    )
    if len(component_rows) > 0 and len(component_cols) > 0:
        _write_heatmap(
            component_values,
            component_rows,
            component_cols,
            title='Expert load by component (%s)' % direction,
            xlabel='Expert ID',
            ylabel='Component',
            output_path=os.path.join(output_dir, 'fig_expert_component_heatmap_%s.png' % direction),
        )


def main():
    parent = build_arg_parser()
    parser = argparse.ArgumentParser(parents=[parent], conflict_handler='resolve')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--splits', nargs='+', default=['target_test'],
                        choices=['source_train', 'source_dev', 'target_train', 'target_dev', 'target_test'])
    parser.add_argument('--output_dir', default=os.path.join(PROJECT_ROOT, 'outputs', 'experiments', 'interpretability'))
    args = parser.parse_args()
    args.moe_bottleneck_dim = args.moe_bottleneck_dim if args.moe_bottleneck_dim > 0 else None

    context = prepare_protocol_context(args.direction, args.parser, protocol=args.protocol, args=args)
    rows = collect_rows(context, args)
    sample_path = os.path.join(args.output_dir, 'routing_samples_%s.csv' % args.direction)
    summary_path = os.path.join(args.output_dir, 'routing_summary_by_expert_%s.csv' % args.direction)
    _write_rows(sample_path, rows, [
        'sample_id', 'direction', 'split', 'dataset', 'component', 'label',
        'top1_expert', 'topk_experts', 'top1_base_distance', 'routing_probs',
    ])
    summary_rows = summarize(rows, args.moe_num_experts)
    _write_rows(summary_path, summary_rows, [
        'direction', 'split', 'dataset', 'component', 'label', 'expert',
        'total_samples', 'top1_count', 'top1_rate', 'topk_count', 'topk_load',
    ])
    write_plot(summary_rows, os.path.join(args.output_dir, 'fig_expert_load_bar_%s.png' % args.direction))
    write_heatmaps(summary_rows, args.output_dir, args.direction)
    print('Wrote routing analysis to %s and %s' % (sample_path, summary_path))


if __name__ == '__main__':
    main()
