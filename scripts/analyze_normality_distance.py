import argparse
import csv
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
    sanitize_probs,
)


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


def _split_dataset(context, split_name):
    if split_name.startswith('source'):
        return context['direction'].source_dataset
    return context['direction'].target_dataset


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
    rng = np.random.RandomState(seed)
    rows = []
    anomaly_id = context['label2id']['Anomalous']
    for split_name in args.splits:
        instances = context[split_name]
        dataset = _split_dataset(context, split_name)
        for batch in iterate_batches(instances, args.batch_size, rng, shuffle=False):
            tinst = build_training_tinsts(batch, context['vocab'])
            move_tinst_to_runtime_device(tinst)
            logits = metalog.model(tinst.inputs)
            probs = sanitize_probs(logits)[:, anomaly_id].detach().cpu().numpy()
            cache = metalog.model.proj._last_cache
            global_distance = cache['global_distance'].detach().cpu().numpy()
            base_expert_distance = cache['base_expert_distance'].detach().cpu().numpy()
            expert_distance = cache['expert_distance'].detach().cpu().numpy()
            routing_probs = cache['routing_probs'].detach().cpu().numpy()
            weighted_base_distance = (routing_probs * base_expert_distance).sum(axis=-1)
            weighted_expert_distance = (routing_probs * expert_distance).sum(axis=-1)
            min_base_distance = base_expert_distance.min(axis=-1)
            for idx, inst in enumerate(batch):
                rows.append({
                    'sample_id': inst.id,
                    'direction': args.direction,
                    'split': split_name,
                    'dataset': dataset,
                    'label': inst.label,
                    'global_distance': float(np.sqrt(global_distance[idx] + 1e-9)),
                    'min_base_expert_distance': float(np.sqrt(min_base_distance[idx] + 1e-9)),
                    'weighted_base_expert_distance': float(np.sqrt(weighted_base_distance[idx] + 1e-9)),
                    'weighted_expert_distance': float(np.sqrt(weighted_expert_distance[idx] + 1e-9)),
                    'anomaly_score': float(probs[idx]),
                })
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        for distance_type, key in (
            ('Global-Or-Min Prototype', 'global_distance'),
            ('Min Base-Expert Prototype', 'min_base_expert_distance'),
            ('Weighted Base-Expert Prototype', 'weighted_base_expert_distance'),
            ('Weighted Expert Prototype', 'weighted_expert_distance'),
        ):
            grouped[(row['direction'], row['split'], row['dataset'], distance_type, row['label'])].append(float(row[key]))
    summary_rows = []
    pivot = {}
    for key, values in grouped.items():
        direction, split_name, dataset, distance_type, label = key
        pivot.setdefault((direction, split_name, dataset, distance_type), {})[label] = values
    for key, by_label in sorted(pivot.items()):
        direction, split_name, dataset, distance_type = key
        normal = np.asarray(by_label.get('Normal', []), dtype=np.float64)
        anomaly = np.asarray(by_label.get('Anomalous', []), dtype=np.float64)
        normal_mean = float(normal.mean()) if len(normal) else 0.0
        anomaly_mean = float(anomaly.mean()) if len(anomaly) else 0.0
        summary_rows.append({
            'direction': direction,
            'split': split_name,
            'dataset': dataset,
            'distance_type': distance_type,
            'normal_mean': normal_mean,
            'anomaly_mean': anomaly_mean,
            'gap': anomaly_mean - normal_mean,
            'gap_ratio': (anomaly_mean / normal_mean) if normal_mean > 0 else 0.0,
            'normal_count': int(len(normal)),
            'anomaly_count': int(len(anomaly)),
        })
    return summary_rows


def write_plots(rows, output_dir, direction):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = ['Normal', 'Anomalous']
    colors = {'Normal': '#4f6fd7', 'Anomalous': '#f05a68'}
    for field, filename, xlabel in (
        ('global_distance', 'fig_global_distance_distribution_%s.png' % direction, 'Global-or-min prototype distance'),
        ('min_base_expert_distance', 'fig_min_base_distance_distribution_%s.png' % direction, 'Min base-expert distance'),
        ('weighted_base_expert_distance', 'fig_base_distance_distribution_%s.png' % direction, 'Weighted base-expert distance'),
        ('weighted_expert_distance', 'fig_expert_distance_distribution_%s.png' % direction, 'Weighted expert distance'),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        data = [[float(row[field]) for row in rows if row['label'] == label] for label in labels]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for idx, body in enumerate(parts['bodies']):
            body.set_facecolor(colors[labels[idx]])
            body.set_alpha(0.55)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels)
        ax.set_ylabel(xlabel)
        ax.grid(axis='y', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for label in labels:
        x = [float(row['weighted_base_expert_distance']) for row in rows if row['label'] == label]
        y = [float(row['anomaly_score']) for row in rows if row['label'] == label]
        if len(x) > 5000:
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(x), size=5000, replace=False)
            x = [x[index] for index in indices]
            y = [y[index] for index in indices]
        ax.scatter(x, y, s=5, alpha=0.35, label=label, color=colors[label])
    ax.set_xlabel('Weighted base-expert distance')
    ax.set_ylabel('Anomaly score')
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_distance_score_scatter_%s.png' % direction), dpi=300)
    plt.close(fig)


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
    sample_path = os.path.join(args.output_dir, 'prototype_distance_samples_%s.csv' % args.direction)
    summary_path = os.path.join(args.output_dir, 'prototype_distance_summary_%s.csv' % args.direction)
    _write_rows(sample_path, rows, [
        'sample_id', 'direction', 'split', 'dataset', 'label',
        'global_distance', 'min_base_expert_distance', 'weighted_base_expert_distance',
        'weighted_expert_distance', 'anomaly_score',
    ])
    summary_rows = summarize(rows)
    _write_rows(summary_path, summary_rows, [
        'direction', 'split', 'dataset', 'distance_type',
        'normal_mean', 'anomaly_mean', 'gap', 'gap_ratio',
        'normal_count', 'anomaly_count',
    ])
    write_plots(rows, args.output_dir, args.direction)
    print('Wrote prototype distance analysis to %s and %s' % (sample_path, summary_path))


if __name__ == '__main__':
    main()
