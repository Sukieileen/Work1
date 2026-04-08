import sys

sys.path.extend([".", ".."])

import argparse
import csv
import itertools
import os
import random
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def format_float(value):
    text = ('%g' % value)
    return text.replace('.', 'p')


def build_search_space():
    search_space = {
        'model_lr': [1e-4, 2e-4, 3e-4, 5e-4],
        'dropout': [0.0, 0.1],
        'mamba_state': [32, 64, 96, 128],
        'mamba_conv': [4, 8],
        'mamba_expand': [2, 4],
    }
    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]
    configs = []
    for combo in itertools.product(*values):
        config = {key: value for key, value in zip(keys, combo)}
        configs.append(config)
    return configs


def sample_stage1_configs(all_configs, trial_count, seed):
    rng = random.Random(seed)
    if trial_count >= len(all_configs):
        return list(all_configs)
    return rng.sample(all_configs, trial_count)


def ensure_dirs(base_dir):
    for name in ['epoch_metrics', 'logs', 'summaries']:
        (base_dir / name).mkdir(parents=True, exist_ok=True)


def run_name_for(stage_name, trial_index, config, train_epochs):
    return (
        f'{stage_name}_trial{trial_index:02d}'
        f'_lr{format_float(config["model_lr"])}'
        f'_do{format_float(config["dropout"])}'
        f'_st{config["mamba_state"]}'
        f'_cv{config["mamba_conv"]}'
        f'_ex{config["mamba_expand"]}'
        f'_ep{train_epochs}'
    )


def epoch_csv_path(base_dir, run_name):
    return base_dir / 'epoch_metrics' / f'{run_name}.csv'


def log_path(base_dir, run_name):
    return base_dir / 'logs' / f'{run_name}.log'


def read_epoch_rows(epoch_csv):
    with epoch_csv.open('r', encoding='utf-8', newline='') as reader:
        rows = list(csv.DictReader(reader))
    return rows


def summarize_trial(stage_name, trial_index, config, train_epochs, run_name, epoch_csv, log_file):
    rows = read_epoch_rows(epoch_csv)
    if not rows:
        raise RuntimeError('No epoch metrics found in %s' % epoch_csv)
    best_row = max(
        rows,
        key=lambda row: (
            float(row['dev_f1']) if row['dev_f1'] else float('-inf'),
            float(row['f1']),
            float(row['aucpr']),
        ),
    )
    return {
        'stage': stage_name,
        'trial_index': trial_index,
        'run_name': run_name,
        'epochs': train_epochs,
        'model_lr': config['model_lr'],
        'dropout': config['dropout'],
        'mamba_state': config['mamba_state'],
        'mamba_conv': config['mamba_conv'],
        'mamba_expand': config['mamba_expand'],
        'best_epoch': int(best_row['epoch']),
        'selected_threshold': float(best_row['selected_threshold']),
        'best_dev_f1': float(best_row['dev_f1']) if best_row['dev_f1'] else float('nan'),
        'test_precision': float(best_row['precision']),
        'test_recall': float(best_row['recall']),
        'test_f1': float(best_row['f1']),
        'test_auroc': float(best_row['auroc']),
        'test_aucpr': float(best_row['aucpr']),
        'epoch_csv': str(epoch_csv),
        'log_file': str(log_file),
    }


def write_summary_csv(summary_file, rows):
    if not rows:
        return
    fieldnames = [
        'stage', 'trial_index', 'run_name', 'epochs', 'model_lr', 'dropout', 'mamba_state', 'mamba_conv',
        'mamba_expand', 'best_epoch', 'selected_threshold', 'best_dev_f1', 'test_precision', 'test_recall',
        'test_f1', 'test_auroc', 'test_aucpr', 'epoch_csv', 'log_file',
    ]
    with summary_file.open('w', encoding='utf-8', newline='') as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow({
                key: ('%.6f' % value if isinstance(value, float) else value)
                for key, value in row.items()
            })


def run_trial(base_dir, stage_name, trial_index, config, train_epochs, resume):
    run_name = run_name_for(stage_name, trial_index, config, train_epochs)
    metrics_file = epoch_csv_path(base_dir, run_name)
    trial_log = log_path(base_dir, run_name)

    if resume and metrics_file.exists():
        rows = read_epoch_rows(metrics_file)
        if rows:
            return summarize_trial(stage_name, trial_index, config, train_epochs, run_name, metrics_file, trial_log)

    cmd = [
        'conda', 'run', '--no-capture-output', '-n', 'work1',
        'python', 'approaches/MetaLog.py',
        '--mode', 'train',
        '--backbone', 'bimamba',
        '--auto_threshold',
        '--threshold_min', '0.50',
        '--threshold_max', '0.95',
        '--threshold_step', '0.005',
        '--model_lr', str(config['model_lr']),
        '--epochs', str(train_epochs),
        '--dropout', str(config['dropout']),
        '--mamba_state', str(config['mamba_state']),
        '--mamba_conv', str(config['mamba_conv']),
        '--mamba_expand', str(config['mamba_expand']),
        '--run_name', run_name,
        '--epoch_metrics_file', str(metrics_file),
    ]

    with trial_log.open('w', encoding='utf-8') as writer:
        writer.write('COMMAND: %s\n\n' % ' '.join(cmd))
        writer.flush()
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=writer,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError('Trial %s failed, see %s' % (run_name, trial_log))

    return summarize_trial(stage_name, trial_index, config, train_epochs, run_name, metrics_file, trial_log)


def select_top_trials(stage_rows, top_k):
    ranked = sorted(
        stage_rows,
        key=lambda row: (row['best_dev_f1'], row['test_aucpr'], row['test_auroc']),
        reverse=True,
    )
    return ranked[:top_k]


def write_search_plan(plan_file, configs, stage_name, train_epochs):
    fieldnames = ['stage', 'trial_index', 'epochs', 'model_lr', 'dropout', 'mamba_state', 'mamba_conv', 'mamba_expand']
    with plan_file.open('w', encoding='utf-8', newline='') as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        csv_writer.writeheader()
        for idx, config in enumerate(configs):
            csv_writer.writerow({
                'stage': stage_name,
                'trial_index': idx,
                'epochs': train_epochs,
                'model_lr': config['model_lr'],
                'dropout': config['dropout'],
                'mamba_state': config['mamba_state'],
                'mamba_conv': config['mamba_conv'],
                'mamba_expand': config['mamba_expand'],
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_trials', type=int, default=8, help='How many random configs to test in stage 1.')
    parser.add_argument('--stage1_epochs', type=int, default=6, help='Epochs for the coarse search stage.')
    parser.add_argument('--stage2_top_k', type=int, default=3, help='How many configs to promote to stage 2.')
    parser.add_argument('--stage2_epochs', type=int, default=12, help='Epochs for the refinement stage.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used to sample stage 1 configs.')
    parser.add_argument(
        '--search_dir',
        type=str,
        default=os.path.join(str(PROJECT_ROOT), 'backbone_compare', 'bimamba_hparam_search'),
        help='Root directory for search outputs.',
    )
    parser.add_argument('--resume', action='store_true', help='Reuse finished trials when their epoch CSV exists.')
    args = parser.parse_args()

    base_dir = Path(args.search_dir)
    ensure_dirs(base_dir)

    all_configs = build_search_space()
    stage1_configs = sample_stage1_configs(all_configs, args.stage1_trials, args.seed)
    write_search_plan(base_dir / 'stage1_plan.csv', stage1_configs, 'stage1', args.stage1_epochs)

    stage1_rows = []
    for idx, config in enumerate(stage1_configs):
        stage1_rows.append(run_trial(base_dir, 'stage1', idx, config, args.stage1_epochs, args.resume))
        write_summary_csv(base_dir / 'summaries' / 'stage1_summary.csv', stage1_rows)

    promoted_rows = select_top_trials(stage1_rows, args.stage2_top_k)
    promoted_configs = [
        {
            'model_lr': row['model_lr'],
            'dropout': row['dropout'],
            'mamba_state': row['mamba_state'],
            'mamba_conv': row['mamba_conv'],
            'mamba_expand': row['mamba_expand'],
        }
        for row in promoted_rows
    ]
    write_search_plan(base_dir / 'stage2_plan.csv', promoted_configs, 'stage2', args.stage2_epochs)

    stage2_rows = []
    for idx, config in enumerate(promoted_configs):
        stage2_rows.append(run_trial(base_dir, 'stage2', idx, config, args.stage2_epochs, args.resume))
        write_summary_csv(base_dir / 'summaries' / 'stage2_summary.csv', stage2_rows)

    combined_rows = stage1_rows + stage2_rows
    write_summary_csv(base_dir / 'summaries' / 'search_summary.csv', combined_rows)

    if stage2_rows:
        best_stage2 = select_top_trials(stage2_rows, 1)[0]
        with (base_dir / 'summaries' / 'best_stage2.txt').open('w', encoding='utf-8') as writer:
            for key in [
                'run_name', 'epochs', 'model_lr', 'dropout', 'mamba_state', 'mamba_conv', 'mamba_expand',
                'best_epoch', 'selected_threshold', 'best_dev_f1', 'test_precision', 'test_recall',
                'test_f1', 'test_auroc', 'test_aucpr', 'epoch_csv', 'log_file',
            ]:
                writer.write('%s: %s\n' % (key, best_stage2[key]))

    print('Search outputs written to %s' % base_dir)
    print('Stage 1 summary: %s' % (base_dir / 'summaries' / 'stage1_summary.csv'))
    print('Stage 2 summary: %s' % (base_dir / 'summaries' / 'stage2_summary.csv'))
    print('Combined summary: %s' % (base_dir / 'summaries' / 'search_summary.csv'))


if __name__ == '__main__':
    main()
