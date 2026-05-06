from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import FIGURE_CAPTIONS, build_manifest


def _require_gradio():
    try:
        import gradio as gr  # type: ignore
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The interactive explorer requires the local app environment. Install it with `pip install -e .[app]` from the project root."
        ) from exc
    return gr, pd


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _results_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for noise_type in manifest['noise_types']:
        run = manifest['runs'][noise_type]
        training = run['training']
        evaluation = run['evaluation']
        rows.append(
            {
                'distribution': noise_type.title(),
                'role': run['label'],
                'final_epoch_loss': round(training['final_epoch_avg_loss'], 5),
                'avg_denoising_mse': round(evaluation['avg_denoising_mse'], 5),
                'mse_t500': round(evaluation['mse_t500'], 5),
                'mse_t999': round(evaluation['mse_t999'], 5),
                'sample_variance': round(evaluation['sample_variance'], 5),
                'runtime_min': round(training['duration_minutes'], 2),
                'peak_gpu_gb': round(training['peak_memory_gb'], 2),
            }
        )
    return rows


def _controls_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    config = manifest['config']
    order = [
        'dataset', 'image_size', 'channels', 'T', 'schedule', 'num_epochs', 'batch_size',
        'lr', 'optimizer', 'weight_decay', 'model_channels', 'channel_mults', 'num_workers',
        'seed', 'sample_count'
    ]
    return [{'parameter': key, 'value': config.get(key)} for key in order if key in config]


def _run_markdown(manifest: dict[str, Any], noise_type: str) -> str:
    run = manifest['runs'][noise_type]
    training = run['training']
    evaluation = run['evaluation']
    best_epoch = run['files'].get('best_logged_epoch') or {}
    return '\n'.join(
        [
            f"## {run['label']}",
            '',
            training.get('training_objective', 'Training objective not recorded.'),
            '',
            f"- Final epoch loss: `{training['final_epoch_avg_loss']:.5f}`",
            f"- Average denoising MSE: `{evaluation['avg_denoising_mse']:.5f}`",
            f"- MSE at `t=500`: `{evaluation['mse_t500']:.5f}`",
            f"- MSE at `t=999`: `{evaluation['mse_t999']:.5f}`",
            f"- Runtime: `{training['duration_minutes']:.2f}` minutes",
            f"- Peak GPU memory: `{training['peak_memory_gb']:.2f}` GB",
            f"- Half-loss epoch: `{evaluation.get('half_loss_epoch', 'n/a')}`",
            f"- Best logged epoch: `{best_epoch.get('epoch', 'n/a')}` with epoch-average loss `{best_epoch.get('epoch_avg_loss', 0.0):.5f}`",
            f"- Saved checkpoints: `{run['files']['checkpoint_count']}`",
        ]
    )


def _loss_history_rows(manifest: dict[str, Any], noise_type: str) -> list[dict[str, Any]]:
    return [
        {
            'epoch': row['epoch'],
            'step': row['step'],
            'batch_loss': round(row['loss'], 6),
            'epoch_avg_loss': round(row['epoch_avg_loss'], 6),
            'elapsed_s': round(row['elapsed_s'], 1),
        }
        for row in manifest['runs'][noise_type]['loss_history']
    ]


def _checkpoint_rows(manifest: dict[str, Any], noise_type: str) -> list[dict[str, Any]]:
    return [
        {'name': payload['name'], 'path': payload['relative']}
        for payload in manifest['runs'][noise_type]['files'].get('checkpoint_files', [])
    ]


def _figure_payload(manifest: dict[str, Any], figure_name: str) -> tuple[str, str | None]:
    payload = manifest['figures'][figure_name]
    caption = payload.get('caption', FIGURE_CAPTIONS.get(figure_name, 'Saved figure from the executed notebooks.'))
    note = '\n'.join([f"## {figure_name}", '', caption, '', f"Artifact path: `{payload['relative']}`"])
    return note, payload['absolute']


def _saved_sample_rows(saved_sample_audit: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not saved_sample_audit:
        return []
    rows = []
    for noise_type, run in saved_sample_audit['runs'].items():
        rows.append(
            {
                'distribution': noise_type.title(),
                'avg_confidence': round(run['avg_confidence'], 4),
                'class_coverage': run['class_coverage'],
                'prediction_entropy': round(run['prediction_entropy'], 4),
                'prototype_mse': round(run['prototype_mse'], 5),
                'feature_diversity_l2': round(run['feature_diversity_l2'], 4),
                'foreground_ratio': round(run['foreground_ratio'], 4),
                'sharpness_score': round(run['sharpness_score'], 4),
            }
        )
    return rows


def _saved_sample_rankings(saved_sample_audit: dict[str, Any] | None, metric: str, pd):
    if not saved_sample_audit:
        return 'No saved-sample audit found.', pd.DataFrame([])
    ranking = saved_sample_audit['rankings'][metric]
    note = ranking[0]['label'] if ranking else metric
    rows = [{'rank': row['rank'], 'distribution': row['noise_type'].title(), 'value': round(row['value'], 5)} for row in ranking]
    return note, pd.DataFrame(rows)


def _nn_rows(nn_audit: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not nn_audit:
        return []
    rows = []
    for noise_type, run in nn_audit['runs'].items():
        rows.append(
            {
                'distribution': noise_type.title(),
                'avg_confidence': round(run['avg_confidence'], 4),
                'avg_nn_distance': round(run['avg_nn_distance'], 4),
                'label_coverage': run['matched_label_coverage'],
                'label_entropy': round(run['matched_label_entropy'], 4),
                'match_rate': round(run['prediction_match_rate'], 4),
            }
        )
    return rows


def _nn_run_payload(nn_audit: dict[str, Any] | None, noise_type: str) -> tuple[str, str | None]:
    if not nn_audit:
        return 'No nearest-neighbor audit found.', None
    run = nn_audit['runs'][noise_type]
    text = '\n'.join(
        [
            f"## {noise_type.title()} nearest-neighbor audit",
            '',
            f"- Average classifier confidence: `{run['avg_confidence']:.4f}`",
            f"- Average nearest-neighbor distance: `{run['avg_nn_distance']:.4f}`",
            f"- Matched-label coverage: `{run['matched_label_coverage']}`",
            f"- Matched-label entropy: `{run['matched_label_entropy']:.4f}`",
            f"- Prediction / nearest-label agreement: `{run['prediction_match_rate']:.4f}`",
            f"- Top matched labels: `{run['top_matched_labels']}`",
        ]
    )
    return text, run['montage_path']


def _training_dynamics_rows(td_audit: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not td_audit:
        return []
    rows = []
    for row in td_audit['summary_rows']:
        rows.append(
            {
                'distribution': row['noise_type'].title(),
                'initial_loss': round(row['initial_epoch_loss'], 5),
                'final_loss': round(row['final_epoch_loss'], 5),
                'relative_improvement_pct': round(row['relative_improvement_pct'], 2),
                'best_epoch': row['best_epoch'],
                'late_stage_std': round(row['late_stage_std'], 5),
                'auc_epoch_loss': round(row['auc_epoch_loss'], 4),
            }
        )
    return rows


def build_app(artifacts_root=None):
    gr, pd = _require_gradio()
    manifest = build_manifest(artifacts_root)
    repo_root = Path(manifest['repo_root'])
    output_dir = repo_root / 'research_extension_output'
    saved_sample_audit = _load_optional_json(output_dir / 'saved_sample_audit.json')
    nearest_neighbor_audit = _load_optional_json(output_dir / 'nearest_neighbor_audit.json')
    training_dynamics_audit = _load_optional_json(output_dir / 'training_dynamics_audit.json')

    battery_root = Path(manifest['artifacts_root']) / 'battery_sentinel'
    battery_logs_dir = battery_root / 'logs'
    battery_figures_dir = battery_root / 'figures'
    battery_simulation = _load_optional_json(battery_logs_dir / 'simulation_summary.json')
    battery_twin = _load_optional_json(battery_logs_dir / 'twin_training_summary.json')
    battery_router = _load_optional_json(battery_logs_dir / 'router_metrics.json')
    battery_summary = _load_optional_json(battery_logs_dir / 'battery_sentinel_summary.json')

    figure_choices = sorted(manifest['figures'])
    noise_choices = manifest['noise_types']
    default_run = noise_choices[0]

    results_df = pd.DataFrame(_results_rows(manifest))
    controls_df = pd.DataFrame(_controls_rows(manifest))
    saved_sample_df = pd.DataFrame(_saved_sample_rows(saved_sample_audit))
    nn_df = pd.DataFrame(_nn_rows(nearest_neighbor_audit))
    td_df = pd.DataFrame(_training_dynamics_rows(training_dynamics_audit))

    battery_metrics_df = pd.DataFrame(battery_twin.get('test_metrics', [])) if battery_twin else pd.DataFrame([])
    battery_actions_df = pd.DataFrame([
        {'action': key, 'count': value}
        for key, value in (battery_router.get('action_counts', {}) if battery_router else {}).items()
    ])

    def show_run(noise_type: str):
        run = manifest['runs'][noise_type]
        sample_path = run['files'].get('sample_image_absolute')
        return (
            _run_markdown(manifest, noise_type),
            sample_path,
            pd.DataFrame(_loss_history_rows(manifest, noise_type)),
            pd.DataFrame(_checkpoint_rows(manifest, noise_type)),
        )

    def show_figure(figure_name: str):
        note, path = _figure_payload(manifest, figure_name)
        return note, path

    def show_saved_ranking(metric: str):
        return _saved_sample_rankings(saved_sample_audit, metric, pd)

    def show_nn_run(noise_type: str):
        return _nn_run_payload(nearest_neighbor_audit, noise_type)

    overview_text = '\n'.join(
        [
            f"# {manifest['project_title']}",
            '',
            manifest['objective'],
            '',
            f"- Artifact root: `{manifest['artifact_root_relative']}`",
            f"- Shared controls verified: `{manifest['shared_controls_verified']}`",
            f"- Full comparison ready: `{manifest['full_comparison_ready']}`",
            f"- Reverse-sampler note: {manifest['sampler_note']}",
            f"- Battery Sentinel ready: `{battery_summary is not None}`",
        ]
    )

    gallery_names = [
        'summary_figure.png', 'campaign_loss_overview.png', 'campaign_samples.png',
        'metric_overview.png', 'trajectories.png', 'kurtosis_diagnostic.png'
    ]
    gallery_items = [(manifest['figures'][name]['absolute'], name) for name in gallery_names if name in manifest['figures']]

    with gr.Blocks(title=manifest['project_title']) as demo:
        gr.Markdown(overview_text)

        with gr.Tab('Dashboard'):
            gr.Dataframe(value=results_df, label='Comparative metric ledger', interactive=False)
            with gr.Accordion('Controlled configuration', open=False):
                gr.Dataframe(value=controls_df, label='Configuration', interactive=False)
            summary_path = manifest['figures'].get('summary_figure.png', {}).get('absolute')
            if summary_path:
                gr.Image(value=summary_path, label='Summary dashboard', type='filepath')
            gr.Gallery(value=gallery_items, label='Campaign figure set', columns=2, preview=True, object_fit='contain')

        with gr.Tab('Run Profiles'):
            run_selector = gr.Dropdown(choices=noise_choices, value=default_run, label='Distribution')
            run_md = gr.Markdown()
            run_sample = gr.Image(label='Saved sample grid', type='filepath')
            loss_table = gr.Dataframe(label='Logged loss history', interactive=False)
            checkpoint_table = gr.Dataframe(label='Checkpoint ledger', interactive=False)
            run_selector.change(show_run, inputs=run_selector, outputs=[run_md, run_sample, loss_table, checkpoint_table])
            demo.load(show_run, inputs=run_selector, outputs=[run_md, run_sample, loss_table, checkpoint_table])

        with gr.Tab('Saved Sample Audit'):
            gr.Dataframe(value=saved_sample_df, label='Saved sample metrics', interactive=False)
            metric_choices = [
                ('Average classifier confidence', 'avg_confidence'),
                ('Predicted class coverage', 'class_coverage'),
                ('Prediction entropy', 'prediction_entropy'),
                ('Prototype MSE', 'prototype_mse'),
                ('Feature diversity', 'feature_diversity_l2'),
                ('Sharpness score', 'sharpness_score'),
            ]
            saved_metric = gr.Dropdown(choices=metric_choices, value='avg_confidence', label='Ranking metric')
            saved_note = gr.Markdown()
            saved_rank = gr.Dataframe(label='Metric ordering', interactive=False)
            saved_metric.change(show_saved_ranking, inputs=saved_metric, outputs=[saved_note, saved_rank])
            demo.load(show_saved_ranking, inputs=saved_metric, outputs=[saved_note, saved_rank])

        with gr.Tab('Nearest-Neighbor Audit'):
            gr.Dataframe(value=nn_df, label='Nearest-neighbor summary', interactive=False)
            overview_fig = None
            if nearest_neighbor_audit:
                overview_fig = nearest_neighbor_audit.get('overview_figure')
            if overview_fig:
                gr.Image(value=overview_fig, label='Nearest-neighbor overview', type='filepath')
            nn_selector = gr.Dropdown(choices=noise_choices, value=default_run, label='Distribution')
            nn_text = gr.Markdown()
            nn_montage = gr.Image(label='Nearest-neighbor montage', type='filepath')
            nn_selector.change(show_nn_run, inputs=nn_selector, outputs=[nn_text, nn_montage])
            demo.load(show_nn_run, inputs=nn_selector, outputs=[nn_text, nn_montage])

        with gr.Tab('Training Dynamics'):
            gr.Dataframe(value=td_df, label='Training dynamics summary', interactive=False)
            td_fig = training_dynamics_audit.get('figure') if training_dynamics_audit else None
            if td_fig:
                gr.Image(value=td_fig, label='Training dynamics audit', type='filepath')

        with gr.Tab('Battery Sentinel'):
            if battery_summary:
                battery_text = '\n'.join([
                    '# Battery Sentinel',
                    '',
                    'Applied continuation of the tri-noise study for EV battery monitoring.',
                    '',
                    f"- Simulated sessions: `{battery_simulation.get('num_sessions', 'n/a') if battery_simulation else 'n/a'}`",
                    f"- Sequence length: `{battery_simulation.get('sequence_length', 'n/a') if battery_simulation else 'n/a'}`",
                    f"- Router macro-F1: `{battery_router.get('macro_f1', 0.0):.4f}`",
                    f"- Binary baseline F1: `{battery_router.get('baseline_binary_f1', 0.0):.4f}`",
                ])
                gr.Markdown(battery_text)
                if not battery_metrics_df.empty:
                    gr.Dataframe(value=battery_metrics_df, label='Twin evaluation by regime', interactive=False)
                if not battery_actions_df.empty:
                    gr.Dataframe(value=battery_actions_df, label='Recommended actions', interactive=False)
                battery_gallery_items = []
                for name in ['battery_regime_examples.png', 'battery_twin_training_curves.png', 'battery_router_confusion.png', 'battery_router_case_studies.png', 'battery_sentinel_dashboard.png']:
                    candidate = battery_figures_dir / name
                    if candidate.exists():
                        battery_gallery_items.append((str(candidate.resolve()), name))
                if battery_gallery_items:
                    gr.Gallery(value=battery_gallery_items, label='Battery Sentinel figures', columns=2, preview=True, object_fit='contain')
            else:
                gr.Markdown('Battery Sentinel outputs were not found yet. Run notebooks `7` to `10` to populate this tab.')

        with gr.Tab('Figure Atlas'):
            figure_selector = gr.Dropdown(choices=figure_choices, value='summary_figure.png', label='Figure')
            figure_note = gr.Markdown()
            figure_view = gr.Image(label='Saved figure', type='filepath')
            figure_selector.change(show_figure, inputs=figure_selector, outputs=[figure_note, figure_view])
            demo.load(show_figure, inputs=figure_selector, outputs=[figure_note, figure_view])

        with gr.Tab('Artifact Ledger'):
            figure_df = pd.DataFrame([
                {'name': name, 'path': payload['relative'], 'caption': payload.get('caption', '')}
                for name, payload in sorted(manifest['figures'].items())
            ])
            log_df = pd.DataFrame([
                {'name': name, 'path': payload['relative']}
                for name, payload in sorted(manifest['logs'].items())
            ])
            checkpoint_df = pd.DataFrame([
                {
                    'distribution': manifest['runs'][noise_type]['label'],
                    'checkpoint_dir': manifest['runs'][noise_type]['files']['checkpoint_dir'],
                    'saved_checkpoints': manifest['runs'][noise_type]['files']['checkpoint_count'],
                    'tensorboard_dir': manifest['runs'][noise_type]['files']['tensorboard_dir'],
                    'tensorboard_files': manifest['runs'][noise_type]['files']['tensorboard_file_count'],
                }
                for noise_type in manifest['noise_types']
            ])
            gr.Dataframe(value=figure_df, label='Figures', interactive=False)
            gr.Dataframe(value=log_df, label='Logs', interactive=False)
            gr.Dataframe(value=checkpoint_df, label='Checkpoint and TensorBoard directories', interactive=False)

    return demo


def launch_app(artifacts_root=None, host: str = '127.0.0.1', port: int = 9000, share: bool = False) -> None:
    demo = build_app(artifacts_root)
    demo.launch(server_name=host, server_port=port, share=share)
