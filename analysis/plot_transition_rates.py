"""Plot stance transition rates of the revision strategies from results/*.csv.

Writes figures/transition_rates_gpt-5.2.png (GPT-5.2 simulator) and
figures/transition_rates_three_simulators.png (GPT-5.2, Sonnet-4.6, Qwen3.5-Plus).
Rate = count of a transition / number of rows whose inferred stance is negative or neutral (per target).

Usage: python analysis/plot_transition_rates.py
"""
import argparse
import math
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MODEL_COL = 'topic'
INFER_COL = 'inferred'
REVISION_COLS = {'Add': 'add', 'Explain': 'explain', 'Paraphrase': 'paraphrase'}
MEME_REVISION_COLS = {'Meme': 'meme'}

STRATEGY_ORDER = ['Paraphrase', 'Explain', 'Add', 'Meme']
INFER_ORDER = ['negative', 'neutral']
LABEL_THRESHOLD = 2.0

STANCE_SCORE = {'negative': -1, 'neutral': 0, 'positive': 1}
STANCE_COLORS = {'negative': '#d73027', 'neutral': '#8c8c8c', 'positive': '#1a9850'}

PANELS = [
    ('gpt-5.2-2025-12-11', 'GPT-5.2 infer stance'),
    ('claude-sonnet-4-6', 'Sonnet-4.6 infer stance'),
    ('qwen3.5-plus-2026-02-15', 'Qwen3.5-Plus infer stance'),
]


def load_tables(results_dir, simulator):
    return (pd.read_csv(os.path.join(results_dir, f'{simulator}_text.csv')),
            pd.read_csv(os.path.join(results_dir, f'{simulator}_meme.csv')))


def build_shift_grouped_df(df, meme_df):
    all_rows = []

    def append_shift_rows(source_df, strategy_cols):
        for model_name in source_df[MODEL_COL].dropna().unique():
            denom = len(source_df[(source_df[MODEL_COL] == model_name) & (source_df[INFER_COL].isin(INFER_ORDER))])
            if denom == 0:
                continue
            for strategy_name, revision_col in strategy_cols.items():
                temp = source_df[[MODEL_COL, INFER_COL, revision_col]].dropna().copy()
                temp = temp[
                    (temp[MODEL_COL] == model_name) &
                    (temp[INFER_COL].isin(INFER_ORDER)) &
                    (temp[revision_col].isin(STANCE_SCORE.keys()))
                ]
                temp['strategy'] = strategy_name
                temp['revision_stance'] = temp[revision_col]
                temp['denom'] = denom
                temp['change'] = temp['revision_stance'].map(STANCE_SCORE) - temp[INFER_COL].map(STANCE_SCORE)
                temp = temp[temp['change'] != 0]
                all_rows.append(temp[[MODEL_COL, INFER_COL, 'strategy', 'revision_stance', 'change', 'denom']])

    append_shift_rows(df, REVISION_COLS)
    append_shift_rows(meme_df, MEME_REVISION_COLS)
    plot_df = pd.concat(all_rows, ignore_index=True)

    grouped = (
        plot_df
        .groupby([MODEL_COL, 'strategy', INFER_COL, 'revision_stance', 'change', 'denom'])
        .size()
        .reset_index(name='count')
    )
    grouped['signed_percent'] = grouped.apply(
        lambda x: (x['count'] / x['denom']) * 100 if x['change'] > 0 else -(x['count'] / x['denom']) * 100,
        axis=1
    )
    grouped['percent_label'] = grouped['signed_percent'].abs().round(1).astype(str) + '%'
    return grouped


def add_shift_panel(fig, grouped, xaxis_title, row=None, col=1, showlegend=True, xaxis_title_standoff=70):
    model_order = list(grouped[MODEL_COL].dropna().unique())

    model_gap = 1.45
    strategy_gap = 0.30
    infer_offset = {'negative': -0.06, 'neutral': 0.06}
    model_base = {model: i * model_gap for i, model in enumerate(model_order)}
    strategy_offset = {'Paraphrase': 0.00, 'Explain': strategy_gap, 'Add': strategy_gap * 2, 'Meme': strategy_gap * 3}
    strategy_band_colors = {
        'Paraphrase': 'rgba(0, 0, 0, 0.05)',
        'Explain': 'rgba(0, 0, 0, 0.15)',
        'Add': 'rgba(0, 0, 0, 0.35)',
        'Meme': 'rgba(0, 0, 0, 0.45)',
    }

    strategy_band_edges = {}
    for strategy in STRATEGY_ORDER:
        center = strategy_offset[strategy]
        left_neighbors = [offset for offset in strategy_offset.values() if offset < center]
        right_neighbors = [offset for offset in strategy_offset.values() if offset > center]
        left_edge = (max(left_neighbors) + center) / 2 if left_neighbors else center - strategy_gap / 2
        right_edge = (center + min(right_neighbors)) / 2 if right_neighbors else center + strategy_gap / 2
        strategy_band_edges[strategy] = (left_edge, right_edge)

    def get_x(model, strategy, infer):
        return model_base[model] + strategy_offset[strategy] + infer_offset[infer]

    revised_stance_order = ['neutral', 'positive', 'negative']
    text_colors = {'neutral': '#f2f2f2', 'positive': '#f2f2f2', 'negative': '#f2f2f2'}
    axis_index = row if row is not None else 1
    xref = 'x' if axis_index == 1 else f'x{axis_index}'
    yref = 'y' if axis_index == 1 else f'y{axis_index}'
    ydomain_ref = f'{yref} domain'

    for revised_stance in revised_stance_order:
        temp = grouped[grouped['revision_stance'] == revised_stance]
        x_vals = [get_x(m, s, i) for m, s, i in zip(temp[MODEL_COL], temp['strategy'], temp[INFER_COL])]
        bar = go.Bar(
            name=f'simulated {revised_stance}',
            x=x_vals,
            y=temp['signed_percent'],
            marker_color=STANCE_COLORS[revised_stance],
            legendgroup=revised_stance,
            showlegend=showlegend,
            text=[
                r['percent_label'] if abs(r['signed_percent']) >= LABEL_THRESHOLD else ''
                for _, r in temp.iterrows()
            ],
            textposition='inside',
            width=0.10,
            textfont=dict(size=11, color=text_colors[revised_stance])
        )
        if row is None:
            fig.add_trace(bar)
        else:
            fig.add_trace(bar, row=row, col=col)

    # Labels of small bars are drawn above/below the bar stack.
    label_gap = max(grouped['signed_percent'].abs().max() * 0.012, 0.08)
    stack_offsets = {}
    for revised_stance in revised_stance_order:
        temp = grouped[grouped['revision_stance'] == revised_stance]
        for _, r in temp.iterrows():
            bar_key = (r[MODEL_COL], r['strategy'], r[INFER_COL])
            stack_key = (bar_key, 'positive' if r['signed_percent'] > 0 else 'negative')
            stack_start = stack_offsets.get(stack_key, 0)
            stack_end = stack_start + r['signed_percent']
            stack_offsets[stack_key] = stack_end
            if abs(r['signed_percent']) >= LABEL_THRESHOLD:
                continue
            x = get_x(r[MODEL_COL], r['strategy'], r[INFER_COL])
            block_bottom = min(stack_start, stack_end)
            block_top = max(stack_start, stack_end)
            y = block_top + label_gap if r['signed_percent'] > 0 else block_bottom - label_gap
            yanchor = 'bottom' if r['signed_percent'] > 0 else 'top'
            fig.add_annotation(x=x, y=y, xref=xref, yref=yref, text=r['percent_label'], showarrow=False,
                               xanchor='center', yanchor=yanchor, font=dict(size=9, color='black'))

    tickvals = []
    ticktext = []
    strategy_centers = []
    for model in model_order:
        for strategy in STRATEGY_ORDER:
            left = get_x(model, strategy, 'negative')
            right = get_x(model, strategy, 'neutral')
            strategy_centers.append(((left + right) / 2, strategy.lower()))
            for infer in INFER_ORDER:
                tickvals.append(get_x(model, strategy, infer))
                ticktext.append(infer)

    model_centers = []
    for model in model_order:
        left = get_x(model, 'Paraphrase', 'negative')
        right = get_x(model, 'Meme', 'neutral')
        model_centers.append(((left + right) / 2, model))

    y_min = grouped['signed_percent'].min()
    y_max = grouped['signed_percent'].max()
    y_lower = min(0, y_min) * 1.18
    y_upper = max(0, y_max) * 1.18
    tick_step = 5 if y_upper - y_lower > 15 else 2
    tick_start = math.floor(y_lower / tick_step) * tick_step
    tick_end = math.ceil(y_upper / tick_step) * tick_step
    y_ticks = list(range(int(tick_start), int(tick_end) + tick_step, tick_step))

    xaxis = dict(
        title=xaxis_title,
        title_standoff=xaxis_title_standoff,
        tickmode='array',
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=45,
        tickfont=dict(size=11),
        range=[min(tickvals) - 0.14, max(tickvals) + 0.14],
        showline=True,
        linecolor='black',
        ticks='outside'
    )
    yaxis = dict(
        title='Stance shift rate (%)',
        tickmode='array',
        tickvals=y_ticks,
        ticktext=[f'{abs(tick)}%' for tick in y_ticks],
        range=[tick_start, tick_end],
        showgrid=True,
        gridcolor='rgba(0,0,0,0.12)',
        griddash='dot',
        zeroline=False,
        showline=True,
        linecolor='black',
        ticks='outside'
    )
    if row is None:
        fig.update_layout(xaxis=xaxis, yaxis=yaxis)
    else:
        fig.update_xaxes(**xaxis, row=row, col=col)
        fig.update_yaxes(**yaxis, row=row, col=col)

    for model in model_order:
        for strategy in STRATEGY_ORDER:
            left_offset, right_offset = strategy_band_edges[strategy]
            fig.add_shape(type='rect', xref=xref, yref=ydomain_ref,
                          x0=model_base[model] + left_offset, x1=model_base[model] + right_offset, y0=0, y1=1,
                          fillcolor=strategy_band_colors[strategy], line=dict(width=0), layer='below')

    direction_label_x = min(tickvals) - 0.11
    direction_label_font = dict(size=14, color='black')
    fig.add_annotation(x=direction_label_x, y=tick_end - (tick_end - tick_start) * 0.04, xref=xref, yref=yref,
                       text='↑ change to positive/neutral', showarrow=False, xanchor='left', yanchor='middle',
                       font=direction_label_font)
    fig.add_annotation(x=direction_label_x, y=tick_start + (tick_end - tick_start) * 0.04, xref=xref, yref=yref,
                       text='↓ change to negative', showarrow=False, xanchor='left', yanchor='middle',
                       font=direction_label_font)
    for center, strategy in strategy_centers:
        fig.add_annotation(x=center, y=-0.20, xref=xref, yref=ydomain_ref, text=strategy, showarrow=False,
                           font=dict(size=12))
    for center, model in model_centers:
        fig.add_annotation(x=center, y=-0.28, xref=xref, yref=ydomain_ref, text=f'<b>{model}</b>', showarrow=False,
                           font=dict(size=13))

    if row is None:
        fig.add_hline(y=0, line_width=1, line_color='black')
    else:
        fig.add_hline(y=0, line_width=1, line_color='black', row=row, col=col)
    return tick_start, tick_end


def apply_shift_layout(fig, width, height, bottom_margin=150, top_margin=20):
    fig.update_layout(
        template='simple_white',
        barmode='relative',
        width=width,
        height=height,
        legend_title_text='',
        legend=dict(x=0.98, y=0.995, xanchor='right', yanchor='top', bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='rgba(0,0,0,0.25)', borderwidth=0.5),
        bargap=0,
        bargroupgap=0,
        margin=dict(l=85, r=35, b=bottom_margin, t=top_margin),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )


def plot_single(results_dir, output_path):
    grouped = build_shift_grouped_df(*load_tables(results_dir, PANELS[0][0]))
    fig = go.Figure()
    add_shift_panel(fig, grouped, xaxis_title='infer stance')
    apply_shift_layout(fig, width=1400, height=560)
    fig.write_image(output_path, scale=2)


def plot_three_simulators(results_dir, output_path):
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.12)
    for row, (simulator, xaxis_title) in enumerate(PANELS, start=1):
        grouped = build_shift_grouped_df(*load_tables(results_dir, simulator))
        add_shift_panel(fig, grouped, xaxis_title=xaxis_title, row=row, showlegend=(row == 1))
    apply_shift_layout(fig, width=1400, height=1680)
    fig.write_image(output_path, scale=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot stance transition rates (needs plotly and kaleido).')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--output_dir', default='figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for name, plot in [('transition_rates_gpt-5.2.png', plot_single),
                       ('transition_rates_three_simulators.png', plot_three_simulators)]:
        path = os.path.join(args.output_dir, name)
        plot(args.results_dir, path)
        print(f'Saved {path}')
