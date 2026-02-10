import plotly.graph_objects as go
import pandas as pd

# Data
data = [
    {"config": "128-0.05-3-0.001", "NDCG@10": 0.271885, "Precision@10": 0.268407, "Recall@10": 0.145600},
    {"config": "100-0.05-5-0.001", "NDCG@10": 0.271737, "Precision@10": 0.267422, "Recall@10": 0.145218},
    {"config": "100-0.05-3-0.001", "NDCG@10": 0.271659, "Precision@10": 0.267882, "Recall@10": 0.146606},
    {"config": "128-0.05-10-0.001", "NDCG@10": 0.271259, "Precision@10": 0.268686, "Recall@10": 0.145867},
    {"config": "128-0.05-5-0.001", "NDCG@10": 0.271237, "Precision@10": 0.268013, "Recall@10": 0.144660},
    {"config": "100-0.05-10-0.001", "NDCG@10": 0.269974, "Precision@10": 0.265517, "Recall@10": 0.143950},
    {"config": "64-0.05-5-0.001", "NDCG@10": 0.267823, "Precision@10": 0.264910, "Recall@10": 0.143695},
    {"config": "64-0.05-10-0.001", "NDCG@10": 0.267504, "Precision@10": 0.264302, "Recall@10": 0.143140},
    {"config": "64-0.05-3-0.001", "NDCG@10": 0.266193, "Precision@10": 0.264236, "Recall@10": 0.142812},
    {"config": "128-0.1-10-0.001", "NDCG@10": 0.259061, "Precision@10": 0.257291, "Recall@10": 0.146537}
]

df = pd.DataFrame(data)

# Create grouped bar chart
fig = go.Figure()

# Add bars for each metric using brand colors
fig.add_trace(go.Bar(
    name='NDCG@10',
    x=df['config'],
    y=df['NDCG@10'],
    marker_color='#1FB8CD',
    text=[f'{val:.3f}' for val in df['NDCG@10']],
    textposition='none',
    hovertemplate='<b>%{x}</b><br>NDCG@10: %{y:.3f}<extra></extra>'
))

fig.add_trace(go.Bar(
    name='Precision@10',
    x=df['config'],
    y=df['Precision@10'],
    marker_color='#DB4545',
    text=[f'{val:.3f}' for val in df['Precision@10']],
    textposition='none',
    hovertemplate='<b>%{x}</b><br>Precision@10: %{y:.3f}<extra></extra>'
))

fig.add_trace(go.Bar(
    name='Recall@10',
    x=df['config'],
    y=df['Recall@10'],
    marker_color='#2E8B57',
    text=[f'{val:.3f}' for val in df['Recall@10']],
    textposition='none',
    hovertemplate='<b>%{x}</b><br>Recall@10: %{y:.3f}<extra></extra>'
))

# Update layout
fig.update_layout(

    xaxis_title="Configuration",
    yaxis_title="Metric Value",
    barmode='group',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5
    )
)

# Set y-axis range and format
fig.update_yaxes(range=[0, 0.30], tickformat='.2f')

# Angle x-axis labels for readability
fig.update_xaxes(tickangle=-45)

# Clip on axis
fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image("bestConfig.png")
fig.write_image("bestConfig.svg", format="svg")
fig.write_image("bestConfig.pdf", format="pdf")