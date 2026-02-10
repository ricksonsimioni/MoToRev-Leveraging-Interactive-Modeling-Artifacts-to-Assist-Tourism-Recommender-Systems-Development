import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Load the data
data = [
{"learningRate": 0.01, "regularization": 0.001, "avgNDCG": 0.197988, "stdNDCG": 0.001552},
{"learningRate": 0.01, "regularization": 0.001, "avgNDCG": 0.197601, "stdNDCG": 0.002043},
{"learningRate": 0.01, "regularization": 0.001, "avgNDCG": 0.196177, "stdNDCG": 0.002005},
{"learningRate": 0.01, "regularization": 0.001, "avgNDCG": 0.195846, "stdNDCG": 0.002811},
{"learningRate": 0.01, "regularization": 0.001, "avgNDCG": 0.195363, "stdNDCG": 0.002887},
{"learningRate": 0.01, "regularization": 0.001, "avgNDCG": 0.194153, "stdNDCG": 0.002471},
{"learningRate": 0.01, "regularization": 0.001, "avgNDCG": 0.193284, "stdNDCG": 0.002504},
{"learningRate": 0.01, "regularization": 0.001, "avgNDCG": 0.192815, "stdNDCG": 0.001937},
{"learningRate": 0.01, "regularization": 0.001, "avgNDCG": 0.192668, "stdNDCG": 0.001666},
{"learningRate": 0.05, "regularization": 0.001, "avgNDCG": 0.271885, "stdNDCG": 0.006043},
{"learningRate": 0.05, "regularization": 0.001, "avgNDCG": 0.271737, "stdNDCG": 0.002116},
{"learningRate": 0.05, "regularization": 0.001, "avgNDCG": 0.271659, "stdNDCG": 0.004044},
{"learningRate": 0.05, "regularization": 0.001, "avgNDCG": 0.271259, "stdNDCG": 0.003419},
{"learningRate": 0.05, "regularization": 0.001, "avgNDCG": 0.271237, "stdNDCG": 0.004306},
{"learningRate": 0.05, "regularization": 0.001, "avgNDCG": 0.269974, "stdNDCG": 0.004545},
{"learningRate": 0.05, "regularization": 0.001, "avgNDCG": 0.267823, "stdNDCG": 0.002861},
{"learningRate": 0.05, "regularization": 0.001, "avgNDCG": 0.267504, "stdNDCG": 0.004595},
{"learningRate": 0.05, "regularization": 0.001, "avgNDCG": 0.266193, "stdNDCG": 0.003179},
{"learningRate": 0.1, "regularization": 0.001, "avgNDCG": 0.259061, "stdNDCG": 0.004252},
{"learningRate": 0.1, "regularization": 0.001, "avgNDCG": 0.255453, "stdNDCG": 0.003419},
{"learningRate": 0.1, "regularization": 0.001, "avgNDCG": 0.254597, "stdNDCG": 0.004190},
{"learningRate": 0.1, "regularization": 0.001, "avgNDCG": 0.254028, "stdNDCG": 0.003388},
{"learningRate": 0.1, "regularization": 0.001, "avgNDCG": 0.252328, "stdNDCG": 0.005081},
{"learningRate": 0.1, "regularization": 0.001, "avgNDCG": 0.250940, "stdNDCG": 0.003861},
{"learningRate": 0.1, "regularization": 0.001, "avgNDCG": 0.248393, "stdNDCG": 0.004914},
{"learningRate": 0.1, "regularization": 0.001, "avgNDCG": 0.248351, "stdNDCG": 0.005393},
{"learningRate": 0.1, "regularization": 0.001, "avgNDCG": 0.247288, "stdNDCG": 0.005993},
{"learningRate": 0.05, "regularization": 0.01, "avgNDCG": 0.223867, "stdNDCG": 0.004131},
{"learningRate": 0.05, "regularization": 0.01, "avgNDCG": 0.222961, "stdNDCG": 0.002420},
{"learningRate": 0.05, "regularization": 0.01, "avgNDCG": 0.221833, "stdNDCG": 0.003205},
{"learningRate": 0.05, "regularization": 0.01, "avgNDCG": 0.221617, "stdNDCG": 0.003783},
{"learningRate": 0.05, "regularization": 0.01, "avgNDCG": 0.221230, "stdNDCG": 0.003679},
{"learningRate": 0.05, "regularization": 0.01, "avgNDCG": 0.220496, "stdNDCG": 0.003740},
{"learningRate": 0.05, "regularization": 0.01, "avgNDCG": 0.220035, "stdNDCG": 0.005441},
{"learningRate": 0.05, "regularization": 0.01, "avgNDCG": 0.218720, "stdNDCG": 0.004482},
{"learningRate": 0.05, "regularization": 0.01, "avgNDCG": 0.216263, "stdNDCG": 0.003999},
{"learningRate": 0.01, "regularization": 0.01, "avgNDCG": 0.183030, "stdNDCG": 0.005395},
{"learningRate": 0.01, "regularization": 0.01, "avgNDCG": 0.182545, "stdNDCG": 0.003479},
{"learningRate": 0.01, "regularization": 0.01, "avgNDCG": 0.179888, "stdNDCG": 0.004176},
{"learningRate": 0.01, "regularization": 0.01, "avgNDCG": 0.179341, "stdNDCG": 0.001854},
{"learningRate": 0.01, "regularization": 0.01, "avgNDCG": 0.178800, "stdNDCG": 0.002389},
{"learningRate": 0.01, "regularization": 0.01, "avgNDCG": 0.177067, "stdNDCG": 0.002525},
{"learningRate": 0.01, "regularization": 0.01, "avgNDCG": 0.174298, "stdNDCG": 0.004021},
{"learningRate": 0.01, "regularization": 0.01, "avgNDCG": 0.173419, "stdNDCG": 0.002226},
{"learningRate": 0.01, "regularization": 0.01, "avgNDCG": 0.173076, "stdNDCG": 0.005476},
{"learningRate": 0.01, "regularization": 0.1, "avgNDCG": 0.158486, "stdNDCG": 0.002835},
{"learningRate": 0.01, "regularization": 0.1, "avgNDCG": 0.157392, "stdNDCG": 0.003425},
{"learningRate": 0.01, "regularization": 0.1, "avgNDCG": 0.157167, "stdNDCG": 0.002504},
{"learningRate": 0.01, "regularization": 0.1, "avgNDCG": 0.156753, "stdNDCG": 0.002809},
{"learningRate": 0.01, "regularization": 0.1, "avgNDCG": 0.156450, "stdNDCG": 0.003895},
{"learningRate": 0.01, "regularization": 0.1, "avgNDCG": 0.156433, "stdNDCG": 0.003167},
{"learningRate": 0.01, "regularization": 0.1, "avgNDCG": 0.156426, "stdNDCG": 0.002164},
{"learningRate": 0.01, "regularization": 0.1, "avgNDCG": 0.155859, "stdNDCG": 0.003558},
{"learningRate": 0.01, "regularization": 0.1, "avgNDCG": 0.155850, "stdNDCG": 0.002882}
]

df = pd.DataFrame(data)

# Group by learningRate and regularization, calculate mean avgNDCG and mean stdNDCG
grouped = df.groupby(['learningRate', 'regularization']).agg({
    'avgNDCG': 'mean',
    'stdNDCG': 'mean'
}).reset_index()

# Color mapping for regularization values
colors = {0.001: '#1FB8CD', 0.01: '#DB4545', 0.1: '#2E8B57'}
reg_labels = {0.001: 'Reg: 0.001', 0.01: 'Reg: 0.01', 0.1: 'Reg: 0.1'}

fig = go.Figure()

# Add scatter plots for each regularization value
for reg in [0.001, 0.01, 0.1]:
    subset = grouped[grouped['regularization'] == reg]
    fig.add_trace(go.Scatter(
        x=subset['learningRate'],
        y=subset['avgNDCG'],
        error_y=dict(
            type='data',
            array=subset['stdNDCG'],
            visible=True
        ),
        mode='markers',
        marker=dict(size=10, color=colors[reg]),
        name=reg_labels[reg],
        hovertemplate='LR: %{x}<br>NDCG: %{y:.4f}<extra></extra>'
    ))

# Add trend line using all data points
x_all = grouped['learningRate'].values
y_all = grouped['avgNDCG'].values
z = np.polyfit(x_all, y_all, 1)
p = np.poly1d(z)
x_trend = np.linspace(x_all.min(), x_all.max(), 100)
y_trend = p(x_trend)

fig.add_trace(go.Scatter(
    x=x_trend,
    y=y_trend,
    mode='lines',
    line=dict(color='gray', dash='dash', width=2),
    name='Trend',
    hoverinfo='skip'
))

fig.update_layout(

    xaxis_title='Learning Rate',
    yaxis_title='Avg NDCG@10',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image('learningRate.png')
fig.write_image('learningRate.svg', format='svg')
fig.write_image('learningRate.pdf', format='pdf')