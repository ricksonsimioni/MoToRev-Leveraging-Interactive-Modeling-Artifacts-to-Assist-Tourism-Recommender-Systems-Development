import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Data
data = [
{"factors": 128, "groupSize": 3, "avgNDCG": 0.271885},
{"factors": 100, "groupSize": 5, "avgNDCG": 0.271737},
{"factors": 100, "groupSize": 3, "avgNDCG": 0.271659},
{"factors": 128, "groupSize": 10, "avgNDCG": 0.271259},
{"factors": 128, "groupSize": 5, "avgNDCG": 0.271237},
{"factors": 100, "groupSize": 10, "avgNDCG": 0.269974},
{"factors": 64, "groupSize": 5, "avgNDCG": 0.267823},
{"factors": 64, "groupSize": 10, "avgNDCG": 0.267504},
{"factors": 64, "groupSize": 3, "avgNDCG": 0.266193}
]

df = pd.DataFrame(data)

# Pivot to create matrix for heatmap
heatmap_data = df.pivot(index='factors', columns='groupSize', values='avgNDCG')

# Sort to ensure proper order
heatmap_data = heatmap_data.sort_index(ascending=True)  # factors: 64, 100, 128
heatmap_data = heatmap_data[sorted(heatmap_data.columns)]  # groupSize: 3, 5, 10

# Create heatmap
fig = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale='Blues',  # Darker = better
    text=np.round(heatmap_data.values, 6),
    texttemplate='%{text}',
    textfont={"size": 12},
    colorbar=dict(title="avgNDCG"),
    hovertemplate='Factors: %{y}<br>Group Size: %{x}<br>avgNDCG: %{z:.6f}<extra></extra>'
))

fig.update_xaxes(title_text="Group Size")
fig.update_yaxes(title_text="Factors")

# Save as PNG and SVG
fig.write_image("heatmap.png")
fig.write_image("heatmap.svg", format="svg")
fig.write_image("heatmap.pdf", format="pdf")