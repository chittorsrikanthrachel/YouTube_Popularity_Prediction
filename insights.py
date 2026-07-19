from model_views import predict_views
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import io
import base64

def heatmap_insight(new_video):
    # Get predicted views from all models
    result = predict_views(new_video)

    # Convert K → number
    def clean(val):
        val = str(val).upper().replace("K", "")
        return float(val) * 1000

    predicted_views = np.mean([clean(v) for v in result.values()])

    # Days 1-7, hours ±3
    days = np.arange(1, 8)
    hours = np.arange(new_video["publish_hour"] - 3, new_video["publish_hour"] + 4)

    data = []
    for d in days:
        for h in hours:
            score = predicted_views * (1 + ((d - new_video["publish_day"]) * 0.05)) * \
                                    (1 + ((h - new_video["publish_hour"]) * 0.03))
            data.append([d, h, score])

    df = pd.DataFrame(data, columns=["publish_day", "publish_hour", "views"])
    heatmap_df = df.pivot(index="publish_day", columns="publish_hour", values="views")

    fig = px.imshow(
        heatmap_df,
        labels=dict(x="Publish Hour", y="Publish Day", color="Predicted Views"),
        title="Heatmap: Predicted Views vs Publish Time",
        width=750,
        height=550
    )

    fig.update_traces(
        hovertemplate="Publish Day: %{y}<br>Publish Hour: %{x}<br>Predicted Views: %{z:.0f}<extra></extra>"
    )

    return fig

def avg_views_hour_day(new_video):
    # First generate same heatmap dataframe
    result = predict_views(new_video)
    def clean(val):
        val = str(val).upper().replace("K", "")
        return float(val) * 1000
    predicted_views = np.mean([clean(v) for v in result.values()])

    days = np.arange(1, 8)
    hours = np.arange(new_video["publish_hour"] - 3, new_video["publish_hour"] + 4)
    data = []
    for d in days:
        for h in hours:
            score = predicted_views * (1 + ((d - new_video["publish_day"]) * 0.05)) * \
                                    (1 + ((h - new_video["publish_hour"]) * 0.03))
            data.append([d, h, score])
    df = pd.DataFrame(data, columns=["publish_day", "publish_hour", "views"])

    # Average by hour/day
    hour_views = df.groupby('publish_hour')['views'].mean()
    day_views = df.groupby('publish_day')['views'].mean()

    fig, axs = plt.subplots(1, 2, figsize=(14,5))

    axs[0].plot(hour_views.index, hour_views.values, marker='o', color='green')
    axs[0].set_title("Average Views by Hour of Upload")
    axs[0].set_xlabel("Publish Hour (0–23)")
    axs[0].set_ylabel("Average Views")
    axs[0].grid(True)

    axs[1].plot(day_views.index, day_views.values, marker='o', color='green')
    axs[1].set_title("Average Views by Day of Week")
    axs[1].set_xlabel("Publish Day (1–7) 1=Monday")
    axs[1].set_ylabel("Average Views")
    axs[1].grid(True)

    plt.tight_layout()
    return fig

def model_comparison(new_video):
    result = predict_views(new_video)
    def clean(x):
        x = str(x).replace("K", "")
        return float(x) * 1000

    models = list(result.keys())
    values = [clean(result[m]) for m in models]

    df = pd.DataFrame({
        "Model": models,
        "Predicted Views": values
    })

    fig = px.bar(
        df,
        x="Model",
        y="Predicted Views",
        text="Predicted Views",
        title="Model Comparison: Predicted Views",
        color_discrete_sequence=["orange"],
        width=900,
        height=320
    )

    fig.update_traces(
        texttemplate='%{text:.0f}',
        textposition='inside',
        insidetextfont=dict(color="black")
    )
    fig.update_layout(
        yaxis_title="Views",
        xaxis_title="ML Model",
        margin=dict(t=60, b=40, l=30, r=30),
        hoverlabel=dict(font_color="black", bgcolor="white")
    )
    return fig

def engagement_pie(new_video):
    likes = new_video["likes"]
    comments = new_video["comment_count"]
    dislikes = new_video["dislikes"]

    df = pd.DataFrame({
        "Engagement": ["Likes", "Comments", "Dislikes"],
        "Count": [likes, comments, dislikes]
    })

    fig = px.pie(
        df,
        names="Engagement",
        values="Count",
        title="Engagement Breakdown",
        hole=0.4,
        color_discrete_sequence=["#2ECC71", "#3498DB", "#E67E22"],
        width=650,
        height=500
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig
