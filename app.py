import gradio as gr
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def train_estimators(max_depth,n_estimators):
    rng = np.random.RandomState(1)
    X = np.linspace(0, 6, 100)[:, np.newaxis]
    y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

    regr_1 = DecisionTreeRegressor(max_depth=4)
    regr_2 = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=max_depth), n_estimators=n_estimators, random_state=rng
    )
    regr_1.fit(X, y)
    regr_2.fit(X, y)
    y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)
    colors = sns.color_palette("colorblind")

    fig, ax = plt.subplots()
    ax.scatter(X, y, color=colors[0], label="training samples")
    ax.plot(X, y_1, color=colors[1], label="Decision tree (max_depth=4)", linewidth=2)
    ax.plot(X, y_2, color=colors[2], label=f"Adaboost (max_depth={max_depth}, estimators={n_estimators})", linewidth=2)
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.legend()
    return fig

title = "Decision Tree Regression with AdaBoost"
with gr.Blocks(title=title) as demo:
    gr.Markdown(f"## {title}")
    gr.Markdown("This app demonstrates bosting of decision tree regressor using Adaboost")

    max_depth = gr.Slider(minimum=1, maximum=50, step=1, label = "Maximum Depth")
    n_estimators = gr.Slider(minimum=1, maximum=300, step=1, label = "Number of Estimators")

    plot = gr.Plot(label=title)
    n_estimators.change(fn=train_estimators, inputs=[max_depth,n_estimators], outputs=[plot])
    max_depth.change(fn=train_estimators, inputs=[max_depth,n_estimators], outputs=[plot])

demo.launch()

