import shap
import matplotlib.pyplot as plt

def plot_shap_beeswarm(model, X, max_display=10):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    fig = shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    return fig
