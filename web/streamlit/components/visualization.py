from typing import Dict, Any
import plotly.express as px
import pandas as pd

def visualize_predictions(predictions: Dict[str, Any]) -> None:
    """Visualize the predictions from the models."""
    df = pd.DataFrame(predictions.items(), columns=['Model', 'Prediction'])
    
    fig = px.bar(df, x='Model', y='Prediction', title='Exoplanet Classification Predictions',
                 labels={'Prediction': 'Predicted Class', 'Model': 'Model Name'})
    
    fig.update_layout(yaxis=dict(tickvals=[0, 1, 2], ticktext=["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]))
    
    fig.show()