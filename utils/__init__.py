"""Crop prediction utilities package."""

from .load_data import (load_models, prepare_input_data, make_prediction, get_feature_importance,
                       get_shap_explanation, create_shap_plot, explain_crop_prediction)

from .crop_advice import (get_crop_specific_advice, get_fertilizer_recommendations)

__all__ = [
    'load_models',
    'prepare_input_data',
    'make_prediction',
    'get_feature_importance',
    'get_shap_explanation',
    'create_shap_plot',
    'explain_crop_prediction',
    'get_crop_specific_advice',
    'get_fertilizer_recommendations'
]