"""
GUI-приложение для предсказания параметров популяционного алгоритма
"""
from .app import population_algorithmApp
from .function_parser import FunctionParser
from .predictor import CascadePredictor, load_predictor, predict_params
from .feature_info import FEATURE_INFO
from .tooltip import ToolTip
from .clipboard import ClipboardManager