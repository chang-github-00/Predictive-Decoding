from .generation import Generation
from .cot import COT  
from .tot import TOT
from common.registry import registry


def load_algorithm(name, config, llm_model):
    algorithm = registry.get_algorithm_class(name).from_config(llm_model, config)
    return algorithm