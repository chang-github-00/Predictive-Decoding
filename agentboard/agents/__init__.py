from .vanilla_agent import VanillaAgent
from .react_agent import ReactAgent
from .lookahead_agent import LookAheadAgent 
from .lookahead_eval_agent import LookAheadEvalAgent
from .lookahead_eval_agent_ablation import LookAheadEvalAgentAblation
from .chain_of_thought_agent import COTAgent
from common.registry import registry

__all__ = ["VanillaAgent", "ReactAgent"]


def load_agent(name, config, llm_model):
    agent = registry.get_agent_class(name).from_config(llm_model, config)
    return agent
