from strong_reject.load_datasets import load_strongreject_small, load_strongreject
from strong_reject.jailbreaks import apply_jailbreaks_to_dataset, register_jailbreak
from strong_reject.generate import generate_to_dataset
from strong_reject.evaluate import evaluate_dataset
import pdb


# load the small version of the StrongREJECT dataset
forbidden_prompt_dataset = load_strongreject()

pdb.set_trace()

