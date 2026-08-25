import math
import random

from redsi.formats import formats
from redsi.plugins import Param

PLUGIN_DESCRIPTION = "Randomly assign unique document identifiers from a fixed or relative budget."

PLUGIN_PARAMS = [
    Param(
        "budget",
        "int",
        default=320000,
        help=(
            "Size of the identifier space from which unique document IDs are sampled. "
            "Must be >= number of documents. "
            "Controls the docID entropy and identifier difficulty: "
            "a larger budget increases sparsity and generation difficulty, "
            "while a smaller budget (close to corpus size) reduces entropy. "
            "If set to 0, the budget is automatically set to the number of documents "
            "(minimal collision-free space)."
        ),
    ),
    Param(
        "budget_multiplier",
        "float",
        default=0.0,
        help=(
            "Optional relative identifier budget. If > 0, the identifier budget is "
            "ceil(budget_multiplier * number_of_documents). This can only be used "
            "when budget <= 0, so explicit absolute budgets and relative budgets "
            "do not silently combine."
        ),
    ),
    Param(
        "pad_to_max_length",
        "bool",
        default=False,
        help=(
            "If True, left-pad all identifiers with zeros so that they have "
            "uniform length equal to the maximum possible identifier length "
            "(len(str(budget - 1))). "
            "This removes length variability and isolates entropy effects "
            "from identifier-length bias."
        ),
    ),
]

# Errors
ERR_PARAMS_BUDGET = "naive: budget ({budget}) must be >= number of documents ({n})"
ERR_PARAMS_BUDGET_CONFLICT = (
    "naive: budget_multiplier can only be used with budget <= 0 "
    "(got budget={budget}, budget_multiplier={budget_multiplier})"
)
ERR_PARAMS_BUDGET_MULTIPLIER = "naive: budget_multiplier ({budget_multiplier}) must be >= 0"


class NaiveIdentifierPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def run(self, documents, params, seed):
        num_documents = len(documents)
        budget = resolve_budget(num_documents, params.budget, params.budget_multiplier)

        chosen = make_random_unique_assignment(num_documents, budget, seed)
        formatter = build_identifier_formatter(budget, params.pad_to_max_length)

        return {doc[formats.FEATURE_DOCID]: formatter(idx) for idx, doc in zip(chosen, documents)}


def resolve_budget(num_documents, budget, budget_multiplier):
    budget_multiplier = float(budget_multiplier or 0.0)

    if budget_multiplier < 0:
        raise ValueError(ERR_PARAMS_BUDGET_MULTIPLIER.format(budget_multiplier=budget_multiplier))

    if budget_multiplier > 0:
        if budget > 0:
            raise ValueError(
                ERR_PARAMS_BUDGET_CONFLICT.format(
                    budget=budget,
                    budget_multiplier=budget_multiplier,
                )
            )

        budget = math.ceil(budget_multiplier * num_documents)

    elif budget <= 0:
        budget = num_documents

    if budget < num_documents:
        raise ValueError(ERR_PARAMS_BUDGET.format(budget=budget, n=num_documents))

    return budget


def build_identifier_formatter(budget, pad_to_max_length):
    if pad_to_max_length:
        width = len(str(budget - 1))
        return lambda idx: str(idx).zfill(width)

    return lambda idx: str(idx)


def make_random_unique_assignment(num_documents, budget, seed):
    ids = list(range(budget))

    rng = random.Random(int(seed))
    rng.shuffle(ids)
    return ids[:num_documents]
