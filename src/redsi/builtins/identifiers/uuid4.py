import random
from uuid import UUID

from redsi.formats import formats
from redsi.plugins import Param

PLUGIN_DESCRIPTION = (
    "Generate deterministic UUIDv4 document identifiers using a seeded RNG. "
    "Reproducible given the same seed and document order."
)

PLUGIN_PARAMS = [
    Param(
        "output",
        "str",
        default="hex",
        choices=["hex", "int"],
        help="UUID output format.",
    ),
]


class UUID4IdentifierPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def run(self, documents, params, seed):
        output = params.output.lower()

        rng = random.Random(int(seed))

        out = {}
        for doc in documents:
            key = doc[formats.FEATURE_DOCID]
            uuid = UUID(int=rng.getrandbits(128), version=4)
            out[key] = uuid.hex if output == "hex" else uuid.int

        return out
