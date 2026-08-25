from redsi.formats import formats

PLUGIN_DESCRIPTION = "Identity mapping: keep the existing document key as the identifier (example plugin)."

PLUGIN_PARAMS = []


class IdentityIdentifierPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def run(self, documents, params, seed):
        return {doc[formats.FEATURE_DOCID]: doc[formats.FEATURE_DOCID] for doc in documents}
