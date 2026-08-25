import math
import random
from dataclasses import dataclass

from tqdm.auto import tqdm

from redsi.formats import formats
from redsi.plugins import Param

PLUGIN_DESCRIPTION = (
    "Assign semantically structured decimal identifiers by recursively applying "
    "scikit-learn k-means over BERT document embeddings, following the DSI "
    "semantic docid scheme."
)

TEXT_FIELDS = (formats.FEATURE_TITLE, formats.FEATURE_TEXT)

PLUGIN_PARAMS = [
    Param(
        "model_name",
        "str",
        default="google/bert_uncased_L-8_H-512_A-8",
        help="Transformer encoder used to embed documents.",
    ),
    Param(
        "branching_factor",
        "int",
        default=10,
        help="Number of clusters created at each internal trie node.",
    ),
    Param(
        "max_leaf_size",
        "int",
        default=100,
        help="Clusters with this many documents or fewer receive arbitrary local suffixes.",
    ),
    Param(
        "batch_size",
        "int",
        default=32,
        help="Batch size used when computing transformer embeddings.",
    ),
    Param(
        "max_length",
        "int",
        default=512,
        help="Maximum token length used when computing transformer embeddings.",
    ),
    Param(
        "pooling",
        "str",
        default="cls",
        choices=["mean", "cls"],
        help='Embedding pooling strategy: "mean" or "cls".',
    ),
    Param(
        "normalize_embeddings",
        "bool",
        default=True,
        help="Whether to L2-normalize document embeddings before clustering.",
    ),
    Param(
        "kmeans_backend",
        "str",
        default="sklearn",
        choices=["sklearn", "builtin"],
        help='K-means backend: "sklearn" matches the DSI paper; "builtin" avoids the scikit-learn dependency.',
    ),
    Param(
        "kmeans_max_iter",
        "int",
        default=100,
        help="Maximum Lloyd iterations per k-means run when using the built-in fallback k-means.",
    ),
    Param(
        "kmeans_n_init",
        "int",
        default=4,
        help="Number of seeded k-means initializations when using the built-in fallback k-means.",
    ),
]

ERR_BRANCHING_FACTOR = "semantic: branching_factor must be >= 2 (got {value})"
ERR_MAX_LEAF_SIZE = "semantic: max_leaf_size must be >= 1 (got {value})"
ERR_BATCH_SIZE = "semantic: batch_size must be >= 1 (got {value})"
ERR_MAX_LENGTH = "semantic: max_length must be >= 1 (got {value})"
ERR_KMEANS_MAX_ITER = "semantic: kmeans_max_iter must be >= 1 (got {value})"
ERR_KMEANS_N_INIT = "semantic: kmeans_n_init must be >= 1 (got {value})"
ERR_SKLEARN_MISSING = "semantic: scikit-learn is required when kmeans_backend='sklearn'"
ERR_MISSING_TEXT = "semantic: no usable text in fields {fields} for document key={key}"
ERR_KMEANS_BACKEND = "semantic: unsupported kmeans_backend value: {value!r}"


class SemanticIdentifierPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def run(self, documents, params, seed):
        documents = list(documents)
        if not documents:
            return {}

        settings = SemanticIdentifierSettings.from_params(params=params, seed=seed)
        embeddings = build_document_embeddings(documents, settings)
        semantic_ids = generate_semantic_ids(embeddings, settings)

        return map_documents_to_semantic_ids(documents, semantic_ids)


@dataclass(frozen=True)
class SemanticIdentifierSettings:
    seed: int
    model_name: str
    branching_factor: int
    max_leaf_size: int
    batch_size: int
    max_length: int
    pooling: str
    normalize_embeddings: bool
    kmeans_backend: str
    kmeans_max_iter: int
    kmeans_n_init: int

    @classmethod
    def from_params(cls, params, seed):
        settings = cls(
            seed=int(seed),
            model_name=params.model_name,
            branching_factor=int(params.branching_factor),
            max_leaf_size=int(params.max_leaf_size),
            batch_size=int(params.batch_size),
            max_length=int(params.max_length),
            pooling=params.pooling,
            normalize_embeddings=bool(params.normalize_embeddings),
            kmeans_backend=params.kmeans_backend,
            kmeans_max_iter=int(params.kmeans_max_iter),
            kmeans_n_init=int(params.kmeans_n_init),
        )
        settings.validate()
        return settings

    def validate(self):
        if self.branching_factor < 2:
            raise ValueError(ERR_BRANCHING_FACTOR.format(value=self.branching_factor))
        if self.max_leaf_size < 1:
            raise ValueError(ERR_MAX_LEAF_SIZE.format(value=self.max_leaf_size))
        if self.batch_size < 1:
            raise ValueError(ERR_BATCH_SIZE.format(value=self.batch_size))
        if self.max_length < 1:
            raise ValueError(ERR_MAX_LENGTH.format(value=self.max_length))
        if self.kmeans_max_iter < 1:
            raise ValueError(ERR_KMEANS_MAX_ITER.format(value=self.kmeans_max_iter))
        if self.kmeans_n_init < 1:
            raise ValueError(ERR_KMEANS_N_INIT.format(value=self.kmeans_n_init))
        if self.kmeans_backend not in {"sklearn", "builtin"}:
            raise ValueError(ERR_KMEANS_BACKEND.format(value=self.kmeans_backend))


def build_document_embeddings(documents, settings):
    document_texts = build_document_texts(documents)
    embeddings = embed_texts(document_texts, settings)

    if settings.normalize_embeddings:
        return normalize_rows(embeddings)

    return embeddings


def map_documents_to_semantic_ids(documents, semantic_ids):
    return {doc[formats.FEATURE_DOCID]: semantic_id for doc, semantic_id in zip(documents, semantic_ids)}


def build_document_texts(documents):
    return [build_document_text(doc) for doc in documents]


def build_document_text(doc):
    parts = [text for text in (clean_text(doc.get(field)) for field in TEXT_FIELDS) if text]

    if not parts:
        raise ValueError(ERR_MISSING_TEXT.format(fields=list(TEXT_FIELDS), key=doc[formats.FEATURE_DOCID]))

    return " ".join(parts)


def clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


def embed_texts(texts, settings):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = resolve_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, use_fast=True)
    model = AutoModel.from_pretrained(settings.model_name)
    model.to(device)
    model.eval()

    embeddings = []
    total_batches = math.ceil(len(texts) / settings.batch_size)
    with torch.no_grad():
        for batch_texts in tqdm(
            batched(texts, settings.batch_size),
            total=total_batches,
            desc="Embed semantic documents",
            unit="batch",
        ):
            embeddings.extend(embed_text_batch(batch_texts, tokenizer, model, settings, device))

    return embeddings


def embed_text_batch(texts, tokenizer, model, settings, device):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=settings.max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    outputs = model(**encoded)
    pooled = pool_hidden_state(outputs.last_hidden_state, encoded["attention_mask"], settings.pooling)
    return pooled.cpu().tolist()


def batched(items, batch_size):
    for offset in range(0, len(items), batch_size):
        yield items[offset : offset + batch_size]


def resolve_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pool_hidden_state(hidden_state, attention_mask, pooling):
    if pooling == "cls":
        return hidden_state[:, 0]

    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def normalize_rows(rows):
    out = []
    for row in rows:
        norm = math.sqrt(sum(value * value for value in row))
        if norm == 0:
            out.append(list(row))
        else:
            out.append([value / norm for value in row])
    return out


def generate_semantic_ids(embeddings, settings):
    semantic_ids = [None] * len(embeddings)
    indexed_embeddings = list(enumerate(embeddings))

    with tqdm(total=1, desc="Build semantic id tree", unit="node") as progress:
        assign_semantic_ids_to_node(
            indexed_embeddings=indexed_embeddings,
            semantic_ids=semantic_ids,
            prefix="",
            seed=settings.seed,
            settings=settings,
            progress=progress,
        )

    return semantic_ids


def assign_semantic_ids_to_node(indexed_embeddings, semantic_ids, prefix, seed, settings, progress=None):
    if is_leaf_node(indexed_embeddings, settings):
        assign_leaf_ids(indexed_embeddings, semantic_ids, prefix)
        update_progress(progress, size=len(indexed_embeddings), action="leaf")
        return

    set_progress_status(progress, size=len(indexed_embeddings), action="kmeans")
    clusters = cluster_node_embeddings(indexed_embeddings, seed, settings)
    child_clusters = [(cluster_id, cluster) for cluster_id, cluster in enumerate(clusters) if cluster]
    add_progress_total(progress, len(child_clusters))
    update_progress(progress, size=len(indexed_embeddings), action="split")

    for cluster_id, cluster_items in child_clusters:
        assign_semantic_ids_to_node(
            indexed_embeddings=cluster_items,
            semantic_ids=semantic_ids,
            prefix=f"{prefix}{cluster_id}",
            seed=next_seed(seed, cluster_id),
            settings=settings,
            progress=progress,
        )


def set_progress_status(progress, size, action):
    if progress is not None:
        progress.set_postfix_str(f"{action} n={size}", refresh=True)


def add_progress_total(progress, count):
    if progress is not None and count:
        progress.total = (progress.total or 0) + count


def update_progress(progress, size, action):
    if progress is not None:
        progress.set_postfix_str(f"{action} n={size}", refresh=False)
        progress.update(1)


def is_leaf_node(indexed_embeddings, settings):
    return len(indexed_embeddings) <= settings.max_leaf_size


def assign_leaf_ids(indexed_embeddings, semantic_ids, prefix):
    for suffix, (original_index, _) in enumerate(indexed_embeddings):
        semantic_ids[original_index] = f"{prefix}{suffix}"


def cluster_node_embeddings(indexed_embeddings, seed, settings):
    k = min(settings.branching_factor, len(indexed_embeddings))
    vectors = [embedding for _, embedding in indexed_embeddings]
    labels = kmeans_labels(vectors=vectors, k=k, seed=seed, settings=settings)
    clusters = group_items_by_cluster(indexed_embeddings, labels, k)

    if clustering_makes_progress(clusters, len(indexed_embeddings)):
        return clusters

    return make_balanced_clusters(indexed_embeddings, k)


def group_items_by_cluster(items, labels, num_clusters):
    clusters = [[] for _ in range(num_clusters)]
    for item, label in zip(items, labels):
        clusters[label].append(item)
    return clusters


def clustering_makes_progress(clusters, parent_size):
    non_empty = [cluster for cluster in clusters if cluster]
    return len(non_empty) > 1 and all(len(cluster) < parent_size for cluster in non_empty)


def make_balanced_clusters(indexed_embeddings, k):
    clusters = [[] for _ in range(k)]
    for index, item in enumerate(indexed_embeddings):
        clusters[index % k].append(item)
    return clusters


def kmeans_labels(vectors, k, seed, settings):
    if settings.kmeans_backend == "sklearn":
        return sklearn_kmeans_labels(vectors, k, seed)

    if settings.kmeans_backend == "builtin":
        return builtin_kmeans_labels(vectors, k, seed, settings)

    raise ValueError(ERR_KMEANS_BACKEND.format(value=settings.kmeans_backend))


def sklearn_kmeans_labels(vectors, k, seed):
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError(ERR_SKLEARN_MISSING) from exc

    return KMeans(n_clusters=k, random_state=seed).fit_predict(vectors).tolist()


def builtin_kmeans_labels(vectors, k, seed, settings):
    best_labels = None
    best_inertia = None

    for init_index in range(settings.kmeans_n_init):
        init_seed = next_seed(seed, init_index)
        labels, inertia = run_kmeans_once(vectors, k, init_seed, settings.kmeans_max_iter)
        if best_inertia is None or inertia < best_inertia:
            best_labels = labels
            best_inertia = inertia

    return best_labels


def run_kmeans_once(vectors, k, seed, max_iter):
    rng = random.Random(seed)
    centroids = initialize_centroids(vectors, k, rng)
    labels = [0] * len(vectors)

    for _ in range(max_iter):
        next_labels = [nearest_centroid(vector, centroids) for vector in vectors]
        if next_labels == labels:
            break

        labels = next_labels
        centroids = recompute_centroids(vectors, labels, centroids, rng)

    inertia = sum(squared_distance(vector, centroids[label]) for vector, label in zip(vectors, labels))
    return labels, inertia


def initialize_centroids(vectors, k, rng):
    first_index = rng.randrange(len(vectors))
    centroid_indices = [first_index]

    while len(centroid_indices) < k:
        distances = distance_to_nearest_selected_centroid(vectors, centroid_indices)
        centroid_indices.append(select_next_centroid_index(distances, centroid_indices, rng))

    return [list(vectors[index]) for index in centroid_indices]


def distance_to_nearest_selected_centroid(vectors, centroid_indices):
    return [
        min(squared_distance(vector, vectors[centroid_index]) for centroid_index in centroid_indices)
        for vector in vectors
    ]


def select_next_centroid_index(distances, centroid_indices, rng):
    total_distance = sum(distances)
    available_indices = [index for index in range(len(distances)) if index not in centroid_indices]

    if total_distance == 0:
        return available_indices[0]

    threshold = rng.random() * total_distance
    cumulative = 0.0
    for index, distance in enumerate(distances):
        cumulative += distance
        if cumulative >= threshold and index in available_indices:
            return index

    return max(available_indices, key=lambda index: distances[index])


def nearest_centroid(vector, centroids):
    best_index = 0
    best_distance = squared_distance(vector, centroids[0])

    for index in range(1, len(centroids)):
        distance = squared_distance(vector, centroids[index])
        if distance < best_distance:
            best_index = index
            best_distance = distance

    return best_index


def recompute_centroids(vectors, labels, centroids, rng):
    sums = [[0.0] * len(vectors[0]) for _ in centroids]
    counts = [0] * len(centroids)

    for vector, label in zip(vectors, labels):
        counts[label] += 1
        add_vector_to_sum(sums[label], vector)

    return [
        list(vectors[rng.randrange(len(vectors))]) if count == 0 else average_vector(vector_sum, count)
        for vector_sum, count in zip(sums, counts)
    ]


def add_vector_to_sum(vector_sum, vector):
    for dim_index, value in enumerate(vector):
        vector_sum[dim_index] += value


def average_vector(vector_sum, count):
    return [value / count for value in vector_sum]


def squared_distance(left, right):
    return sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right))


def next_seed(seed, value):
    return (int(seed) * 1_000_003 + int(value) + 97) % (2**32)
