import hashlib

from redsi.formats import formats
from redsi.plugins import Param

# =============================================================================
# Alphabet constants
# =============================================================================

ALPHABET_BASE32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
ALPHABET_BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
ALPHABET_BASE64_URLSAFE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

ALPHABET_PRESETS = {
    "base32": ALPHABET_BASE32,
    "base64": ALPHABET_BASE64,
    "base64_urlsafe": ALPHABET_BASE64_URLSAFE,
}

ALPHABET_NAMES = tuple(ALPHABET_PRESETS.keys()) + ("custom",)


# =============================================================================
# Plugin metadata
# =============================================================================

PLUGIN_DESCRIPTION = (
    "Assign unique fixed-length hashed document identifiers using a configurable "
    "base-N alphabet (default base64). Identifiers are derived from a stable "
    "per-document key via sha256 and encoded in base=len(alphabet). "
    "Collisions are resolved by rehashing with incrementing salt."
)

PLUGIN_PARAMS = [
    Param(
        "length",
        "int",
        default=4,
        help=(
            "Number of characters in the emitted identifier. "
            "Identifier space size is base^length. "
            "Example (base64): length=4 -> 16.7M codes; length=5 -> 1.07B codes."
        ),
    ),
    Param(
        "alphabet",
        "str",
        default="base64_urlsafe",
        help=(f"Alphabet preset to use. Supported: {ALPHABET_NAMES}. Use 'custom' together with custom_alphabet."),
    ),
    Param(
        "custom_alphabet",
        "str",
        default="",
        help=("Custom alphabet string when alphabet='custom'. Must contain distinct characters only."),
    ),
    Param(
        "salt_start",
        "int",
        default=0,
        help="Initial salt used for collision resolution.",
    ),
]


# =============================================================================
# Errors
# =============================================================================

ERR_LENGTH = "hash_baseN: length must be >= 1 (got {length})"
ERR_ALPHABET_NAME = "hash_baseN: unsupported alphabet preset: {name}"
ERR_CUSTOM_EMPTY = "hash_baseN: custom_alphabet must be provided when alphabet='custom'"
ERR_CUSTOM_DUP = "hash_baseN: custom_alphabet contains duplicate characters"
ERR_CUSTOM_TOO_SMALL = "hash_baseN: custom_alphabet must contain at least 2 characters"


# =============================================================================
# Plugin
# =============================================================================


class HashBaseNIdentifierPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def run(self, documents, params, seed):
        length = int(params.length)
        if length < 1:
            raise ValueError(ERR_LENGTH.format(length=length))

        alphabet = resolve_alphabet(params.alphabet, params.custom_alphabet)
        base = len(alphabet)

        used = set()
        out = {}

        salt_start = int(params.salt_start)

        for doc in documents:
            key = doc[formats.FEATURE_DOCID]

            code = make_unique_code(
                key=key,
                alphabet=alphabet,
                base=base,
                length=length,
                used=used,
                salt_start=salt_start,
            )

            out[key] = code

        return out


# =============================================================================
# Core logic
# =============================================================================


def resolve_alphabet(name: str, custom: str) -> str:
    if name in ALPHABET_PRESETS:
        return ALPHABET_PRESETS[name]

    if name == "custom":
        if not custom:
            raise ValueError(ERR_CUSTOM_EMPTY)
        if len(set(custom)) != len(custom):
            raise ValueError(ERR_CUSTOM_DUP)
        if len(custom) < 2:
            raise ValueError(ERR_CUSTOM_TOO_SMALL)
        return custom

    raise ValueError(ERR_ALPHABET_NAME.format(name=name))


def make_unique_code(key, alphabet, base, length, used, salt_start):
    salt = salt_start

    while True:
        digest = sha256_with_salt(key, salt)

        # Take 64 bits from digest
        value = int.from_bytes(digest[:8], "big", signed=False)

        code = encode_baseN(value, alphabet, base, length)

        if code not in used:
            used.add(code)
            return code

        salt += 1


def sha256_with_salt(key, salt):
    h = hashlib.sha256()
    h.update(key.encode("utf-8"))
    h.update(salt.to_bytes(8, "big", signed=False))
    return h.digest()


def encode_baseN(value, alphabet, base, length):
    chars = []
    for _ in range(length):
        value, r = divmod(value, base)
        chars.append(alphabet[r])
    return "".join(chars)
