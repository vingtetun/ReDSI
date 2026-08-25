# Natural Questions (NQ)

**Natural Questions (NQ)** is a large-scale question answering dataset built from real anonymized Google search queries, with answers derived from Wikipedia pages.

Each query includes human annotations containing:

- A **long answer**
- A **short answer**
- A **Yes/No answer**
- Or **no answer**

---

## Dataset Size

| Split       | Examples | Annotation Type |
|------------|----------|-----------------|
| Train      | 307,373  | Single annotation |
| Validation | 7,830    | 5-way annotations |
| Test       | Not publicly released | — |

---

## Dataset Formats

The dataset exists in two common formats:

- **Full format** (original Google release)
- **Simplified format** (lighter processing, flattened structure)

---

## Full Format Example

```json
{
  "example_id": 6915606477668963399,
  "question_text": "what do the 3 dots mean in math",
  "question_tokens": [
    "what", "do", "the", "3", "dots", "mean", "in", "math"
  ],
  "document_url": "https://en.wikipedia.org//w/index.php?title=Therefore_sign&oldid=815234923",
  "document_title": "Therefore sign",
  "document_html": "<!DOCTYPE html> ...",
  "document_tokens": [
    {
      "start_byte": 92,
      "end_byte": 101,
      "html_token": false,
      "token": "Therefore"
    }
  ],
  "long_answer_candidates": [
    {
      "start_token": 14,
      "end_token": 808,
      "top_level": true
    }
  ],
  "annotations": [
    {
      "annotation_id": 13591449469826568799,
      "long_answer": {
        "candidate_index": 92,
        "start_token": 808,
        "end_token": 925
      },
      "short_answers": [
        {
          "start_token": 816,
          "end_token": 837
        }
      ],
      "yes_no_answer": "NONE"
    }
  ]
}
```

---

## Simplified Format Example

```json
{
  "example_id": 5655493461695504401,
  "question_text": "which is the most common use of opt-in e-mail marketing",
  "document_url": "https://en.wikipedia.org//w/index.php?title=Email_marketing&oldid=814071202",
  "document_text": "Email marketing - Wikipedia <H1> Email marketing </H1> Jump to ...",
  "long_answer_candidates": [
    {
      "start_token": 14,
      "end_token": 170,
      "top_level": true
    }
  ],
  "annotations": [
    {
      "annotation_id": 593165450220027640,
      "yes_no_answer": "NONE",
      "long_answer": {
        "candidate_index": 54,
        "start_token": 1952,
        "end_token": 2019
      },
      "short_answers": [
        {
          "start_token": 1960,
          "end_token": 1969
        }
      ]
    }
  ]
}
```

---

## Document Text Styles

We support the following document text formatting styles:

| Style | Description |
|-------|-------------|
| `simplified` | Direct Natural Questions simplified text join, preserving the simplified HTML-like tags. |
| `simplenorm` | Simple normalization of `simplified`: strips HTML-like tags and normalizes whitespace. |
| `title-abstract-content` | Uses the first `<H1>` from `document_text`, the first `<P>` as abstract, then the remaining stripped content. |
| `ncinorm` | NCI-compatible title asymmetry: train uses the first `<H1>` from `document_text`; validation/eval uses `document_title`. Empty titles are kept empty, matching the NCI recipe. Abstract/content extraction is the same as `title-abstract-content`. |

---

## Document Deduplication Strategies

Natural Questions contains multiple queries pointing to the same Wikipedia document.
Different works deduplicate documents differently.

We support the following deduplication keys:

| Strategy | Description |
|----------|-------------|
| `text[:4000]` | First 4000 UTF-8 characters of the document text |
| `lower(text[:4000])` | Lowercased first 4000 UTF-8 characters of the document text. This corresponds to the deduplication strategy used in the original **DSI** paper. |
| `url` | Exact URL string match |
| `title` | Wikipedia title extracted from the `title` query parameter |
| `lower(title)` | Lowercased title |
| `h1` | Content of the first HTML `<h1>` tag |
| `lower(title)` | Lowercased h1 |
| `nci-h1` | `<h1>` normalized using NCI-style procedure (BERT tokenization + decode) |
| `ncititle` | Split-dependent NCI normalization computed before text formatting:<br>• Train → original `<h1>`<br>• Eval → original `document_title`<br>Empty titles are kept empty, so missing-title documents collapse into the same NCI dedup bucket. Both normalized as in `nci-h1` |
| `page-id` | Canonical Wikipedia `pageid` recovered from `oldid` (revision id). Robust to title changes |

---

For formatted text styles that remove HTML structure, such as `htmlnorm`, the
NCI key is carried in the simplified cache as `nci_dedup_title`. If an older
cache does not contain that field, rebuild the style-specific Natural Questions
cache before using `dedup_method: ncititle`.

### Example

```
https://en.wikipedia.org/w/index.php?title=Email_marketing&oldid=814071202
```

- `title` → `Email marketing`
- `page-id` → recovered from the revision identifier `oldid`
- `page-id` is stable to title changes

---

## Unique Document Counts per Deduplication Strategy

| Split | # docs | text4k |lower(text4k) (DSI) |  url | title | lower(title) | h1 | lower(h1) | nci-h1 | ncititle | pageid |
|-------|-----------|-----|-------|--------------|----|-----------|-----|-----|--------|
| Train | 307,373 |  198,918 | 198,202 | 226,180 | 108,071 | 108,015 | 108,998 | 108,036 | 108,026 | 108,026 | 107,593 |
| Validation | 7,830 | 7,280 | 7,271 | 7,369 | 6,930 | 6,930 | 6,935 | 6,930 | 6,930 | 6,930 | 6,926 |
| **Union** | 315,203 | 202,970| 202,221 | 231,695 | 109,712 | 109,654 | 110,660 |  109,676 | 109,666 | 109,739 | 109,229 |

---

## Why This Matters

Small deduplication differences:

- Change the number of documents
- Change the document identifier space
- Affect invalid generation rates in generative retrieval
- Influence reported Hits@k and MRR

These differences are often under-documented in the literature but may affect reproducibility.
