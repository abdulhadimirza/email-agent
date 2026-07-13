import re
from typing import Any, cast
import numpy as np
import tiktoken
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder


# Initialize the embedder and reranker once so we don't load the models on every function call
embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
try:
    reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
except Exception as e:
    print(f"Warning: Failed to load cross-encoder: {e}. Falling back to Bi-Encoder only.")
    reranker = None

def get_token_count(text: str, model: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text, disallowed_special=()))
    except Exception:
        # Fallback if tiktoken is not available
        return len(text) // 4

def chunk_text_by_tokens(text: str, chunk_tokens: int = 200, overlap_tokens: int = 40) -> list[tuple[int, str]]:
    """Splits text into chunks of specified token sizes with overlap. Returns list of (index, chunk)."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text, disallowed_special=())
        chunks = []
        i = 0
        idx = 0
        while i < len(tokens):
            chunk = encoding.decode(tokens[i:i + chunk_tokens])
            if chunk.strip():
                chunks.append((idx, chunk.strip()))
                idx += 1
            i += chunk_tokens - overlap_tokens
        return chunks
    except Exception:
        # Fallback if tiktoken is not available
        words = text.split()
        chunks = []
        i = 0
        idx = 0
        while i < len(words):
            chunk = " ".join(words[i:i + (chunk_tokens)])
            if chunk.strip():
                chunks.append((idx, chunk.strip()))
                idx += 1
            i += chunk_tokens - overlap_tokens
        return chunks

def clean_markdown(text: str) -> str:
    """Strip navigation bars, footer link definitions, and tag/post lists from scraped markdown."""
    # 1. Remove link reference definitions at the end (e.g. [1]: http://...)
    text = re.sub(r'^\[\d+\]:\s+\S+', '', text, flags=re.MULTILINE)
    
    # 2. Filter lines that are navigation blocks or isolated lists of links
    cleaned_lines = []
    lines = text.split('\n')
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        
        # Skip pure list items containing only a markdown link (e.g. "* [ All posts ][6]")
        if re.match(r'^[\*\-\+\d\.]?\s*\[[^\]]+\]\s*(?:\[\d+\]|\([^\)]+\))\s*$', stripped):
            continue
            
        # Skip lines that are mostly links and short (heuristics for navigation bars/tag clouds)
        links_count = len(re.findall(r'\[[^\]]+\]\s*(?:\[\d+\]|\([^\)]+\))', stripped))
        if links_count > 0 and links_count * 15 >= len(stripped):
            continue
            
        cleaned_lines.append(line)
        
    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()

def is_junk_paragraph(p: str) -> bool:
    """Check if a paragraph is likely junk (navigation, tracking links, redirects)."""
    p_lower = p.lower()
    # Check for URL query parameter / redirect noise
    if "%2f" in p_lower or "%3a" in p_lower or "redirect=" in p_lower or "source=post_page" in p_lower:
        return True
    # Check for extremely long tokens (e.g. tracking parameters or base64 data)
    words = p.split()
    if words:
        max_len = max(len(w) for w in words)
        if max_len > 55:
            return True
    # Check if the paragraph consists only of references, brackets, digits, list markers
    stripped = p.strip()
    if re.match(r'^[\]\[\s\d\-\+\*]*$', stripped):
        return True
    return False

def extract_relevant_content(text: str, query: str, max_tokens: int = 1500) -> str:
    """Extracts the most relevant chunks of text based on the query using Semantic Search and Cross-Encoder."""
    if not text:
        return ""
        
    if not query.strip():
        return text[:max_tokens * 4]

    # Tokenize the entire text once
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text, disallowed_special=())
    except Exception:
        # Fallback if tiktoken is not available
        encoding = None
        tokens = text.split()

    chunk_tokens = 200
    overlap_tokens = 40
    
    # 1. Generate chunks as (start_idx, end_idx) token ranges
    ranges = []
    i = 0
    while i < len(tokens):
        end = min(i + chunk_tokens, len(tokens))
        ranges.append((i, end))
        if end == len(tokens):
            break
        i += chunk_tokens - overlap_tokens

    valid_ranges = []
    valid_chunks = []
    for start, end in ranges:
        if encoding:
            chunk = encoding.decode(cast(list[int], tokens)[start:end]).strip()
        else:
            chunk = " ".join(cast(list[str], tokens)[start:end]).strip()
            
        if not chunk or is_junk_paragraph(chunk):
            continue
            
        valid_chunks.append(chunk)
        valid_ranges.append((start, end))

    if not valid_chunks:
        return ""

    # Step 1: Fast initial retrieval using Bi-Encoder
    query_embedding = next(iter(embedder.embed([query])))
    chunk_embeddings = np.array(list(embedder.embed(valid_chunks)))
    
    query_norm = np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
    cosine_scores = np.dot(chunk_embeddings, query_embedding) / (query_norm * chunk_norms + 1e-10)
    
    # Get top 20 candidates
    top_n = min(20, len(valid_chunks))
    top_indices = np.argsort(cosine_scores)[-top_n:][::-1]
    
    candidates = [valid_chunks[i] for i in top_indices]
    candidate_ranges = [valid_ranges[i] for i in top_indices]
    candidate_cosine_scores = [cosine_scores[i] for i in top_indices]

    # Step 2: Accurate Re-ranking using Cross-Encoder
    if reranker is not None:
        try:
            cross_scores = list(reranker.rerank(query, candidates))
        except Exception as e:
            print(f"Warning: Reranking failed ({e}), falling back to Bi-Encoder scores.")
            cross_scores = candidate_cosine_scores
    else:
        cross_scores = candidate_cosine_scores

    # Pair scores with ranges and sort by score descending
    scored_candidates = []
    for i, score_val in enumerate(cast(list[Any], cross_scores)):
        try:
            score = float(score_val)
        except TypeError:
            score = float(score_val[0])
            
        chunk = candidates[i]
        start, end = candidate_ranges[i]
        
        # Give a small boost to headings
        if chunk.startswith('#') and score > 0:
            score *= 1.2
            
        scored_candidates.append((score, start, end))
    
    scored_candidates.sort(reverse=True)
    
    # Step 3: Dynamic Thresholding and Token Budgeting
    max_score = scored_candidates[0][0]
    threshold = max_score * 0.40 if max_score > 0 else 0
    
    selected_ranges = []
    current_tokens = 0
    
    for score, start, end in scored_candidates:
        if score < threshold and current_tokens > 0:
            break
            
        chunk_len = end - start
        if current_tokens + chunk_len <= max_tokens:
            selected_ranges.append((start, end))
            current_tokens += chunk_len
        else:
            break

    if not selected_ranges:
        return ""

    # Sort selected ranges by start position for interval merging
    selected_ranges.sort()
    
    # 2. Merge overlapping or adjacent ranges
    merged_ranges = []
    for start, end in selected_ranges:
        if not merged_ranges:
            merged_ranges.append([start, end])
        else:
            prev_start, prev_end = merged_ranges[-1]
            if start <= prev_end:  # Overlapping or contiguous (start <= prev_end)
                merged_ranges[-1][1] = max(prev_end, end)
            else:
                merged_ranges.append([start, end])

    # 3. Decode only the merged ranges
    extracted_parts = []
    for start, end in merged_ranges:
        if encoding:
            part = encoding.decode(cast(list[int], tokens)[start:end]).strip()
        else:
            part = " ".join(cast(list[str], tokens)[start:end]).strip()
        if part:
            extracted_parts.append(part)
            
    return "\n\n...\n\n".join(extracted_parts)
