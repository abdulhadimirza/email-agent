import re
import numpy as np
from fastembed import TextEmbedding

# Initialize the embedder once so we don't load the model on every function call
embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

def extract_relevant_content(text: str, query: str, max_chars: int = 1500) -> str:
    """Extracts the most relevant chunks of text based on the query using Semantic Search."""
    if not text:
        return ""
        
    if not query.strip():
        return text[:max_chars]

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    valid_paragraphs = []
    valid_indices = []
    for i, p in enumerate(paragraphs):
        if is_junk_paragraph(p):
            continue
        valid_paragraphs.append(p)
        valid_indices.append(i)
        
    if not valid_paragraphs:
        return ""
        
    # Compute embeddings
    query_embedding = next(embedder.embed([query]))
    paragraph_embeddings = np.array(list(embedder.embed(valid_paragraphs)))
    
    # Compute cosine similarities
    query_norm = np.linalg.norm(query_embedding)
    paragraph_norms = np.linalg.norm(paragraph_embeddings, axis=1)
    cosine_scores = np.dot(paragraph_embeddings, query_embedding) / (query_norm * paragraph_norms + 1e-10)
    
    scored_paragraphs = []
    for idx, score_val in enumerate(cosine_scores):
        score = float(score_val)
        p = valid_paragraphs[idx]
        original_idx = valid_indices[idx]
        
        # Give a small boost to headings
        if p.startswith('#') and score > 0:
            score *= 1.2
            
        # -original_idx ensures that earlier paragraphs are favored on a tie
        scored_paragraphs.append((score, -original_idx, p))
    
    # Sort by score descending, then by position (ascending original order)
    scored_paragraphs.sort(reverse=True)
    
    selected_paragraphs = []
    current_length = 0
    for score, neg_i, p in scored_paragraphs:
        if score == 0 and current_length > 0:
            continue
            
        if current_length + len(p) <= max_chars:
            selected_paragraphs.append((neg_i, p))
            current_length += len(p) + 2 # +2 for \n\n
        else:
            remaining = max_chars - current_length
            if remaining > 3:
                if '|' in p: # Try to truncate tables nicely by line
                    lines = p.split('\n')
                    table_lines = []
                    table_len = 0
                    for line in lines:
                        if table_len + len(line) + 1 <= remaining:
                            table_lines.append(line)
                            table_len += len(line) + 1
                        else:
                            break
                    if len(table_lines) >= 2:
                        selected_paragraphs.append((neg_i, '\n'.join(table_lines)))
                        current_length += table_len + 2
                else:
                    selected_paragraphs.append((neg_i, p[:remaining - 3] + "..."))
            break
            
    # Sort back to original order
    selected_paragraphs.sort(reverse=True)
    
    return "\n\n".join(p for _, p in selected_paragraphs)
