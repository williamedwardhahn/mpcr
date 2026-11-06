"""Protein sequences from UniProt, turned into lists of numbers."""
import re

import requests

__all__ = ['get_uniprot_data', 'process_strings', 'uniprot_sequences']

_URL = 'https://rest.uniprot.org/uniprotkb/search'


def uniprot_sequences(query, limit=100, timeout=60):
    """Up to `limit` protein sequences (lower case) matching a UniProt query, e.g. 'insulin'."""
    query = query.lstrip('=').strip()
    out = []
    url, params = _URL, {'query': query, 'fields': 'sequence', 'format': 'tsv', 'size': min(limit, 500)}
    while url and len(out) < limit:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        out += [s.lower() for s in r.text.split('\n')[1:] if s]
        nxt = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get('Link', ''))
        url, params = (nxt.group(1), None) if nxt else (None, None)
    return out[:limit]


def get_uniprot_data(kw1, kw2, numxs):
    """Sequences for two keywords, up to numxs each: x, y = get_uniprot_data('kinase', 'collagen', 100).

    (UniProt retired the old uniprot.org/uniprot/?query= address this used; it now uses the REST API.)"""
    return uniprot_sequences(kw1, numxs), uniprot_sequences(kw2, numxs)


def process_strings(c):
    """Each sequence as a list of numbers: a -> 0, b -> 1, ..., anything before 'a' -> 0."""
    return [[max(ord(letter) - 97, 0) for letter in seq] for seq in c]
