"""
Pure-function tests for the ingest watcher's chunk splitter.

Covers: default chunk size, custom size, overlap, word boundaries, edge cases
(empty, single word, exact chunk-size boundary).
"""
from braindb.ingest_watcher import split_chunks, CHUNK_WORDS, CHUNK_OVERLAP


def test_empty_text():
    assert split_chunks("") == []
    assert split_chunks("   \n\t  ") == []


def test_single_word():
    out = split_chunks("hello")
    assert out == ["hello"]


def test_short_text_fits_in_one_chunk():
    text = " ".join(f"w{i}" for i in range(100))   # 100 words, under 600
    out = split_chunks(text)
    assert len(out) == 1
    assert out[0] == text


def test_exact_chunk_size_boundary():
    # Exactly CHUNK_WORDS words — should be one chunk, no empty second chunk
    text = " ".join(f"w{i}" for i in range(CHUNK_WORDS))
    out = split_chunks(text)
    assert len(out) == 1
    assert len(out[0].split()) == CHUNK_WORDS


def test_one_more_than_chunk_size():
    # CHUNK_WORDS + 1 words → should produce 2 chunks (the second one small)
    total = CHUNK_WORDS + 1
    text = " ".join(f"w{i}" for i in range(total))
    out = split_chunks(text)
    assert len(out) == 2
    # First chunk is CHUNK_WORDS long
    assert len(out[0].split()) == CHUNK_WORDS
    # Second chunk starts at step = CHUNK_WORDS - CHUNK_OVERLAP
    step = CHUNK_WORDS - CHUNK_OVERLAP
    assert out[1].split()[0] == f"w{step}"
    # And contains the final word
    assert out[1].split()[-1] == f"w{total - 1}"


def test_overlap_is_as_documented():
    # Verify the configured overlap actually happens between adjacent chunks
    total = CHUNK_WORDS * 2 + 50
    text = " ".join(f"w{i}" for i in range(total))
    out = split_chunks(text)
    # At minimum the first two chunks should share CHUNK_OVERLAP words at the boundary
    first = out[0].split()
    second = out[1].split()
    # Last CHUNK_OVERLAP words of first chunk == first CHUNK_OVERLAP words of second chunk
    assert first[-CHUNK_OVERLAP:] == second[:CHUNK_OVERLAP]


def test_custom_chunk_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(30))
    out = split_chunks(text, chunk_words=10, overlap=2)
    # step = 8, so starts: 0, 8, 16, 24 — last chunk grabs what's left
    starts = [chunk.split()[0] for chunk in out]
    assert starts == ["w0", "w8", "w16", "w24"]


def test_overlap_equal_or_greater_than_chunk_falls_back_to_zero():
    # If someone misconfigures overlap >= chunk_words, the splitter must still
    # make forward progress (no infinite loop, no empty chunks)
    text = " ".join(f"w{i}" for i in range(50))
    out = split_chunks(text, chunk_words=10, overlap=15)   # nonsense config
    # Should degrade to non-overlapping 10-word chunks: 5 chunks
    assert len(out) == 5
    # And every chunk should have content
    assert all(c.strip() for c in out)


def test_no_empty_trailing_chunk():
    """Regression: splitter used to sometimes emit an empty last chunk."""
    text = " ".join(f"w{i}" for i in range(200))
    out = split_chunks(text, chunk_words=50, overlap=10)
    assert all(c.strip() for c in out), "empty chunk found"


def test_words_are_preserved_exactly():
    """Split is whitespace-based — no word should ever be cut mid-word."""
    text = "one two three four five six seven eight nine ten"
    out = split_chunks(text, chunk_words=3, overlap=1)
    # Reconstruct every word referenced in any chunk
    seen = set()
    for chunk in out:
        for word in chunk.split():
            seen.add(word)
    assert seen == {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
