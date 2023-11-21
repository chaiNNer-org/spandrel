def get_max_seq_index(state: dict, key_pattern: str, start: int = 0) -> int:
    """
    Returns the maximum number `i` such that `key_pattern.format(str(i))` is in `state`.

    This is useful for detecting the number of elements in a sequence. Since this
    function returns the highest index, the length of the sequence is simply
    `get_max_seq_index(state, pattern) + 1`. This even correctly accounts for empty
    sequences.

    If no such key is in state, then `start - 1` is returned.

    Example:
        get_max_seq_index(state, "body.{}.weight") -> 5
    """
    i = start
    while True:
        key = key_pattern.format(str(i))
        if key not in state:
            return i - 1
        i += 1
