def _parse_table_cols(header_line: str) -> list[str]:
    """Parse column names from a Markdown table header row."""
    return [c.strip() for c in header_line.split('|') if c.strip()]


def _is_divider_row(row: str) -> bool:
    """Return True if row is a Markdown table divider (---|---|...)."""
    return all(ch in '-|: ' for ch in row)


def _parse_row_cells(row: str, n_cols: int) -> list[str]:
    """Parse a Markdown table row into exactly n_cols cell values.

    Preserves empty cells so column alignment is maintained — unlike a naive
    ``row.split('|') if v.strip()`` which discards empty cells and causes
    column shifting for grouped tables (where column 0 is blank on continuation
    rows because it was merged with the row above in the original PDF).

    If the row has fewer cells than n_cols the cells are right-aligned (empty
    cells prepended) on the assumption that leading columns were merged/omitted.
    """
    raw = row.split('|')
    # Strip leading/trailing empty strings introduced by the outer pipes
    if raw and raw[0].strip() == '':
        raw = raw[1:]
    if raw and raw[-1].strip() == '':
        raw = raw[:-1]

    cells = [c.strip() for c in raw]

    if len(cells) < n_cols:
        # Right-align: prepend empty cells so existing values map to the
        # rightmost columns.  This handles rows like "| tire | range | energy |"
        # that should actually be "| (empty) | tire | range | energy |".
        cells = [''] * (n_cols - len(cells)) + cells
    else:
        cells = cells[:n_cols]

    return cells


def _table_to_text(heading: str, header_line: str, data_rows: list[str]) -> list[str]:
    """Convert table data rows to natural language sentences for embedding.

    Handles grouped (multi-row) tables where continuation rows have an empty
    first cell because the value is inherited from the row above (fill-down).
    Without fill-down, continuation rows get garbled sentences (e.g. a tire
    spec appears in the battery-capacity column).

    Each data row becomes one sentence:
      "{heading}：{col1}为{val1}，{col2}为{val2}，..."

    Returns list of sentences (one per non-empty data row).
    """
    cols = _parse_table_cols(header_line)
    if not cols:
        return []
    n_cols = len(cols)

    sentences = []
    prev_cells: list[str] = [''] * n_cols  # fill-down state

    for row in data_rows:
        if not row.strip() or _is_divider_row(row):
            continue

        cells = _parse_row_cells(row, n_cols)

        # Fill-down: inherit non-empty values from the previous row for any
        # cell that is blank in the current row.
        for i, cell in enumerate(cells):
            if not cell and prev_cells[i]:
                cells[i] = prev_cells[i]

        prev_cells = cells[:]

        pairs = [
            f"{cols[i]}为{cells[i]}"
            for i in range(n_cols)
            if cells[i]
        ]
        if not pairs:
            continue

        prefix = f"{heading}：" if heading else ""
        sentences.append(prefix + "，".join(pairs))

    return sentences


def _group_sentences(sentences: list[str], max_size: int) -> list[str]:
    """Group sentences into chunks not exceeding max_size characters."""
    chunks = []
    current: list[str] = []
    current_len = 0

    for s in sentences:
        s_len = len(s) + 1
        if current_len + s_len > max_size and current:
            chunks.append('\n'.join(current))
            current = []
            current_len = 0
        current.append(s)
        current_len += s_len

    if current:
        chunks.append('\n'.join(current))

    return chunks


def enforce_hard_max_length(chunks: list[str], hard_max_length: int = 1200) -> list[str]:
    """Force all chunks to be within hard_max_length by splitting on line boundaries."""
    final_chunks = []

    for chunk in chunks:
        if len(chunk) <= hard_max_length:
            final_chunks.append(chunk)
        else:
            lines = chunk.split('\n')
            current = []
            current_len = 0
            for line in lines:
                line_len = len(line) + 1
                if current_len + line_len > hard_max_length and current:
                    final_chunks.append('\n'.join(current))
                    current = []
                    current_len = 0
                current.append(line)
                current_len += line_len
            if current:
                final_chunks.append('\n'.join(current))

    return final_chunks


def chunk_text(text: str, max_chunk_size: int = 600, hard_max_length: int = 1200) -> list[tuple[str, str]]:
    """Split text into (small_chunk, parent_chunk) tuples.

    small_chunk  — used for embedding:
                   • plain-text chunks: same as before
                   • table chunks: natural language sentences (table-to-text) so
                     embedding models can find rows by semantic similarity instead
                     of matching raw pipe characters
    parent_chunk — returned to the LLM:
                   • always the original content (heading + full table or plain text)
                   so the LLM sees the real data, not the converted sentences

    Table-to-text format per row:
        "{heading}：{col1}为{val1}，{col2}为{val2}，..."

    Args:
        max_chunk_size:  Soft target length for plain-text and sentence groups.
        hard_max_length: Hard ceiling applied to small_chunks after conversion.
    """
    lines = text.split('\n')

    result: list[tuple[str, str]] = []
    current_chunk: list[str] = []
    current_size = 0
    in_table = False
    table_content: list[str] = []
    last_heading = ""

    def flush_text():
        nonlocal current_chunk, current_size
        if current_chunk:
            txt = '\n'.join(current_chunk)
            result.append((txt, txt))
            current_chunk = []
            current_size = 0

    def flush_table():
        nonlocal table_content
        if not table_content:
            return

        table_text = '\n'.join(table_content)

        # parent = heading + full original table (capped at 4000 chars for Milvus)
        parent_text = ((last_heading + '\n' + table_text) if last_heading else table_text)[:4000]

        # Parse table structure for table-to-text conversion
        t_lines = [l for l in table_text.strip().split('\n') if l.strip()]
        if len(t_lines) >= 2 and '|' in t_lines[0]:
            header_line = t_lines[0]
            data_rows = [l for l in t_lines[1:] if not _is_divider_row(l)]

            sentences = _table_to_text(last_heading, header_line, data_rows)

            if sentences:
                # Group sentences into small chunks by max_chunk_size
                for group in _group_sentences(sentences, max_chunk_size):
                    result.append((group, parent_text))
            else:
                # No sentences generated (e.g. empty table), fall back to original
                small = (last_heading + '\n' + table_text) if last_heading else table_text
                result.append((small[:hard_max_length], parent_text))
        else:
            # Non-standard table, keep original
            small = (last_heading + '\n' + table_text) if last_heading else table_text
            result.append((small[:hard_max_length], parent_text))

        table_content = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            last_heading = stripped

        if '|' in line and not in_table:
            in_table = True
            flush_text()
            table_content.append(line)
        elif in_table:
            table_content.append(line)
            if not line.strip():
                in_table = False
                flush_table()
        else:
            line_length = len(line)
            if current_size + line_length > max_chunk_size and current_chunk:
                flush_text()
            current_chunk.append(line)
            current_size += line_length
            if not line.strip() and current_size > max_chunk_size // 2:
                flush_text()

    if in_table and table_content:
        flush_table()
    flush_text()

    # Enforce hard ceiling on small chunks
    final: list[tuple[str, str]] = []
    for small, parent in result:
        if len(small) <= hard_max_length:
            final.append((small, parent))
        else:
            for sc in enforce_hard_max_length([small], hard_max_length):
                final.append((sc, parent))

    return final
