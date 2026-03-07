


def split_table_if_needed(table_text, hard_max_length=512):
    """
    将超长表格按“表头+N行”的形式拆分，保留表头，确保每段不超过 hard_max_length。
    """
    lines = table_text.strip().split('\n')
    if len('\n'.join(lines)) <= hard_max_length:
        return [table_text]  # 不用拆分
    
    if len(lines) < 2 or '|' not in lines[0]:
        return [table_text]  # 非标准表格，返回原样

    header = lines[0]
    divider = lines[1]
    rows = lines[2:]
    
    chunks = []
    current_chunk = [header, divider]
    current_len = len(header) + len(divider) + 2  # 2 for newlines
    
    for row in rows:
        row_len = len(row) + 1
        if current_len + row_len > hard_max_length:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [header, divider, row]
            current_len = len(header) + len(divider) + len(row) + 3
        else:
            current_chunk.append(row)
            current_len += row_len

    if len(current_chunk) > 2:
        chunks.append('\n'.join(current_chunk))

    return chunks


def enforce_hard_max_length(chunks, hard_max_length=512):
    """
    强制将 chunk 控制在 hard_max_length 长度以内。
    若超长，则按行拆分。
    """
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


def chunk_text(text, max_chunk_size=300, hard_max_length=512):
    """
    将文本切分成固定大小的块，保持段落和表格的完整性，并控制最大长度。
    """
    lines = text.split('\n')
    
    chunks = []
    current_chunk = []
    current_size = 0
    in_table = False
    table_content = []

    for line in lines:
        if '|' in line and not in_table:
            in_table = True
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            table_content.append(line)
        elif in_table:
            table_content.append(line)
            if not line.strip():
                in_table = False
                table_text = '\n'.join(table_content)
                table_chunks = split_table_if_needed(table_text, hard_max_length)
                chunks.extend(table_chunks)
                table_content = []
        else:
            line_length = len(line)
            if current_size + line_length > max_chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(line)
            current_size += line_length
            if not line.strip() and current_size > max_chunk_size // 2:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0

    if in_table and table_content:
        table_text = '\n'.join(table_content)
        table_chunks = split_table_if_needed(table_text, hard_max_length)
        chunks.extend(table_chunks)

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    # 确保所有 chunk 都小于 hard_max_length
    return enforce_hard_max_length(chunks, hard_max_length=hard_max_length)
