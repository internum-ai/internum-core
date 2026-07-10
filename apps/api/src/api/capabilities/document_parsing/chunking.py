"""Pure logic for splitting a large tabular document into row-range chunks.

No I/O, no model calls, and no imports from the OpenRouter client or the
parsing service — this module only transforms markdown/schema/data structures.
"""

from dataclasses import dataclass
from typing import Any

from api.capabilities.document_parsing.models import (
    ChunkOutcome,
    ChunkPlan,
    DocumentChunk,
    RowArrayLocation,
)


@dataclass(frozen=True)
class ChunkingConfig:
    row_threshold: int
    rows_per_chunk: int


@dataclass(frozen=True)
class _TableBlock:
    start_line: int
    end_line: int
    header_line: str
    delimiter_line: str
    data_lines: list[str]


def detect_table_row_count(markdown: str) -> int:
    block = _find_dominant_table(markdown)
    if block is None:
        return 0
    return len(block.data_lines)


def locate_row_array(schema: dict[str, Any]) -> RowArrayLocation | None:
    if not isinstance(schema, dict):
        return None
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None

    matches: list[RowArrayLocation] = []
    for name, subschema in properties.items():
        if not isinstance(subschema, dict):
            continue
        if subschema.get("type") != "array":
            continue
        items = subschema.get("items")
        if not isinstance(items, dict):
            continue
        if items.get("type") == "object" or "properties" in items:
            matches.append(RowArrayLocation(property_name=name, item_schema=items))

    if len(matches) != 1:
        return None
    return matches[0]


def build_chunk_plan(
    markdown: str,
    schema: dict[str, Any],
    config: ChunkingConfig,
) -> ChunkPlan | None:
    total_rows = detect_table_row_count(markdown)
    if total_rows < config.row_threshold:
        return None

    array_location = locate_row_array(schema)
    if array_location is None:
        return None

    block = _find_dominant_table(markdown)
    if block is None:
        return None

    lines = markdown.splitlines()
    preamble = "\n".join(lines[: block.start_line]).strip()
    footer = "\n".join(lines[block.end_line + 1 :]).strip()

    chunks: list[DocumentChunk] = []
    for index, start in enumerate(range(0, len(block.data_lines), config.rows_per_chunk)):
        group = block.data_lines[start : start + config.rows_per_chunk]
        chunk_markdown = "\n".join([block.header_line, block.delimiter_line, *group])
        chunks.append(
            DocumentChunk(
                index=index,
                start_row=start,
                end_row=start + len(group),
                markdown=chunk_markdown,
            )
        )

    chunk_schema = {
        "type": "object",
        "properties": {
            array_location.property_name: {
                "type": "array",
                "items": array_location.item_schema,
            }
        },
        "required": [array_location.property_name],
        "additionalProperties": False,
    }

    other_properties = {
        name: subschema
        for name, subschema in schema.get("properties", {}).items()
        if name != array_location.property_name
    }
    summary_schema: dict[str, Any] | None = None
    if other_properties:
        summary_schema = {
            "type": "object",
            "properties": other_properties,
            "required": list(other_properties.keys()),
            "additionalProperties": False,
        }

    leading_data_lines = "\n".join(block.data_lines[: config.rows_per_chunk])
    summary_sections = [
        section
        for section in (
            preamble,
            block.header_line,
            block.delimiter_line,
            leading_data_lines,
            footer,
        )
        if section
    ]
    summary_markdown = "\n".join(summary_sections)

    return ChunkPlan(
        array_location=array_location,
        chunk_schema=chunk_schema,
        summary_schema=summary_schema,
        summary_markdown=summary_markdown,
        chunks=chunks,
        total_rows=total_rows,
    )


def merge_chunk_rows(outcomes: list[ChunkOutcome]) -> tuple[list[Any], list[int]]:
    ordered = sorted(outcomes, key=lambda outcome: outcome.index)
    merged: list[Any] = []
    failed: list[int] = []
    for outcome in ordered:
        if outcome.rows is None:
            failed.append(outcome.index)
            continue
        rows = outcome.rows
        if merged and rows and rows[0] == merged[-1]:
            rows = rows[1:]
        merged.extend(rows)
    return merged, failed


def assemble_result(
    plan: ChunkPlan,
    merged_rows: list[Any],
    summary_data: dict[str, Any] | None,
) -> dict[str, Any]:
    result = dict(summary_data or {})
    result[plan.array_location.property_name] = merged_rows
    return result


def _is_pipe_line(line: str) -> bool:
    return line.strip().startswith("|")


def _is_delimiter_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if not cells:
        return False
    return all(cell != "" and set(cell) <= set("-: ") for cell in cells)


def _find_dominant_table(markdown: str) -> _TableBlock | None:
    lines = markdown.splitlines()
    blocks: list[_TableBlock] = []

    index = 0
    while index < len(lines):
        if not _is_pipe_line(lines[index]):
            index += 1
            continue
        start = index
        end = index
        while end + 1 < len(lines) and _is_pipe_line(lines[end + 1]):
            end += 1
        run = lines[start : end + 1]
        if len(run) >= 2 and _is_delimiter_line(run[1]):
            blocks.append(
                _TableBlock(
                    start_line=start,
                    end_line=end,
                    header_line=run[0],
                    delimiter_line=run[1],
                    data_lines=run[2:],
                )
            )
        index = end + 1

    if not blocks:
        return None
    return max(blocks, key=lambda block: len(block.data_lines))
