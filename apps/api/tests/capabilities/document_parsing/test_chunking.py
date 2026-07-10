from api.capabilities.document_parsing.chunking import (
    ChunkingConfig,
    assemble_result,
    build_chunk_plan,
    detect_table_row_count,
    locate_row_array,
    merge_chunk_rows,
)
from api.capabilities.document_parsing.models import ChunkOutcome


def _table_markdown(row_count: int) -> str:
    header = "| id | value |"
    delimiter = "| --- | --- |"
    rows = [f"| {index} | v{index} |" for index in range(row_count)]
    return "\n".join([header, delimiter, *rows])


def _row_array_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "value": {"type": "string"},
                    },
                    "required": ["id", "value"],
                    "additionalProperties": False,
                },
            },
            "total": {"type": "number"},
        },
        "required": ["rows", "total"],
        "additionalProperties": False,
    }


class TestDetectTableRowCount:
    def test_single_table(self) -> None:
        markdown = _table_markdown(5)
        assert detect_table_row_count(markdown) == 5

    def test_table_with_surrounding_prose(self) -> None:
        markdown = f"# Report\n\nSome intro text.\n\n{_table_markdown(3)}\n\nGrand total: 3\n"
        assert detect_table_row_count(markdown) == 3

    def test_no_table_returns_zero(self) -> None:
        assert detect_table_row_count("Just some prose with no tables.") == 0

    def test_picks_dominant_table_when_multiple_present(self) -> None:
        small = _table_markdown(2)
        large = _table_markdown(10)
        markdown = f"{small}\n\nSome separating text.\n\n{large}"
        assert detect_table_row_count(markdown) == 10


class TestLocateRowArray:
    def test_selects_single_array_of_object_property(self) -> None:
        location = locate_row_array(_row_array_schema())
        assert location is not None
        assert location.property_name == "rows"
        assert location.item_schema["properties"]["id"] == {"type": "integer"}

    def test_returns_none_when_no_array_property(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assert locate_row_array(schema) is None

    def test_returns_none_when_multiple_array_properties(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {"type": "object", "properties": {}},
                },
                "other_rows": {
                    "type": "array",
                    "items": {"type": "object", "properties": {}},
                },
            },
        }
        assert locate_row_array(schema) is None

    def test_ignores_array_of_scalars(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }
        assert locate_row_array(schema) is None


class TestBuildChunkPlan:
    def test_returns_none_below_threshold(self) -> None:
        markdown = _table_markdown(5)
        config = ChunkingConfig(row_threshold=60, rows_per_chunk=50)
        assert build_chunk_plan(markdown, _row_array_schema(), config) is None

    def test_returns_none_when_schema_unchunkable(self) -> None:
        markdown = _table_markdown(100)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        config = ChunkingConfig(row_threshold=60, rows_per_chunk=50)
        assert build_chunk_plan(markdown, schema, config) is None

    def test_returns_none_when_no_table(self) -> None:
        config = ChunkingConfig(row_threshold=60, rows_per_chunk=50)
        assert build_chunk_plan("no table here", _row_array_schema(), config) is None

    def test_engaged_plan_produces_disjoint_contiguous_ranges(self) -> None:
        markdown = _table_markdown(120)
        config = ChunkingConfig(row_threshold=60, rows_per_chunk=50)
        plan = build_chunk_plan(markdown, _row_array_schema(), config)

        assert plan is not None
        assert plan.total_rows == 120
        assert len(plan.chunks) == 3
        assert [(chunk.start_row, chunk.end_row) for chunk in plan.chunks] == [
            (0, 50),
            (50, 100),
            (100, 120),
        ]
        # contiguous and covers every row without gaps or overlap
        covered = sum(chunk.end_row - chunk.start_row for chunk in plan.chunks)
        assert covered == 120

    def test_each_chunk_repeats_header_and_delimiter(self) -> None:
        markdown = _table_markdown(100)
        config = ChunkingConfig(row_threshold=60, rows_per_chunk=50)
        plan = build_chunk_plan(markdown, _row_array_schema(), config)

        assert plan is not None
        for chunk in plan.chunks:
            lines = chunk.markdown.splitlines()
            assert lines[0] == "| id | value |"
            assert lines[1] == "| --- | --- |"
            assert len(lines) - 2 == chunk.end_row - chunk.start_row

    def test_chunk_schema_is_row_array_only(self) -> None:
        markdown = _table_markdown(100)
        config = ChunkingConfig(row_threshold=60, rows_per_chunk=50)
        plan = build_chunk_plan(markdown, _row_array_schema(), config)

        assert plan is not None
        assert set(plan.chunk_schema["properties"]) == {"rows"}
        assert plan.chunk_schema["required"] == ["rows"]
        assert plan.chunk_schema["additionalProperties"] is False

    def test_summary_schema_holds_non_array_fields(self) -> None:
        markdown = _table_markdown(100)
        config = ChunkingConfig(row_threshold=60, rows_per_chunk=50)
        plan = build_chunk_plan(markdown, _row_array_schema(), config)

        assert plan is not None
        assert plan.summary_schema is not None
        assert set(plan.summary_schema["properties"]) == {"total"}
        assert plan.summary_schema["required"] == ["total"]

    def test_summary_markdown_includes_leading_data_rows_when_preamble_empty(self) -> None:
        header = "| id | value |"
        delimiter = "| --- | --- |"
        label_rows = [
            "| Dobavljac | Acme d.o.o. |",
            "| Period | 01.05.2026-31.05.2026 |",
        ]
        data_rows = [f"| {index} | v{index} |" for index in range(100)]
        markdown = "\n".join([header, delimiter, *label_rows, *data_rows])
        config = ChunkingConfig(row_threshold=60, rows_per_chunk=50)

        plan = build_chunk_plan(markdown, _row_array_schema(), config)

        assert plan is not None
        assert "Dobavljac" in plan.summary_markdown
        assert "Period" in plan.summary_markdown

    def test_summary_markdown_bounds_leading_rows_to_rows_per_chunk(self) -> None:
        markdown = _table_markdown(100)
        config = ChunkingConfig(row_threshold=60, rows_per_chunk=50)

        plan = build_chunk_plan(markdown, _row_array_schema(), config)

        assert plan is not None
        assert "v49" in plan.summary_markdown
        assert "v50" not in plan.summary_markdown

    def test_summary_schema_is_none_when_no_other_properties(self) -> None:
        markdown = _table_markdown(100)
        schema = {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"id": {"type": "integer"}}},
                }
            },
            "required": ["rows"],
        }
        config = ChunkingConfig(row_threshold=60, rows_per_chunk=50)
        plan = build_chunk_plan(markdown, schema, config)

        assert plan is not None
        assert plan.summary_schema is None


class TestMergeChunkRows:
    def test_concatenates_in_index_order(self) -> None:
        outcomes = [
            ChunkOutcome(index=1, rows=[{"id": 2}], error=None),
            ChunkOutcome(index=0, rows=[{"id": 1}], error=None),
        ]
        merged, failed = merge_chunk_rows(outcomes)
        assert merged == [{"id": 1}, {"id": 2}]
        assert failed == []

    def test_reports_failed_indices(self) -> None:
        outcomes = [
            ChunkOutcome(index=0, rows=[{"id": 1}], error=None),
            ChunkOutcome(index=1, rows=None, error="boom"),
            ChunkOutcome(index=2, rows=[{"id": 3}], error=None),
        ]
        merged, failed = merge_chunk_rows(outcomes)
        assert merged == [{"id": 1}, {"id": 3}]
        assert failed == [1]

    def test_dedups_exact_adjacent_boundary_row(self) -> None:
        outcomes = [
            ChunkOutcome(index=0, rows=[{"id": 1}, {"id": 2}], error=None),
            ChunkOutcome(index=1, rows=[{"id": 2}, {"id": 3}], error=None),
        ]
        merged, failed = merge_chunk_rows(outcomes)
        assert merged == [{"id": 1}, {"id": 2}, {"id": 3}]
        assert failed == []

    def test_never_reorders_non_boundary_duplicates(self) -> None:
        outcomes = [
            ChunkOutcome(index=0, rows=[{"id": 1}], error=None),
            ChunkOutcome(index=1, rows=[{"id": 1}], error=None),
        ]
        merged, failed = merge_chunk_rows(outcomes)
        # boundary rows equal -> deduped, leaving a single occurrence
        assert merged == [{"id": 1}]
        assert failed == []


class TestAssembleResult:
    def test_sets_array_property_and_merges_summary(self) -> None:
        markdown = _table_markdown(100)
        config = ChunkingConfig(row_threshold=60, rows_per_chunk=50)
        plan = build_chunk_plan(markdown, _row_array_schema(), config)
        assert plan is not None

        result = assemble_result(plan, [{"id": 1, "value": "v1"}], {"total": 42})
        assert result == {"rows": [{"id": 1, "value": "v1"}], "total": 42}

    def test_handles_no_summary_data(self) -> None:
        markdown = _table_markdown(100)
        config = ChunkingConfig(row_threshold=60, rows_per_chunk=50)
        plan = build_chunk_plan(markdown, _row_array_schema(), config)
        assert plan is not None

        result = assemble_result(plan, [], None)
        assert result == {"rows": []}
