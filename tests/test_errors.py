"""Tests for schemashift.errors."""

import pytest

from schemashift.errors import (
    AmbiguousFormatError,
    ConfigValidationError,
    DSLRuntimeError,
    DSLSyntaxError,
    FormatDetectionError,
    LLMGenerationError,
    ReaderError,
    ReviewRejectedError,
    SchemaShiftError,
    SchemaValidationError,
    UnsupportedFileError,
)


class TestErrorHierarchy:
    def test_schema_shift_error_is_base(self):
        assert issubclass(SchemaShiftError, Exception)

    def test_config_validation_error_inherits(self):
        assert issubclass(ConfigValidationError, SchemaShiftError)

    def test_dsl_syntax_error_inherits(self):
        assert issubclass(DSLSyntaxError, SchemaShiftError)

    def test_dsl_runtime_error_inherits(self):
        assert issubclass(DSLRuntimeError, SchemaShiftError)

    def test_format_detection_error_inherits(self):
        assert issubclass(FormatDetectionError, SchemaShiftError)

    def test_ambiguous_format_error_inherits_format_detection(self):
        assert issubclass(AmbiguousFormatError, FormatDetectionError)
        assert issubclass(AmbiguousFormatError, SchemaShiftError)

    def test_schema_validation_error_inherits(self):
        assert issubclass(SchemaValidationError, SchemaShiftError)

    def test_unsupported_file_error_inherits(self):
        assert issubclass(UnsupportedFileError, SchemaShiftError)

    def test_llm_generation_error_inherits(self):
        assert issubclass(LLMGenerationError, SchemaShiftError)

    def test_reader_error_inherits(self):
        assert issubclass(ReaderError, SchemaShiftError)


class TestDSLSyntaxError:
    def test_default_attributes(self):
        err = DSLSyntaxError("bad token")
        assert str(err) == "bad token"
        assert err.expression == ""
        assert err.position == -1

    def test_custom_attributes(self):
        err = DSLSyntaxError("unexpected char", expression="col('x') ++ 1", position=12)
        assert err.expression == "col('x') ++ 1"
        assert err.position == 12

    def test_is_catchable_as_schema_shift_error(self):
        with pytest.raises(SchemaShiftError):
            raise DSLSyntaxError("oops")


class TestDSLRuntimeError:
    def test_default_attributes(self):
        err = DSLRuntimeError("division by zero")
        assert str(err) == "division by zero"
        assert err.expression == ""
        assert err.target == ""

    def test_custom_attributes(self):
        err = DSLRuntimeError("column not found", expression="col('missing')", target="price")
        assert err.expression == "col('missing')"
        assert err.target == "price"

    def test_is_catchable_as_schema_shift_error(self):
        with pytest.raises(SchemaShiftError):
            raise DSLRuntimeError("oops")


class TestAmbiguousFormatError:
    def test_default_candidates_is_empty_list(self):
        err = AmbiguousFormatError("ambiguous")
        assert err.candidates == []

    def test_candidates_stored(self):
        err = AmbiguousFormatError("ambiguous", candidates=["fmt_a", "fmt_b"])
        assert err.candidates == ["fmt_a", "fmt_b"]

    def test_is_catchable_as_format_detection_error(self):
        with pytest.raises(FormatDetectionError):
            raise AmbiguousFormatError("ambiguous")

    def test_is_catchable_as_schema_shift_error(self):
        with pytest.raises(SchemaShiftError):
            raise AmbiguousFormatError("ambiguous", candidates=["x"])


class TestSimpleErrors:
    @pytest.mark.parametrize(
        "error_cls",
        [
            ConfigValidationError,
            FormatDetectionError,
            SchemaValidationError,
            UnsupportedFileError,
            LLMGenerationError,
            ReaderError,
        ],
    )
    def test_raises_and_message(self, error_cls):
        with pytest.raises(SchemaShiftError, match="test message"):
            raise error_cls("test message")


class TestReviewRejectedError:
    def test_is_schemashift_error(self):
        assert issubclass(ReviewRejectedError, SchemaShiftError)

    def test_not_format_detection_error(self):
        assert not issubclass(ReviewRejectedError, FormatDetectionError)

    def test_raises_with_message(self):
        with pytest.raises(ReviewRejectedError, match="rejected"):
            raise ReviewRejectedError("config was rejected")
