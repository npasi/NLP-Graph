from __future__ import annotations

from pathlib import Path


class OutputPaths:
    """Resolves all pipeline output paths for a given book.

    Legacy mode (no args): mirrors the historic data/ layout.
    Run mode (output_root + run_id): puts everything under
        outputs/runs/<run_id>/ so each experiment is isolated.
    """

    def __init__(
        self,
        book_id: str,
        output_root: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self._book_id = book_id

        if output_root is not None and run_id is None:
            raise ValueError(
                "--run-id is required when --output-root is provided."
            )
        if run_id is not None and output_root is None:
            output_root = "outputs/runs"

        self._run_dir: Path | None = (
            Path(output_root) / run_id
            if (output_root and run_id)
            else None
        )

    # ------------------------------------------------------------------
    # Run dir (None in legacy mode)
    # ------------------------------------------------------------------

    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    # ------------------------------------------------------------------
    # Books / chapters
    # ------------------------------------------------------------------

    @property
    def books_root(self) -> Path:
        return self._run_dir / "books" if self._run_dir else Path("data/books")

    @property
    def book_chapters_dir(self) -> Path:
        return self.books_root / self._book_id / "chapters"

    # ------------------------------------------------------------------
    # BookNLP output
    # ------------------------------------------------------------------

    @property
    def booknlp_root(self) -> Path:
        return (
            self._run_dir / "booknlp_chapter_output"
            if self._run_dir
            else Path("data/booknlp_chapter_output")
        )

    @property
    def booknlp_book_dir(self) -> Path:
        return self.booknlp_root / self._book_id

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    @property
    def reports_root(self) -> Path:
        return (
            self._run_dir / "reports" if self._run_dir else Path("data/reports")
        )

    @property
    def reports_book_dir(self) -> Path:
        return self.reports_root / self._book_id

    # ------------------------------------------------------------------
    # Graphs
    # ------------------------------------------------------------------

    @property
    def graphs_root(self) -> Path:
        return (
            self._run_dir / "graphs" if self._run_dir else Path("data/graphs")
        )

    @property
    def graphs_chapters_dir(self) -> Path:
        return self.graphs_root / "chapters"

    # ------------------------------------------------------------------
    # ML / BERT datasets
    # ------------------------------------------------------------------

    @property
    def ml_root(self) -> Path:
        return self._run_dir / "ml" if self._run_dir else Path("data/ml")

    @property
    def ml_book_dir(self) -> Path:
        return self.ml_root / self._book_id

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    @property
    def logs_dir(self) -> Path:
        return (
            self._run_dir / "logs"
            if self._run_dir
            else Path("data/evaluation_logs")
        )
