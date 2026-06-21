from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from smart_beauty_resize.batch import (
    BatchConfig,
    ImageProcessingRecord,
    ProcessingStatus,
    process_batch,
)
from smart_beauty_resize.contracts import (
    ResizeConfig,
    SmartBeautyResizeError,
)
from smart_beauty_resize.provenance import (
    BatchArtifactPaths,
    write_batch_artifacts,
)

app = typer.Typer(
    name="smart-beauty-resize",
    help=(
        "Deterministic, aspect-ratio-preserving image resize "
        "preprocessing for Smart Beauty datasets."
    ),
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()
error_console = Console(stderr=True)


def _build_summary_table(
    *,
    run_id: str,
    total: int,
    successful: int,
    failed: int,
    skipped: int,
    success_rate: float,
) -> Table:
    """Build the terminal summary table for one completed run."""
    table = Table(
        title="Smart Beauty batch summary",
        show_header=True,
        header_style="bold",
    )

    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Run ID", run_id)
    table.add_row("Total discovered", str(total))
    table.add_row("Successful", str(successful))
    table.add_row("Failed", str(failed))
    table.add_row("Skipped", str(skipped))
    table.add_row("Success rate", f"{success_rate:.2f}%")

    return table


def _print_record_status(
    record: ImageProcessingRecord,
    *,
    verbose: bool,
) -> None:
    """Print one per-image status when relevant."""
    if record.status is ProcessingStatus.SUCCESS:
        if verbose:
            console.print(f"[green]SUCCESS[/green] {record.source_relative_path.as_posix()}")
        return

    if record.status is ProcessingStatus.SKIPPED:
        style = "yellow"
        label = "SKIPPED"
    else:
        style = "red"
        label = "FAILED"

    error_type = record.error_type or "UnknownError"
    error_message = record.error_message or "No error message"

    console.print(
        f"[{style}]{label}[/{style}] "
        f"{record.source_relative_path.as_posix()} "
        f"[{error_type}] {error_message}"
    )


def _print_artifact_paths(
    artifacts: BatchArtifactPaths,
) -> None:
    """Print persisted audit-artifact locations."""
    console.print()
    console.print(f"[bold]Run directory:[/bold] {artifacts.run_directory}")
    console.print(f"[bold]Manifest:[/bold] {artifacts.manifest_path}")
    console.print(f"[bold]Summary:[/bold] {artifacts.summary_path}")


@app.callback()
def root_command() -> None:
    """Smart Beauty deterministic resize preprocessing commands."""


@app.command("batch")
def batch_command(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir",
            "-i",
            help="Directory containing source images.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for processed images and run artifacts.",
        ),
    ],
    target_width: Annotated[
        int,
        typer.Option(
            "--target-width",
            help="Target canvas width in pixels.",
        ),
    ],
    target_height: Annotated[
        int,
        typer.Option(
            "--target-height",
            help="Target canvas height in pixels.",
        ),
    ],
    allow_upscale: Annotated[
        bool,
        typer.Option(
            "--allow-upscale/--no-allow-upscale",
            help="Permit source images to be enlarged.",
        ),
    ] = True,
    max_upscale_factor: Annotated[
        float,
        typer.Option(
            "--max-upscale-factor",
            help="Maximum permitted enlargement factor.",
        ),
    ] = 1.5,
    padding_red: Annotated[
        int,
        typer.Option(
            "--padding-red",
            min=0,
            max=255,
            help="Red channel value for constant padding.",
        ),
    ] = 127,
    padding_green: Annotated[
        int,
        typer.Option(
            "--padding-green",
            min=0,
            max=255,
            help="Green channel value for constant padding.",
        ),
    ] = 127,
    padding_blue: Annotated[
        int,
        typer.Option(
            "--padding-blue",
            min=0,
            max=255,
            help="Blue channel value for constant padding.",
        ),
    ] = 127,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite/--no-overwrite",
            help="Replace existing processed PNG outputs.",
        ),
    ] = False,
    fail_fast: Annotated[
        bool,
        typer.Option(
            "--fail-fast/--continue-on-error",
            help="Stop immediately on the first processing error.",
        ),
    ] = False,
    preserve_directory_structure: Annotated[
        bool,
        typer.Option(
            "--preserve-directories/--flat-output",
            help="Preserve source subdirectories in the output.",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Print successful per-image records.",
        ),
    ] = False,
) -> None:
    """Resize a directory of images and persist an auditable run manifest."""
    try:
        resize_config = ResizeConfig(
            target_width=target_width,
            target_height=target_height,
            allow_upscale=allow_upscale,
            max_upscale_factor=max_upscale_factor,
            padding_value=(
                padding_red,
                padding_green,
                padding_blue,
            ),
        )

        batch_config = BatchConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            resize_config=resize_config,
            output_format="png",
            overwrite=overwrite,
            fail_fast=fail_fast,
            preserve_directory_structure=(preserve_directory_structure),
        )

        result = process_batch(batch_config)

        artifacts = write_batch_artifacts(
            result,
            batch_config.output_dir,
        )

    except SmartBeautyResizeError as exc:
        error_console.print(f"[bold red]Error:[/bold red] {exc.__class__.__name__}: {exc}")
        raise typer.Exit(code=1) from exc

    for record in result.records:
        _print_record_status(
            record,
            verbose=verbose,
        )

    console.print(
        _build_summary_table(
            run_id=result.summary.run_id,
            total=result.summary.total_discovered,
            successful=result.summary.successful,
            failed=result.summary.failed,
            skipped=result.summary.skipped,
            success_rate=result.summary.success_rate,
        )
    )

    _print_artifact_paths(artifacts)

    if result.summary.failed > 0:
        error_console.print()
        error_console.print("[bold red]Batch completed with failed images.[/bold red]")
        raise typer.Exit(code=2)


def main() -> None:
    """Run the Smart Beauty resize command-line interface."""
    app()


if __name__ == "__main__":
    main()
