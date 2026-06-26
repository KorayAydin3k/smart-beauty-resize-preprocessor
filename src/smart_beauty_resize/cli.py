from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from smart_beauty_resize.batch import (
    BatchConfig,
    ImageProcessingRecord,
    ProcessingStatus,
    process_batch,
)
from smart_beauty_resize.config import load_preprocessing_profile
from smart_beauty_resize.contracts import (
    ProfileConfigurationError,
    ResizeConfig,
    ResizeConfigurationError,
    SmartBeautyResizeError,
)
from smart_beauty_resize.io.contracts import InputPolicy, SourceImageLimits
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


@dataclass(frozen=True, slots=True)
class _ResolvedPreprocessingSettings:
    resize_config: ResizeConfig
    input_policy: InputPolicy
    source_limits: SourceImageLimits


def _resolve_preprocessing_settings(
    *,
    profile: Path | None,
    target_width: int | None,
    target_height: int | None,
    allow_upscale: bool | None,
    max_upscale_factor: float | None,
    padding_red: int | None,
    padding_green: int | None,
    padding_blue: int | None,
    input_policy: InputPolicy | None,
    max_source_width: int | None,
    max_source_height: int | None,
    max_source_pixels: int | None,
) -> _ResolvedPreprocessingSettings:
    """Resolve one unambiguous preprocessing configuration source.

    A versioned profile and manually supplied preprocessing options are
    intentionally mutually exclusive. This keeps configuration provenance
    unambiguous while preserving the historical manual defaults.
    """
    manual_options = {
        "--target-width": target_width,
        "--target-height": target_height,
        "--allow-upscale/--no-allow-upscale": allow_upscale,
        "--max-upscale-factor": max_upscale_factor,
        "--padding-red": padding_red,
        "--padding-green": padding_green,
        "--padding-blue": padding_blue,
        "--input-policy": input_policy,
        "--max-source-width": max_source_width,
        "--max-source-height": max_source_height,
        "--max-source-pixels": max_source_pixels,
    }

    if profile is not None:
        provided_manual_options = [
            option_name for option_name, value in manual_options.items() if value is not None
        ]
        if provided_manual_options:
            options = ", ".join(provided_manual_options)
            raise ProfileConfigurationError(
                f"--profile cannot be combined with manual preprocessing options: {options}"
            )

        loaded_profile = load_preprocessing_profile(profile)
        return _ResolvedPreprocessingSettings(
            resize_config=loaded_profile.resize_config,
            input_policy=loaded_profile.input_policy,
            source_limits=loaded_profile.source_limits,
        )

    if target_width is None or target_height is None:
        raise ResizeConfigurationError(
            "--target-width and --target-height are required when --profile is not used"
        )

    return _ResolvedPreprocessingSettings(
        resize_config=ResizeConfig(
            target_width=target_width,
            target_height=target_height,
            allow_upscale=True if allow_upscale is None else allow_upscale,
            max_upscale_factor=(1.5 if max_upscale_factor is None else max_upscale_factor),
            padding_value=(
                127 if padding_red is None else padding_red,
                127 if padding_green is None else padding_green,
                127 if padding_blue is None else padding_blue,
            ),
        ),
        input_policy=(InputPolicy.AUDIT_ONLY if input_policy is None else input_policy),
        source_limits=SourceImageLimits(
            max_width=max_source_width,
            max_height=max_source_height,
            max_pixels=max_source_pixels,
        ),
    )


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
    console.print(f"[bold]Dataset audit:[/bold] {artifacts.dataset_audit_path}")


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
    profile: Annotated[
        Path | None,
        typer.Option(
            "--profile",
            help=(
                "Versioned YAML preprocessing profile. Cannot be combined "
                "with manual preprocessing options."
            ),
        ),
    ] = None,
    target_width: Annotated[
        int | None,
        typer.Option(
            "--target-width",
            help="Target canvas width in pixels when --profile is not used.",
        ),
    ] = None,
    target_height: Annotated[
        int | None,
        typer.Option(
            "--target-height",
            help="Target canvas height in pixels when --profile is not used.",
        ),
    ] = None,
    allow_upscale: Annotated[
        bool | None,
        typer.Option(
            "--allow-upscale/--no-allow-upscale",
            help="Permit source images to be enlarged in manual configuration mode.",
        ),
    ] = None,
    max_upscale_factor: Annotated[
        float | None,
        typer.Option(
            "--max-upscale-factor",
            help="Maximum permitted enlargement factor in manual configuration mode.",
        ),
    ] = None,
    padding_red: Annotated[
        int | None,
        typer.Option(
            "--padding-red",
            min=0,
            max=255,
            help="Red channel value for constant padding in manual configuration mode.",
        ),
    ] = None,
    padding_green: Annotated[
        int | None,
        typer.Option(
            "--padding-green",
            min=0,
            max=255,
            help="Green channel value for constant padding in manual configuration mode.",
        ),
    ] = None,
    padding_blue: Annotated[
        int | None,
        typer.Option(
            "--padding-blue",
            min=0,
            max=255,
            help="Blue channel value for constant padding in manual configuration mode.",
        ),
    ] = None,
    input_policy: Annotated[
        InputPolicy | None,
        typer.Option(
            "--input-policy",
            case_sensitive=False,
            help=(
                "Source-image acceptance policy in manual mode. audit_only preserves "
                "historical conversion behavior; strict_rgb8 rejects non-RGB, "
                "non-8-bit, alpha, or non-three-channel sources. Profile mode reads "
                "the policy from the profile."
            ),
        ),
    ] = None,
    max_source_width: Annotated[
        int | None,
        typer.Option(
            "--max-source-width",
            min=1,
            help=(
                "Maximum source width accepted before full decode in manual mode. "
                "Omit to disable the width limit."
            ),
        ),
    ] = None,
    max_source_height: Annotated[
        int | None,
        typer.Option(
            "--max-source-height",
            min=1,
            help=(
                "Maximum source height accepted before full decode in manual mode. "
                "Omit to disable the height limit."
            ),
        ),
    ] = None,
    max_source_pixels: Annotated[
        int | None,
        typer.Option(
            "--max-source-pixels",
            min=1,
            help=(
                "Maximum source pixel count accepted before full decode in manual mode. "
                "Omit to disable the pixel-count limit."
            ),
        ),
    ] = None,
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
        settings = _resolve_preprocessing_settings(
            profile=profile,
            target_width=target_width,
            target_height=target_height,
            allow_upscale=allow_upscale,
            max_upscale_factor=max_upscale_factor,
            padding_red=padding_red,
            padding_green=padding_green,
            padding_blue=padding_blue,
            input_policy=input_policy,
            max_source_width=max_source_width,
            max_source_height=max_source_height,
            max_source_pixels=max_source_pixels,
        )

        batch_config = BatchConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            resize_config=settings.resize_config,
            output_format="png",
            overwrite=overwrite,
            fail_fast=fail_fast,
            preserve_directory_structure=(preserve_directory_structure),
            input_policy=settings.input_policy,
            source_limits=settings.source_limits,
        )

        result = process_batch(batch_config)

        artifacts = write_batch_artifacts(
            result,
            batch_config.output_dir,
        )

    except SmartBeautyResizeError as exc:
        error_console.print(
            f"[bold red]Error:[/bold red] {exc.__class__.__name__}: {escape(str(exc))}"
        )
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
