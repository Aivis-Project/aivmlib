
import rich
import typer
from io import BytesIO
from pathlib import Path
from typing import Annotated, Union

from aivmlib import read_aivm_metadata, generate_aivm_metadata, write_aivm_metadata
from aivmlib.schemas.aivm_manifest import ModelArchitecture


app = typer.Typer(help='Aivis Voice Model File (.aivm) Utility Library')


@app.command()
def show_metadata(
    aivm_file_path: Annotated[Path, typer.Argument(help='Path to the AIVM file')]
):
    """
    指定されたパスの AIVM ファイル内に記録されている AIVM メタデータを見やすく出力する
    """
    try:
        with aivm_file_path.open('rb') as aivm_file:
            metadata = read_aivm_metadata(aivm_file)
            rich.print(metadata)
    except Exception as e:
        rich.print(f'[red]Error reading AIVM file: {e}[/red]')


@app.command()
def create_aivm(
    model_architecture: Annotated[ModelArchitecture, typer.Argument(help='Model architecture')],
    safetensors_model_path: Annotated[Path, typer.Argument(help='Path to the Safetensors model file')],
    hyper_parameters_path: Annotated[Path, typer.Argument(help='Path to the hyper parameters file')],
    style_vectors_path: Annotated[Union[Path, None], typer.Argument(help='Path to the style vectors file (optional)')],
):
    """
    与えられたアーキテクチャ・ハイパーパラメータ・スタイルベクトルから AIVM メタデータを生成した上で、
    それを書き込んだ仮の AIVM ファイルを生成する
    """
    try:
        with hyper_parameters_path.open('rb') as hyper_parameters_file:
            style_vectors_file = style_vectors_path.open('rb') if style_vectors_path else None
            metadata = generate_aivm_metadata(model_architecture, hyper_parameters_file, style_vectors_file)

            with safetensors_model_path.open('rb') as safetensors_file:
                safetensors_data = BytesIO(safetensors_file.read())
                new_aivm_file = write_aivm_metadata(safetensors_data, metadata)

                output_path = Path('output.aivm')
                with output_path.open('wb') as f:
                    f.write(new_aivm_file.read())
                rich.print(f'Generated AIVM file at {output_path}')
    except Exception as e:
        rich.print(f'[red]Error creating AIVM file: {e}[/red]')


if __name__ == '__main__':
    app()
