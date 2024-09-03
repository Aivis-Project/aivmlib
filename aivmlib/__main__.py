
import rich
import typer
from pathlib import Path
from rich.rule import Rule
from rich.style import Style
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
            for speaker in metadata.manifest.speakers:
                for style in speaker.styles:
                    style.icon = '(Image Base64 DataURL)'
                    for sample in style.voice_samples:
                        sample.audio = '(Audio Base64 DataURL)'
            rich.print(Rule(title='AIVM Manifest:', characters='=', style=Style(color='#E33157')))
            rich.print(metadata.manifest)
            rich.print(Rule(title='Hyper Parameters:', characters='=', style=Style(color='#E33157')))
            rich.print(metadata.hyper_parameters)
            rich.print(Rule(characters='=', style=Style(color='#E33157')))
    except Exception as e:
        rich.print(Rule(characters='=', style=Style(color='#E33157')))
        rich.print(f'[red]Error reading AIVM file: {e}[/red]')
        rich.print(Rule(characters='=', style=Style(color='#E33157')))


@app.command()
def create_aivm(
    output_path: Annotated[Path, typer.Option('-o', '--output', help='Path to the output AIVM file')],
    safetensors_model_path: Annotated[Path, typer.Option('-m', '--model', help='Path to the Safetensors model file')],
    hyper_parameters_path: Annotated[Path, typer.Option('-h', '--hyper-parameters', help='Path to the hyper parameters file')],
    style_vectors_path: Annotated[Union[Path, None], typer.Option('-s', '--style-vectors', help='Path to the style vectors file (optional)')] = None,
    model_architecture: Annotated[ModelArchitecture, typer.Option('-a', '--model-architecture', help='Model architecture')] = ModelArchitecture.StyleBertVITS2JPExtra,
):
    """
    与えられたアーキテクチャ, 学習済みモデル, ハイパーパラメータ, スタイルベクトルから AIVM メタデータを生成した上で、
    それを書き込んだ仮の AIVM ファイルを生成する
    """
    try:
        with hyper_parameters_path.open('rb') as hyper_parameters_file:
            style_vectors_file = style_vectors_path.open('rb') if style_vectors_path else None
            metadata = generate_aivm_metadata(model_architecture, hyper_parameters_file, style_vectors_file)

            with safetensors_model_path.open('rb') as safetensors_file:
                new_aivm_file_content = write_aivm_metadata(safetensors_file, metadata)

                with output_path.open('wb') as f:
                    f.write(new_aivm_file_content)
                rich.print(Rule(characters='=', style=Style(color='#E33157')))
                rich.print(f'Generated AIVM file: {output_path}')
                rich.print(Rule(characters='=', style=Style(color='#E33157')))
    except Exception as e:
        rich.print(Rule(characters='=', style=Style(color='#E33157')))
        rich.print(f'[red]Error creating AIVM file: {e}[/red]')
        rich.print(Rule(characters='=', style=Style(color='#E33157')))


if __name__ == '__main__':
    app()
