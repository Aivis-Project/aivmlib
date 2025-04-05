import re
import traceback
from pathlib import Path
from typing import Annotated, Union

import rich
import typer
from rich.rule import Rule
from rich.style import Style

import aivmlib
from aivmlib.schemas.aivm_manifest import ModelArchitecture


app = typer.Typer(help='Aivis Voice Model File (.aivm/.aivmx) Utility Library')


@app.command()
def show_metadata(file_path: Annotated[Path, typer.Argument(help='Path to the AIVM / AIVMX file')]):
    """
    指定されたパスの AIVM / AIVMX ファイル内に格納されている AIVM メタデータを見やすく出力する
    """

    try:
        with file_path.open('rb') as file:
            if file_path.suffix == '.aivmx':
                metadata = aivmlib.read_aivmx_metadata(file)
            else:
                metadata = aivmlib.read_aivm_metadata(file)

            for speaker in metadata.manifest.speakers:
                speaker.icon = '(Image Base64 DataURL)'
                for style in speaker.styles:
                    if style.icon:
                        style.icon = '(Image Base64 DataURL)'
                    for sample in style.voice_samples:
                        sample.audio = '(Audio Base64 DataURL)'
            rich.print(Rule(title='AIVM Manifest:', characters='=', style=Style(color='#41A2EC')))
            rich.print(metadata.manifest)
            rich.print(Rule(title='Hyper Parameters:', characters='=', style=Style(color='#41A2EC')))
            rich.print(metadata.hyper_parameters)
            rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
    except Exception as e:
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
        rich.print(f'[red]Error reading AIVM or AIVMX file: {e}[/red]')
        rich.print(Rule(characters='-', style=Style(color='#41A2EC')))
        rich.print(traceback.format_exc())
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))


@app.command()
def create_aivm(
    output_path: Annotated[Path, typer.Option('-o', '--output', help='Path to the output AIVM file')],
    safetensors_model_path: Annotated[Path, typer.Option('-m', '--model', help='Path to the Safetensors model file')],
    hyper_parameters_path: Annotated[
        Union[Path, None], typer.Option('-h', '--hyper-parameters', help='Path to the hyper parameters file (optional)')
    ] = None,
    style_vectors_path: Annotated[
        Union[Path, None], typer.Option('-s', '--style-vectors', help='Path to the style vectors file (optional)')
    ] = None,
    model_architecture: Annotated[
        ModelArchitecture, typer.Option('-a', '--model-architecture', help='Model architecture')
    ] = ModelArchitecture.StyleBertVITS2JPExtra,
):
    """
    与えられたアーキテクチャ, 学習済みモデル, ハイパーパラメータ, スタイルベクトルから AIVM メタデータを生成した上で、
    それを書き込んだ仮の AIVM ファイルを生成する
    """

    # 拡張子チェック
    if safetensors_model_path.suffix != '.safetensors':
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
        rich.print('[red]Safetensors model file must have a .safetensors extension.[/red]')
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
        return
    if output_path.suffix != '.aivm':
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
        rich.print('[red]Output file must have a .aivm extension.[/red]')
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
        return

    try:
        # アーキテクチャに合わせて未指定のファイルパスを自動設定
        if model_architecture in [ModelArchitecture.StyleBertVITS2, ModelArchitecture.StyleBertVITS2JPExtra]:
            model_dir = safetensors_model_path.parent
            if not hyper_parameters_path:
                hyper_parameters_path = model_dir / 'config.json'
            if not style_vectors_path:
                style_vectors_path = model_dir / 'style_vectors.npy'

            # 必要なファイルが存在しない場合はエラーを発生させる
            if not hyper_parameters_path.exists():
                rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
                rich.print(f'[red]Hyper parameters file not found: {hyper_parameters_path}[/red]')
                rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
                return
            if not style_vectors_path.exists():
                rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
                rich.print(f'[red]Style vectors file not found: {style_vectors_path}[/red]')
                rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
                return
        else:
            rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
            rich.print(f'[red]Model architecture {model_architecture} is not supported.[/red]')
            rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
            return

        # AIVM メタデータを生成
        with hyper_parameters_path.open('rb') as hyper_parameters_file:
            with style_vectors_path.open('rb') as style_vectors_file:
                metadata = aivmlib.generate_aivm_metadata(model_architecture, hyper_parameters_file, style_vectors_file)

        # モデルファイル名からエポック数とステップ数を抽出
        model_file_name = safetensors_model_path.name
        epoch_match = re.search(r'e(\d{2,})', model_file_name)  # "e" の後ろに2桁以上の数字
        step_match = re.search(r's(\d{2,})', model_file_name)  # "s" の後ろに2桁以上の数字

        # エポック数を設定
        if epoch_match:
            metadata.manifest.training_epochs = int(epoch_match.group(1))
        # ステップ数を設定
        if step_match:
            metadata.manifest.training_steps = int(step_match.group(1))

        # AIVM ファイルを生成
        with safetensors_model_path.open('rb') as safetensors_file:
            new_aivm_file_content = aivmlib.write_aivm_metadata(safetensors_file, metadata)
            with output_path.open('wb') as f:
                f.write(new_aivm_file_content)

        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
        rich.print(f'Generated AIVM file: {output_path}')
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
    except Exception as e:
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
        rich.print(f'[red]Error creating AIVM file: {e}[/red]')
        rich.print(Rule(characters='-', style=Style(color='#41A2EC')))
        rich.print(traceback.format_exc())
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))


@app.command()
def create_aivmx(
    output_path: Annotated[Path, typer.Option('-o', '--output', help='Path to the output AIVMX file')],
    onnx_model_path: Annotated[Path, typer.Option('-m', '--model', help='Path to the ONNX model file')],
    hyper_parameters_path: Annotated[
        Union[Path, None], typer.Option('-h', '--hyper-parameters', help='Path to the hyper parameters file (optional)')
    ] = None,
    style_vectors_path: Annotated[
        Union[Path, None], typer.Option('-s', '--style-vectors', help='Path to the style vectors file (optional)')
    ] = None,
    model_architecture: Annotated[
        ModelArchitecture, typer.Option('-a', '--model-architecture', help='Model architecture')
    ] = ModelArchitecture.StyleBertVITS2JPExtra,
):
    """
    与えられたアーキテクチャ, 学習済みモデル, ハイパーパラメータ, スタイルベクトルから AIVM メタデータを生成した上で、
    それを書き込んだ仮の AIVMX ファイルを生成する
    """

    # 拡張子チェック
    if onnx_model_path.suffix != '.onnx':
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
        rich.print('[red]ONNX model file must have a .onnx extension.[/red]')
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
        return
    if output_path.suffix != '.aivmx':
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
        rich.print('[red]Output file must have a .aivmx extension.[/red]')
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
        return

    try:
        # アーキテクチャに合わせて未指定のファイルパスを自動設定
        if model_architecture in [ModelArchitecture.StyleBertVITS2, ModelArchitecture.StyleBertVITS2JPExtra]:
            model_dir = onnx_model_path.parent
            if not hyper_parameters_path:
                hyper_parameters_path = model_dir / 'config.json'
            if not style_vectors_path:
                style_vectors_path = model_dir / 'style_vectors.npy'

            # 必要なファイルが存在しない場合はエラーを発生させる
            if not hyper_parameters_path.exists():
                rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
                rich.print(f'[red]Hyper parameters file not found: {hyper_parameters_path}[/red]')
                rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
                return
            if not style_vectors_path.exists():
                rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
                rich.print(f'[red]Style vectors file not found: {style_vectors_path}[/red]')
                rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
                return
        else:
            rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
            rich.print(f'[red]Model architecture {model_architecture} is not supported.[/red]')
            rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
            return

        # AIVM メタデータを生成
        with hyper_parameters_path.open('rb') as hyper_parameters_file:
            with style_vectors_path.open('rb') as style_vectors_file:
                metadata = aivmlib.generate_aivm_metadata(model_architecture, hyper_parameters_file, style_vectors_file)

        # モデルファイル名からエポック数とステップ数を抽出
        model_file_name = onnx_model_path.name
        epoch_match = re.search(r'e(\d{2,})', model_file_name)  # "e" の後ろに2桁以上の数字
        step_match = re.search(r's(\d{2,})', model_file_name)  # "s" の後ろに2桁以上の数字

        # エポック数を設定
        if epoch_match:
            metadata.manifest.training_epochs = int(epoch_match.group(1))
        # ステップ数を設定
        if step_match:
            metadata.manifest.training_steps = int(step_match.group(1))

        # AIVMX ファイルを生成
        with onnx_model_path.open('rb') as onnx_file:
            new_aivmx_file_content = aivmlib.write_aivmx_metadata(onnx_file, metadata)
            with output_path.open('wb') as f:
                f.write(new_aivmx_file_content)

        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
        rich.print(f'Generated AIVMX file: {output_path}')
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
    except Exception as e:
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))
        rich.print(f'[red]Error creating AIVMX file: {e}[/red]')
        rich.print(Rule(characters='-', style=Style(color='#41A2EC')))
        rich.print(traceback.format_exc())
        rich.print(Rule(characters='=', style=Style(color='#41A2EC')))


if __name__ == '__main__':
    app()
