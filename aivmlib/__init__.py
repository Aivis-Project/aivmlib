import base64
import json
import traceback
import uuid
from io import BytesIO
from pydantic import ValidationError
from typing import BinaryIO

from aivmlib.schemas.aivm_manifest import (
    AivmMetadata,
    AivmManifest,
    AivmManifestSpeaker,
    AivmManifestStyle,
    DEFAULT_AIVM_MANIFEST,
    ModelArchitecture,
)
from aivmlib.schemas.aivm_manifest_constants import DEFAULT_ICON_DATA_URL
from aivmlib.schemas.style_bert_vits2 import StyleBertVITS2HyperParameters


# AIVM は、Safetensors 形式の学習済み音声合成モデルのヘッダー領域の中に、カスタムメタデータとして
# 話者メタデータ・ハイパーパラメータ・スタイルベクトルといった各種情報を JSON 文字列として格納したファイルフォーマットである。
# Safetensors 形式の拡張仕様のため、そのまま通常の Safetensors ファイルとしてロードできる。
# Safetensors 同様、先頭 8bytes の符号なし Little-Endian 64bit 整数がヘッダーサイズ、その後ろにヘッダーサイズの長さだけ UTF-8 の JSON 文字列が続く。
# Safetensors のヘッダー JSON にはテンソルのオフセット等が格納されているが、"__metadata__" キーには string から string への map を自由に設定可能な仕様である。
# この仕様を活用し、AIVM は "__metadata__" 内の以下のキーに次のデータを JSON シリアライズして格納する。
# - "aivm_manifest": AIVM マニフェスト (マニフェストバージョンや話者メタデータを含む大半の情報が含まれる)
# - "aivm_hyper_parameters": ハイパーパラメータ (フォーマットはモデルアーキテクチャの実装依存)
# - "aivm_style_vectors": Base64 エンコードされたスタイルベクトル (フォーマットはモデルアーキテクチャの実装依存 / モデルアーキテクチャ次第では省略されうる)
# ref: https://github.com/huggingface/safetensors
# ref: https://huggingface.co/docs/safetensors/main/en/metadata_parsing


def generate_aivm_metadata(
    model_architecture: ModelArchitecture,
    hyper_parameters_file: BinaryIO,
    style_vectors_file: BinaryIO | None = None,
) -> AivmMetadata:
    """
    ハイパーパラメータファイルとスタイルベクトルファイルから AIVM メタデータを生成する

    Args:
        model_architecture (ModelArchitecture): 音声合成モデルのアーキテクチャ
        hyper_parameters_file (BinaryIO): ハイパーパラメータファイル
        style_vectors_file (BinaryIO | None): スタイルベクトルファイル

    Returns:
        AivmMetadata: AIVM メタデータ
    """

    # Style-Bert-VITS2 系の音声合成モデルの場合
    if model_architecture.startswith('Style-Bert-VITS2'):

        # ハイパーパラメータファイル (JSON) を読み込んだ後、Pydantic でバリデーションする
        hyper_parameters_content = hyper_parameters_file.read().decode('utf-8')
        try:
            hyper_parameters = StyleBertVITS2HyperParameters.model_validate_json(hyper_parameters_content)
        except ValidationError:
            traceback.print_exc()
            raise ValueError(f"The format of the hyper-parameters file for {model_architecture} is incorrect.")

        # スタイルベクトルファイルを読み込む
        # Style-Bert-VITS2 モデルアーキテクチャの AIVM ファイルではスタイルベクトルが必須
        if style_vectors_file is None:
            raise ValueError('Style vectors file is not specified.')
        style_vectors = style_vectors_file.read()

        # デフォルトの AIVM マニフェストをコピーした後、ハイパーパラメータに記載の値で一部を上書きする
        manifest = DEFAULT_AIVM_MANIFEST.model_copy()
        manifest.name = hyper_parameters.model_name
        # モデルアーキテクチャは Style-Bert-VITS2 系であれば異なる値が指定されても動作するように、ハイパーパラメータの値を使用する
        manifest.model_architecture = ModelArchitecture.StyleBertVITS2JPExtra if hyper_parameters.data.use_jp_extra else ModelArchitecture.StyleBertVITS2
        # モデル UUID はランダムに生成
        manifest.uuid = uuid.uuid4()

        # spk2id の内容を反映
        manifest.speakers = [
            AivmManifestSpeaker(
                # ハイパーパラメータに記載の話者名を使用
                name = speaker_name,
                # JP-Extra の場合は日本語のみ、それ以外は日本語・英語・中国語をサポート
                supported_languages = ['ja'] if hyper_parameters.data.use_jp_extra else ['ja', 'en', 'zh'],
                # 話者 UUID はランダムに生成
                uuid = uuid.uuid4(),
                # ローカル ID は spk2id の ID の部分を使用
                local_id = speaker_index,
                version = '1.0.0',
                # style2id の内容を反映
                styles = [
                    AivmManifestStyle(
                        name = style_name,
                        icon = DEFAULT_ICON_DATA_URL,
                        voice_samples = [],
                        local_id = style_index,
                    )
                    for style_name, style_index in hyper_parameters.data.style2id.items()
                ],
            )
            for speaker_name, speaker_index in hyper_parameters.data.spk2id.items()
        ]

        return AivmMetadata(
            manifest=manifest,
            hyper_parameters=hyper_parameters,
            style_vectors=style_vectors,
        )

    raise ValueError(f"Unsupported model architecture: {model_architecture}.")


def read_aivm_metadata(aivm_file: BinaryIO) -> AivmMetadata:
    """
    AIVM ファイルから AIVM メタデータを読み込む

    Args:
        aivm_file (BinaryIO): AIVM ファイル

    Returns:
        AivmMetadata: AIVM メタデータ
    """

    # ファイルの内容を読み込む
    array_buffer = aivm_file.read()
    header_size = int.from_bytes(array_buffer[:8], 'little')

    # ヘッダー部分を抽出
    header_bytes = array_buffer[8:8 + header_size]
    header_text = header_bytes.decode('utf-8')
    header_json = json.loads(header_text)

    # "__metadata__" キーから AIVM メタデータを取得
    metadata = header_json.get('__metadata__')
    if not metadata or not metadata.get('aivm_manifest'):
        raise ValueError('AIVM manifest not found.')

    # AIVM マニフェストをパースしてバリデーション
    try:
        aivm_manifest = AivmManifest.model_validate_json(metadata['aivm_manifest'])
    except ValidationError:
        traceback.print_exc()
        raise ValueError('Invalid AIVM manifest format.')

    # ハイパーパラメータのバリデーション
    if 'aivm_hyper_parameters' in metadata:
        try:
            if aivm_manifest.model_architecture.startswith('Style-Bert-VITS2'):
                aivm_hyper_parameters = StyleBertVITS2HyperParameters.model_validate_json(metadata['aivm_hyper_parameters'])
            else:
                raise ValueError(f"Unsupported hyper-parameters for model architecture: {aivm_manifest.model_architecture}.")
        except ValidationError:
            traceback.print_exc()
            raise ValueError('Invalid hyper-parameters format.')
    else:
        raise ValueError('Hyper-parameters not found.')

    # スタイルベクトルのデコード
    aivm_style_vectors = None
    if 'aivm_style_vectors' in metadata:
        try:
            base64_string = metadata['aivm_style_vectors']
            aivm_style_vectors = base64.b64decode(base64_string)
        except Exception:
            traceback.print_exc()
            raise ValueError('Failed to decode style vectors.')

    return AivmMetadata(
        manifest=aivm_manifest,
        hyper_parameters=aivm_hyper_parameters,
        style_vectors=aivm_style_vectors,
    )


def write_aivm_metadata(aivm_file: BinaryIO, aivm_metadata: AivmMetadata) -> BytesIO:
    """
    AIVM メタデータを AIVM ファイルに書き込む

    Args:
        aivm_file (BinaryIO): AIVM ファイル
        aivm_metadata (AivmMetadata): AIVM メタデータ

    Returns:
        BytesIO: 書き込みが完了した AIVM or (メタデータが書き込まれていない素の Safetensors) ファイル
    """

    # Style-Bert-VITS2 系の音声合成モデルでは、AIVM マニフェストの内容をハイパーパラメータにも反映する
    if aivm_metadata.manifest.model_architecture.startswith('Style-Bert-VITS2'):

        # スタイルベクトルが設定されていなければエラー
        if aivm_metadata.style_vectors is None:
            raise ValueError('Style vectors are not set.')

        # モデル名を反映
        aivm_metadata.hyper_parameters.model_name = aivm_metadata.manifest.name

        # 環境依存のパスが含まれるため、training_files と validation_files は固定値に変更
        aivm_metadata.hyper_parameters.data.training_files = 'train.list'
        aivm_metadata.hyper_parameters.data.validation_files = 'val.list'

        # 話者名を反映
        new_spk2id = {speaker.name: speaker.local_id for speaker in aivm_metadata.manifest.speakers}
        aivm_metadata.hyper_parameters.data.spk2id = new_spk2id

        # スタイル名を反映
        new_style2id = {style.name: style.local_id for speaker in aivm_metadata.manifest.speakers for style in speaker.styles}
        aivm_metadata.hyper_parameters.data.style2id = new_style2id

    # AIVM メタデータをシリアライズ
    # Safetensors のメタデータ領域はネストなしの string から string への map でなければならないため、
    # すべてのメタデータを文字列にシリアライズして格納する
    metadata = {
        'aivm_manifest': aivm_metadata.manifest.model_dump_json(),
        'aivm_hyper_parameters': aivm_metadata.hyper_parameters.model_dump_json(),
    }
    if aivm_metadata.style_vectors is not None:
        # スタイルベクトルが存在する場合は Base64 エンコードして追加
        metadata['aivm_style_vectors'] = base64.b64encode(aivm_metadata.style_vectors).decode('utf-8')

    # AIVM ファイルの内容を一度に読み取る
    aivm_file_buffer = aivm_file.read()
    existing_header_size = int.from_bytes(aivm_file_buffer[:8], 'little')
    existing_header_bytes = aivm_file_buffer[8:8 + existing_header_size]
    existing_header_text = existing_header_bytes.decode('utf-8')
    existing_header = json.loads(existing_header_text)

    # 既存の __metadata__ を取得または新規作成
    existing_metadata = existing_header.get('__metadata__', {})

    # 既存の __metadata__ に新しいメタデータを追加
    # 既に存在するキーは上書きされる
    existing_metadata.update(metadata)
    existing_header['__metadata__'] = existing_metadata

    # ヘッダー JSON を UTF-8 にエンコード
    new_header_text = json.dumps(existing_header)
    new_header_bytes = new_header_text.encode('utf-8')

    # ヘッダーサイズを 8 バイトの符号なし Little-Endian 64bit 整数に変換
    new_header_size = len(new_header_bytes).to_bytes(8, 'little')

    # 新しい AIVM ファイルの内容を作成
    new_aivm_file_content = new_header_size + new_header_bytes + aivm_file_buffer[8 + existing_header_size:]
    new_aivm_file = BytesIO(new_aivm_file_content)
    new_aivm_file.seek(0)

    return new_aivm_file


class AivmParseError(Exception):
    """
    AIVM ファイルのパース中にエラーが発生したときに発生する例外
    """
    pass
