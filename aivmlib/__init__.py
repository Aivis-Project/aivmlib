import base64
import json
import onnx
import onnx.onnx_pb
import uuid
from google.protobuf.message import DecodeError
from pydantic import ValidationError
from typing import BinaryIO

from aivmlib.schemas.aivm_manifest import (
    AivmMetadata,
    AivmManifest,
    AivmManifestSpeaker,
    AivmManifestSpeakerStyle,
    DEFAULT_AIVM_MANIFEST,
    ModelArchitecture,
    ModelFormat,
)
from aivmlib.schemas.aivm_manifest_constants import DEFAULT_ICON_DATA_URL
from aivmlib.schemas.style_bert_vits2 import StyleBertVITS2HyperParameters


# AIVM / AIVMX ファイルフォーマットの仕様は下記ドキュメントを参照のこと
# ref: https://github.com/Aivis-Project/aivmlib#aivm-specification


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

    Raises:
        AivmValidationError: ハイパーパラメータのフォーマットが不正・スタイルベクトルが未指定・サポートされていないモデルアーキテクチャの場合
    """

    # 引数として受け取った BinaryIO のカーソルを先頭にシーク
    hyper_parameters_file.seek(0)
    if style_vectors_file is not None:
        style_vectors_file.seek(0)

    # Style-Bert-VITS2 系の音声合成モデルの場合
    if model_architecture in [ModelArchitecture.StyleBertVITS2, ModelArchitecture.StyleBertVITS2JPExtra]:

        # ハイパーパラメータファイル (JSON) を読み込んだ後、Pydantic でバリデーションする
        hyper_parameters_content = hyper_parameters_file.read().decode('utf-8')
        try:
            hyper_parameters = StyleBertVITS2HyperParameters.model_validate_json(hyper_parameters_content)
        except ValidationError:
            raise AivmValidationError(f"The format of the hyper-parameters file for {model_architecture} is incorrect.")

        # スタイルベクトルファイルを読み込む
        # Style-Bert-VITS2 モデルアーキテクチャの AIVM ファイルではスタイルベクトルが必須
        if style_vectors_file is None:
            raise AivmValidationError('Style vectors file is not specified.')
        style_vectors = style_vectors_file.read()

        # 引数として受け取った BinaryIO のカーソルを再度先頭に戻す
        hyper_parameters_file.seek(0)
        style_vectors_file.seek(0)

        # デフォルトの AIVM マニフェストをコピーした後、ハイパーパラメータに記載の値で一部を上書きする
        manifest = DEFAULT_AIVM_MANIFEST.model_copy()
        manifest.name = hyper_parameters.model_name
        # モデルアーキテクチャは Style-Bert-VITS2 系であれば異なる値が指定されても動作するよう、ハイパーパラメータの値を元に設定する
        if hyper_parameters.data.use_jp_extra:
            manifest.model_architecture = ModelArchitecture.StyleBertVITS2JPExtra
        else:
            manifest.model_architecture = ModelArchitecture.StyleBertVITS2
        # モデル UUID はランダムに生成
        manifest.uuid = uuid.uuid4()

        # spk2id の内容を反映
        manifest.speakers = [
            AivmManifestSpeaker(
                # ハイパーパラメータに記載の話者名を使用
                name = speaker_name,
                # デフォルトアイコンを使用
                icon = DEFAULT_ICON_DATA_URL,
                # JP-Extra の場合は日本語のみ、それ以外は日本語・アメリカ英語・標準中国語をサポート
                supported_languages = ['ja'] if hyper_parameters.data.use_jp_extra else ['ja', 'en-US', 'zh-CN'],
                # 話者 UUID はランダムに生成
                uuid = uuid.uuid4(),
                # ローカル ID は spk2id の ID の部分を使用
                local_id = speaker_index,
                # style2id の内容を反映
                styles = [
                    AivmManifestSpeakerStyle(
                        # "Neutral" はより分かりやすい "ノーマル" に変換する
                        # ただし、既にスタイル名が "ノーマル" のスタイルがある場合は "Neutral" のままにする
                        name = 'ノーマル' if (style_name == 'Neutral' and 'ノーマル' not in hyper_parameters.data.style2id) else style_name,
                        icon = None,
                        local_id = style_index,
                        voice_samples = [],
                    )
                    for style_name, style_index in hyper_parameters.data.style2id.items()
                ],
            )
            for speaker_name, speaker_index in hyper_parameters.data.spk2id.items()
        ]

        return AivmMetadata(
            manifest = manifest,
            hyper_parameters = hyper_parameters,
            style_vectors = style_vectors,
        )

    raise AivmValidationError(f"Unsupported model architecture: {model_architecture}.")


def validate_aivm_metadata(raw_metadata: dict[str, str]) -> AivmMetadata:
    """
    AIVM メタデータをバリデーションする

    Args:
        raw_metadata (dict[str, str]): 辞書形式の生のメタデータ

    Returns:
        AivmMetadata: バリデーションが完了した AIVM メタデータ

    Raises:
        AivmValidationError: AIVM メタデータのバリデーションに失敗した場合
    """

    # AIVM マニフェストが存在しない場合
    if not raw_metadata or not raw_metadata.get('aivm_manifest'):
        raise AivmValidationError('AIVM manifest not found.')

    # AIVM マニフェストのバリデーション
    try:
        aivm_manifest = AivmManifest.model_validate_json(raw_metadata['aivm_manifest'])
    except ValidationError:
        raise AivmValidationError('Invalid AIVM manifest format.')

    # ハイパーパラメータのバリデーション
    if 'aivm_hyper_parameters' in raw_metadata:
        try:
            if aivm_manifest.model_architecture in [ModelArchitecture.StyleBertVITS2, ModelArchitecture.StyleBertVITS2JPExtra]:
                aivm_hyper_parameters = StyleBertVITS2HyperParameters.model_validate_json(raw_metadata['aivm_hyper_parameters'])
            else:
                raise AivmValidationError(f"Unsupported hyper-parameters for model architecture: {aivm_manifest.model_architecture}.")
        except ValidationError:
            raise AivmValidationError('Invalid hyper-parameters format.')
    else:
        raise AivmValidationError('Hyper-parameters not found.')

    # スタイルベクトルのデコード
    aivm_style_vectors = None
    if 'aivm_style_vectors' in raw_metadata:
        try:
            base64_string = raw_metadata['aivm_style_vectors']
            aivm_style_vectors = base64.b64decode(base64_string)
        except Exception:
            raise AivmValidationError('Failed to decode style vectors.')

    # AivmMetadata オブジェクトを構築して返す
    return AivmMetadata(
        manifest = aivm_manifest,
        hyper_parameters = aivm_hyper_parameters,
        style_vectors = aivm_style_vectors,
    )


def read_aivm_metadata(aivm_file: BinaryIO) -> AivmMetadata:
    """
    AIVM ファイルから AIVM メタデータを読み込む

    Args:
        aivm_file (BinaryIO): AIVM ファイル

    Returns:
        AivmMetadata: AIVM メタデータ

    Raises:
        AivmValidationError: AIVM ファイルのフォーマットが不正・AIVM メタデータのバリデーションに失敗した場合
    """

    # 引数として受け取った BinaryIO のカーソルを先頭にシーク
    aivm_file.seek(0)

    # ファイルの内容を読み込む
    array_buffer = aivm_file.read()
    header_size = int.from_bytes(array_buffer[:8], 'little')

    # 引数として受け取った BinaryIO のカーソルを再度先頭に戻す
    aivm_file.seek(0)

    # ヘッダー部分を抽出
    header_bytes = array_buffer[8:8 + header_size]
    try:
        header_text = header_bytes.decode('utf-8')
        header_json = json.loads(header_text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise AivmValidationError('Failed to decode AIVM metadata. This file is not an AIVM (Safetensors) file.')

    # "__metadata__" キーから AIVM メタデータを取得
    raw_metadata = header_json.get('__metadata__')

    # バリデーションを行った上で、AivmMetadata オブジェクトを構築して返す
    return validate_aivm_metadata(raw_metadata)


def read_aivmx_metadata(aivmx_file: BinaryIO) -> AivmMetadata:
    """
    AIVMX ファイルから AIVM メタデータを読み込む

    Args:
        aivmx_file (BinaryIO): AIVMX ファイル

    Returns:
        AivmMetadata: AIVM メタデータ

    Raises:
        AivmValidationError: AIVMX ファイルのフォーマットが不正・AIVM メタデータのバリデーションに失敗した場合
    """

    # 引数として受け取った BinaryIO のカーソルを先頭にシーク
    aivmx_file.seek(0)

    # ONNX モデル (Protobuf) をロード
    try:
        model = onnx.load_model(aivmx_file, format='protobuf', load_external_data=False)
    except DecodeError:
        raise AivmValidationError('Failed to decode AIVM metadata. This file is not an AIVMX (ONNX) file.')

    # 引数として受け取った BinaryIO のカーソルを再度先頭に戻す
    aivmx_file.seek(0)

    # AIVM メタデータを取得
    raw_metadata = {prop.key: prop.value for prop in model.metadata_props}

    # バリデーションを行った上で、AivmMetadata オブジェクトを構築して返す
    return validate_aivm_metadata(raw_metadata)


def write_aivm_metadata(aivm_file: BinaryIO, aivm_metadata: AivmMetadata) -> bytes:
    """
    AIVM メタデータを AIVM ファイルに書き込む

    Args:
        aivm_file (BinaryIO): AIVM ファイル
        aivm_metadata (AivmMetadata): AIVM メタデータ

    Returns:
        bytes: 書き込みが完了した AIVM ファイルのバイト列

    Raises:
        AivmValidationError: AIVM ファイルのフォーマットが不正・スタイルベクトルが未指定の場合
    """

    # モデル形式を Safetensors に設定
    # AIVM ファイルのモデル形式は Safetensors のため、AIVM マニフェストにも明示的に反映する
    aivm_metadata.manifest.model_format = ModelFormat.Safetensors

    # AIVM マニフェストの内容をハイパーパラメータにも反映する
    # 結果は AivmMetadata オブジェクトに直接 in-place で反映される
    apply_aivm_manifest_to_hyper_parameters(aivm_metadata)

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

    # 引数として受け取った BinaryIO のカーソルを先頭にシーク
    aivm_file.seek(0)

    # AIVM ファイルの内容を一度に読み取る
    aivm_file_buffer = aivm_file.read()
    existing_header_size = int.from_bytes(aivm_file_buffer[:8], 'little')
    existing_header_bytes = aivm_file_buffer[8:8 + existing_header_size]
    existing_header_text = existing_header_bytes.decode('utf-8')
    try:
        existing_header = json.loads(existing_header_text)
    except json.JSONDecodeError:
        raise AivmValidationError('Failed to decode AIVM metadata. This file is not an AIVM (Safetensors) file.')

    # 引数として受け取った BinaryIO のカーソルを再度先頭に戻す
    aivm_file.seek(0)

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

    return new_aivm_file_content


def write_aivmx_metadata(aivmx_file: BinaryIO, aivm_metadata: AivmMetadata) -> bytes:
    """
    AIVM メタデータを AIVMX ファイルに書き込む

    Args:
        aivmx_file (BinaryIO): AIVMX ファイル
        aivm_metadata (AivmMetadata): AIVM メタデータ

    Returns:
        bytes: 書き込みが完了した AIVMX ファイルのバイト列

    Raises:
        AivmValidationError: AIVMX ファイルのフォーマットが不正・スタイルベクトルが未指定の場合
    """

    # モデル形式を ONNX に設定
    # AIVMX ファイルのモデル形式は ONNX のため、AIVM マニフェストにも明示的に反映する
    aivm_metadata.manifest.model_format = ModelFormat.ONNX

    # AIVM マニフェストの内容をハイパーパラメータにも反映する
    # 結果は AivmMetadata オブジェクトに直接 in-place で反映される
    apply_aivm_manifest_to_hyper_parameters(aivm_metadata)

    # 引数として受け取った BinaryIO のカーソルを先頭にシーク
    aivmx_file.seek(0)

    # ONNX モデル (Protobuf) をロード
    try:
        model = onnx.load_model(aivmx_file, format='protobuf', load_external_data=False)
    except DecodeError:
        raise AivmValidationError('Failed to decode AIVM metadata. This file is not an AIVMX (ONNX) file.')

    # 引数として受け取った BinaryIO のカーソルを再度先頭に戻す
    aivmx_file.seek(0)

    # AIVM メタデータをシリアライズ
    # ONNX のメタデータ領域はネストなしの string から string への key-value でなければならないため、
    # すべてのメタデータを文字列にシリアライズして格納する
    metadata = {
        'aivm_manifest': aivm_metadata.manifest.model_dump_json(),
        'aivm_hyper_parameters': aivm_metadata.hyper_parameters.model_dump_json(),
    }
    if aivm_metadata.style_vectors is not None:
        # スタイルベクトルが存在する場合は Base64 エンコードして追加
        metadata['aivm_style_vectors'] = base64.b64encode(aivm_metadata.style_vectors).decode('utf-8')

    # メタデータを ONNX モデルに追加
    for key, value in metadata.items():
        # 同一のキーが存在する場合は上書き
        for prop in model.metadata_props:
            if prop.key == key:
                prop.value = value
                break
        else:
            model.metadata_props.append(onnx.StringStringEntryProto(key=key, value=value))

    # 新しい AIVMX ファイルの内容をシリアライズ
    new_aivmx_file_content = model.SerializeToString()

    return new_aivmx_file_content


def apply_aivm_manifest_to_hyper_parameters(aivm_metadata: AivmMetadata) -> None:
    """
    AIVM マニフェストの内容をハイパーパラメータにも反映する
    結果は AivmMetadata オブジェクトに直接 in-place で反映される

    Args:
        aivm_metadata (AivmMetadata): AIVM メタデータ

    Raises:
        AivmValidationError: スタイルベクトルが未指定の場合
    """

    # Style-Bert-VITS2 系の音声合成モデルの場合
    if aivm_metadata.manifest.model_architecture in [ModelArchitecture.StyleBertVITS2, ModelArchitecture.StyleBertVITS2JPExtra]:

        # スタイルベクトルが設定されていなければエラー
        if aivm_metadata.style_vectors is None:
            raise AivmValidationError('Style vectors are not set.')

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


class AivmValidationError(Exception):
    """
    AIVM / AIVMX ファイルの読み取り中にエラーが発生したときに発生する例外
    """
    pass
