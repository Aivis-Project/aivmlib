import base64
import json
import uuid
from typing import BinaryIO

import onnx
import onnx.onnx_pb
from google.protobuf.message import DecodeError
from pydantic import ValidationError

from aivmlib.schemas.aivm_manifest import (
    DEFAULT_AIVM_MANIFEST,
    AivmManifest,
    AivmManifestSpeaker,
    AivmManifestSpeakerStyle,
    AivmMetadata,
    ModelArchitecture,
    ModelFormat,
)
from aivmlib.schemas.aivm_manifest_constants import DEFAULT_ICON_DATA_URL
from aivmlib.schemas.style_bert_vits2 import StyleBertVITS2HyperParameters


# AIVM / AIVMX ファイルフォーマットの仕様は下記ドキュメントを参照のこと
# ref: https://github.com/Aivis-Project/aivmlib#aivm-specification


def _load_and_validate_hyper_parameters_and_style_vectors(
    model_architecture: ModelArchitecture,
    hyper_parameters_file: BinaryIO,
    style_vectors_file: BinaryIO | None = None,
) -> tuple[StyleBertVITS2HyperParameters, bytes]:
    """
    ハイパーパラメータファイルとスタイルベクトルファイルを読み込み、バリデーションする内部メソッド

    Args:
        model_architecture (ModelArchitecture): 音声合成モデルのアーキテクチャ
        hyper_parameters_file (BinaryIO): ハイパーパラメータファイル
        style_vectors_file (BinaryIO | None): スタイルベクトルファイル

    Returns:
        tuple[StyleBertVITS2HyperParameters, bytes]: ハイパーパラメータオブジェクトとスタイルベクトルのバイト列

    Raises:
        AivmValidationError: ハイパーパラメータのフォーマットが不正・スタイルベクトルが未指定・サポートされていないモデルアーキテクチャの場合
    """

    # 引数として受け取った BinaryIO のカーソルを先頭にシーク
    hyper_parameters_file.seek(0)
    if style_vectors_file is not None:
        style_vectors_file.seek(0)

    # Style-Bert-VITS2 系の音声合成モデルの場合
    if model_architecture in [ModelArchitecture.StyleBertVITS2, ModelArchitecture.StyleBertVITS2JPExtra]:
        # ハイパーパラメータファイル (JSON) を読み込んだ後、Pydantic でバリデーション
        hyper_parameters_content = hyper_parameters_file.read().decode('utf-8')
        try:
            hyper_parameters = StyleBertVITS2HyperParameters.model_validate_json(hyper_parameters_content)
        except ValidationError:
            raise AivmValidationError(f'The format of the hyper-parameters file for {model_architecture} is incorrect.')

        # 話者情報とスタイル情報の存在チェック
        if not hyper_parameters.data.spk2id:
            raise AivmValidationError('No speaker information found in hyper-parameters.')
        if not hyper_parameters.data.style2id:
            raise AivmValidationError('No style information found in hyper-parameters.')

        # 話者 ID の重複チェック
        speaker_ids = set()
        for speaker_name, speaker_id in hyper_parameters.data.spk2id.items():
            if speaker_id in speaker_ids:
                duplicate_speakers = [name for name, id in hyper_parameters.data.spk2id.items() if id == speaker_id]
                duplicate_speakers_str = ','.join([f'"{name}"' for name in duplicate_speakers])
                raise AivmValidationError(
                    f'Duplicate speaker ID ({speaker_id}). Multiple speakers ({duplicate_speakers_str}) have the same ID.'
                )
            speaker_ids.add(speaker_id)

        # スタイル ID の重複チェック
        style_ids = set()
        for style_name, style_id in hyper_parameters.data.style2id.items():
            if style_id in style_ids:
                duplicate_styles = [name for name, id in hyper_parameters.data.style2id.items() if id == style_id]
                duplicate_styles_str = ','.join([f'"{name}"' for name in duplicate_styles])
                raise AivmValidationError(
                    f'Duplicate style ID ({style_id}). Multiple styles ({duplicate_styles_str}) have the same ID.'
                )
            style_ids.add(style_id)

        # スタイル ID のバリデーション
        # AIVM マニフェストの制約（0 <= id <= 31）を満たしているかチェック
        for style_name, style_id in hyper_parameters.data.style2id.items():
            if style_id < 0 or style_id > 31:
                raise AivmValidationError(
                    f'Style ID ({style_id}) of style "{style_name}" is out of valid range. Style ID must be between 0 and 31.'
                )

        # 話者 ID のバリデーション
        for speaker_name, speaker_id in hyper_parameters.data.spk2id.items():
            if speaker_id < 0:
                raise AivmValidationError(
                    f'Speaker ID ({speaker_id}) of speaker "{speaker_name}" is invalid. Speaker ID must be a non-negative integer.'
                )

        # スタイルベクトルファイルの読み込み
        # Style-Bert-VITS2 モデルアーキテクチャの AIVM ファイルではスタイルベクトルが必須
        if style_vectors_file is None:
            raise AivmValidationError('Style vectors file is not specified.')
        style_vectors = style_vectors_file.read()

        # 引数として受け取った BinaryIO のカーソルを再度先頭に戻す
        hyper_parameters_file.seek(0)
        style_vectors_file.seek(0)

        return hyper_parameters, style_vectors

    raise AivmValidationError(f'Unsupported model architecture: {model_architecture}.')


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

    # ハイパーパラメータとスタイルベクトルの読み込み・バリデーション
    hyper_parameters, style_vectors = _load_and_validate_hyper_parameters_and_style_vectors(
        model_architecture,
        hyper_parameters_file,
        style_vectors_file,
    )

    # Style-Bert-VITS2 系の音声合成モデルの場合
    if model_architecture in [ModelArchitecture.StyleBertVITS2, ModelArchitecture.StyleBertVITS2JPExtra]:
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
                name=speaker_name,
                # デフォルトアイコンを使用
                icon=DEFAULT_ICON_DATA_URL,
                # JP-Extra の場合は日本語のみ、それ以外は日本語・アメリカ英語・標準中国語をサポート
                supported_languages=['ja'] if hyper_parameters.data.use_jp_extra else ['ja', 'en-US', 'zh-CN'],
                # 話者 UUID はランダムに生成
                uuid=uuid.uuid4(),
                # ローカル ID は spk2id の ID の部分を使用
                local_id=speaker_index,
                # style2id の内容を反映
                styles=[
                    AivmManifestSpeakerStyle(
                        # "Neutral" はより分かりやすい "ノーマル" に変換する
                        # ただし、既にスタイル名が "ノーマル" のスタイルがある場合は "Neutral" のままにする
                        name='ノーマル'
                        if (style_name == 'Neutral' and 'ノーマル' not in hyper_parameters.data.style2id)
                        else style_name,
                        icon=None,
                        local_id=style_index,
                        voice_samples=[],
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

    raise AivmValidationError(f'Unsupported model architecture: {model_architecture}.')


def update_aivm_metadata(
    existing_metadata: AivmMetadata,
    hyper_parameters_file: BinaryIO,
    style_vectors_file: BinaryIO | None = None,
) -> tuple[AivmMetadata, list[str]]:
    """
    既存の AIVM メタデータを、新しいハイパーパラメータとスタイルベクトルで更新する（モデル差し替え用）
    既存の UUID やユーザー設定メタデータは可能な限り維持される

    Args:
        existing_metadata (AivmMetadata): 既存の AIVM メタデータ
        hyper_parameters_file (BinaryIO): 新しいハイパーパラメータファイル
        style_vectors_file (BinaryIO | None): 新しいスタイルベクトルファイル

    Returns:
        tuple[AivmMetadata, list[str]]: 更新された AIVM メタデータと警告メッセージのリスト

    Raises:
        AivmValidationError: ハイパーパラメータのフォーマットが不正・スタイルベクトルが未指定・サポートされていないモデルアーキテクチャの場合
    """

    warnings = []

    # 既存の AIVM マニフェストからモデルアーキテクチャを取得
    model_architecture = existing_metadata.manifest.model_architecture

    # ハイパーパラメータとスタイルベクトルの読み込み・バリデーション
    hyper_parameters, style_vectors = _load_and_validate_hyper_parameters_and_style_vectors(
        model_architecture,
        hyper_parameters_file,
        style_vectors_file,
    )

    # Style-Bert-VITS2 系の音声合成モデルの場合
    if model_architecture in [ModelArchitecture.StyleBertVITS2, ModelArchitecture.StyleBertVITS2JPExtra]:
        # 新しい話者・スタイル情報を取得
        new_spk2id = hyper_parameters.data.spk2id
        new_style2id = hyper_parameters.data.style2id

        # 指定された既存の AIVM マニフェストをコピーした後、ハイパーパラメータの記述に応じてモデルアーキテクチャを更新
        # NOTE: 音声合成モデル名は更新せず、既存の AIVM マニフェストの内容を維持している
        manifest = existing_metadata.manifest.model_copy()
        if hyper_parameters.data.use_jp_extra:
            manifest.model_architecture = ModelArchitecture.StyleBertVITS2JPExtra
        else:
            manifest.model_architecture = ModelArchitecture.StyleBertVITS2

        # Map: local_id -> speaker_name
        new_spk_id_to_name_map = {}
        for name, id in new_spk2id.items():
            new_spk_id_to_name_map[id] = name
        # Map: local_id -> style_name
        new_style_id_to_name_map = {}
        for name, id in new_style2id.items():
            new_style_id_to_name_map[id] = name
        processed_new_speaker_local_ids = set()
        updated_speakers = []

        # 既存の話者情報リストを基準にイテレート
        for existing_speaker in existing_metadata.manifest.speakers:
            speaker_local_id = existing_speaker.local_id

            # 既存話者の local_id が新しいハイパーパラメータの spk2id に存在するか確認
            if speaker_local_id in new_spk_id_to_name_map:
                processed_new_speaker_local_ids.add(speaker_local_id)
                processed_new_style_local_ids = set()
                updated_styles = []

                # 既存のスタイル情報リストを基準にイテレート
                for existing_style in existing_speaker.styles:
                    style_local_id = existing_style.local_id

                    # 既存スタイルの local_id が新しいハイパーパラメータの style2id に存在するか確認
                    if style_local_id in new_style_id_to_name_map:
                        processed_new_style_local_ids.add(style_local_id)

                        # 既存のスタイル情報を維持
                        updated_styles.append(existing_style)
                    else:
                        # スタイルが削除された場合
                        warnings.append(
                            f'話者「{existing_speaker.name}」のスタイル「{existing_style.name}」(ID: {style_local_id}) は、新しいハイパーパラメータに存在しないため削除されます。'
                        )

                # 新しいハイパーパラメータで追加されたスタイルを追加
                for style_name, style_local_id in new_style2id.items():
                    if style_local_id not in processed_new_style_local_ids:
                        # "Neutral" はより分かりやすい "ノーマル" に変換する
                        # ただし、既にスタイル名が "ノーマル" のスタイルがある場合は "Neutral" のままにする
                        new_style_name = (
                            'ノーマル' if (style_name == 'Neutral' and 'ノーマル' not in new_style2id) else style_name
                        )
                        updated_styles.append(
                            AivmManifestSpeakerStyle(
                                name=new_style_name,
                                icon=None,
                                local_id=style_local_id,
                                voice_samples=[],
                            )
                        )
                        warnings.append(
                            f'話者「{existing_speaker.name}」にスタイル「{style_name}」(ID: {style_local_id}) が新しく追加されました。'
                        )

                # モデルアーキテクチャが変更された場合に備え、supported_languages を計算し直す
                # JP-Extra の場合は日本語のみ、それ以外は日本語・アメリカ英語・標準中国語をサポート
                supported_languages = existing_speaker.supported_languages
                new_supported_languages = ['ja'] if hyper_parameters.data.use_jp_extra else ['ja', 'en-US', 'zh-CN']
                if supported_languages != new_supported_languages:
                    supported_languages = new_supported_languages
                    warnings.append(
                        f'話者「{existing_speaker.name}」の対応言語が変更されました: {", ".join(supported_languages)}'
                    )

                # 更新された話者情報を追加
                updated_speakers.append(
                    AivmManifestSpeaker(
                        **existing_speaker.model_dump(),  # 既存の話者情報を維持
                        supported_languages=supported_languages,  # 更新された対応言語情報
                        styles=updated_styles,  # 更新されたスタイル情報リスト
                    )
                )

            else:
                # 話者が削除された場合
                warnings.append(
                    f'話者「{existing_speaker.name}」(ID: {speaker_local_id}) は、新しいハイパーパラメータに存在しないため削除されます。'
                )

        # 新しいハイパーパラメータで追加された話者を追加
        for new_speaker_name, new_local_id in new_spk2id.items():
            if new_local_id not in processed_new_speaker_local_ids:
                # 新しいハイパーパラメータに含まれる全スタイルを追加
                new_speaker_styles = []
                for style_name, style_local_id in new_style2id.items():
                    # "Neutral" はより分かりやすい "ノーマル" に変換する
                    # ただし、既にスタイル名が "ノーマル" のスタイルがある場合は "Neutral" のままにする
                    new_style_name = (
                        'ノーマル' if (style_name == 'Neutral' and 'ノーマル' not in new_style2id) else style_name
                    )
                    new_speaker_styles.append(
                        AivmManifestSpeakerStyle(
                            name=new_style_name,
                            icon=None,
                            local_id=style_local_id,
                            voice_samples=[],
                        )
                    )

                # 新しい話者を追加
                updated_speakers.append(
                    AivmManifestSpeaker(
                        # ハイパーパラメータに記載の話者名を使用
                        name=new_speaker_name,
                        # デフォルトアイコンを使用
                        icon=DEFAULT_ICON_DATA_URL,
                        # JP-Extra の場合は日本語のみ、それ以外は日本語・アメリカ英語・標準中国語をサポート
                        supported_languages=['ja'] if hyper_parameters.data.use_jp_extra else ['ja', 'en-US', 'zh-CN'],
                        # 話者 UUID はランダムに生成
                        uuid=uuid.uuid4(),
                        # ローカル ID は spk2id の ID の部分を使用
                        local_id=new_local_id,
                        # style2id の内容を反映
                        styles=new_speaker_styles,
                    )
                )
                warnings.append(f'話者「{new_speaker_name}」(ID: {new_local_id}) が新しく追加されました。')

        # マニフェストに更新された話者情報リストを設定
        manifest.speakers = updated_speakers

        # 処理の結果、話者情報リストが空になった場合はエラーを投げる
        if not updated_speakers:
            raise AivmValidationError(
                'Update resulted in an empty speakers list. The AIVM manifest must have at least one speaker.'
            )

        # 処理の結果、いずれかの話者のスタイル情報リストが空になった場合はエラーを投げる
        for speaker in updated_speakers:
            if not speaker.styles:
                raise AivmValidationError(
                    f"Update resulted in speaker '{speaker.name}' (ID: {speaker.local_id}) having no styles. Each speaker must have at least one style."
                )

        return AivmMetadata(
            manifest=manifest,
            hyper_parameters=hyper_parameters,
            style_vectors=style_vectors,
        ), warnings

    raise AivmValidationError(f'Unsupported model architecture: {model_architecture}.')


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
            if aivm_manifest.model_architecture in [
                ModelArchitecture.StyleBertVITS2,
                ModelArchitecture.StyleBertVITS2JPExtra,
            ]:
                aivm_hyper_parameters = StyleBertVITS2HyperParameters.model_validate_json(
                    raw_metadata['aivm_hyper_parameters']
                )
            else:
                raise AivmValidationError(
                    f'Unsupported hyper-parameters for model architecture: {aivm_manifest.model_architecture}.'
                )
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
        manifest=aivm_manifest,
        hyper_parameters=aivm_hyper_parameters,
        style_vectors=aivm_style_vectors,
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
    header_bytes = array_buffer[8 : 8 + header_size]
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


def serialize_aivm_metadata(aivm_metadata: AivmMetadata) -> dict[str, str]:
    """
    AIVM メタデータを生の辞書形式にシリアライズする

    Args:
        aivm_metadata (AivmMetadata): AIVM メタデータ

    Returns:
        dict[str, str]: シリアライズされた AIVM メタデータ（文字列から文字列へのマップ）
    """

    # AIVM メタデータをシリアライズ
    # Safetensors / ONNX のメタデータ領域はネストなしの string から string への map でなければならないため、
    # すべてのメタデータを文字列にシリアライズして格納する
    raw_metadata = {}
    raw_metadata['aivm_manifest'] = aivm_metadata.manifest.model_dump_json()
    raw_metadata['aivm_hyper_parameters'] = aivm_metadata.hyper_parameters.model_dump_json()

    # スタイルベクトルが存在する場合は Base64 エンコードして追加
    if aivm_metadata.style_vectors is not None:
        raw_metadata['aivm_style_vectors'] = base64.b64encode(aivm_metadata.style_vectors).decode('utf-8')

    return raw_metadata


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

    # AIVM メタデータをシリアライズした上で、書き込む前にバリデーションを行う
    raw_metadata = serialize_aivm_metadata(aivm_metadata)
    validate_aivm_metadata(raw_metadata)

    # 引数として受け取った BinaryIO のカーソルを先頭にシーク
    aivm_file.seek(0)

    # AIVM ファイルの内容を一度に読み取る
    aivm_file_buffer = aivm_file.read()
    existing_header_size = int.from_bytes(aivm_file_buffer[:8], 'little')
    existing_header_bytes = aivm_file_buffer[8 : 8 + existing_header_size]
    existing_header_text = existing_header_bytes.decode('utf-8')
    try:
        existing_header = json.loads(existing_header_text)
    except json.JSONDecodeError:
        raise AivmValidationError('Failed to decode AIVM metadata. This file is not an AIVM (Safetensors) file.')

    # 引数として受け取った BinaryIO のカーソルを再度先頭に戻す (重要)
    # ファイルポインタを先頭に戻さないと、このメソッド終了後にユーザーがファイルを使用する際に
    # カーソルが末尾にある状態となり、正しく読み書きできなくなる可能性がある
    aivm_file.seek(0)

    # 既存の __metadata__ を取得または新規作成
    existing_metadata = existing_header.get('__metadata__', {})

    # 既存の __metadata__ に新しいメタデータを追加
    # 既に存在するキーは上書きされる
    existing_metadata.update(raw_metadata)
    existing_header['__metadata__'] = existing_metadata

    # ヘッダー JSON を UTF-8 にエンコード
    new_header_text = json.dumps(existing_header)
    new_header_bytes = new_header_text.encode('utf-8')

    # ヘッダーサイズを 8 バイトの符号なし Little-Endian 64bit 整数に変換
    new_header_size = len(new_header_bytes).to_bytes(8, 'little')

    # 新しい AIVM ファイルの内容を作成
    new_aivm_file_content = new_header_size + new_header_bytes + aivm_file_buffer[8 + existing_header_size :]

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

    # AIVM メタデータをシリアライズした上で、書き込む前にバリデーションを行う
    raw_metadata = serialize_aivm_metadata(aivm_metadata)
    validate_aivm_metadata(raw_metadata)

    # 引数として受け取った BinaryIO のカーソルを先頭にシーク
    aivmx_file.seek(0)

    # ONNX モデル (Protobuf) をロード
    try:
        model = onnx.load_model(aivmx_file, format='protobuf', load_external_data=False)
    except DecodeError:
        raise AivmValidationError('Failed to decode AIVM metadata. This file is not an AIVMX (ONNX) file.')

    # 引数として受け取った BinaryIO のカーソルを再度先頭に戻す (重要)
    # ファイルポインタを先頭に戻さないと、このメソッド終了後にユーザーがファイルを使用する際に
    # カーソルが末尾にある状態となり、正しく読み書きできなくなる可能性がある
    aivmx_file.seek(0)

    # メタデータを ONNX モデルに追加
    for key, value in raw_metadata.items():
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
    if aivm_metadata.manifest.model_architecture in [
        ModelArchitecture.StyleBertVITS2,
        ModelArchitecture.StyleBertVITS2JPExtra,
    ]:
        # スタイルベクトルが設定されていなければエラー
        if aivm_metadata.style_vectors is None:
            raise AivmValidationError('Style vectors are not set.')

        # モデル名を反映
        aivm_metadata.hyper_parameters.model_name = aivm_metadata.manifest.name

        # 環境依存のパスが含まれるため、training_files と validation_files は固定値に変更
        aivm_metadata.hyper_parameters.data.training_files = 'train.list'
        aivm_metadata.hyper_parameters.data.validation_files = 'val.list'

        # 話者名を反映
        new_spk2id: dict[str, int] = {}
        for speaker in aivm_metadata.manifest.speakers:
            local_id = speaker.local_id
            # 話者のローカル ID が元のハイパーパラメータに存在するかチェック
            old_key = None
            for key, id in aivm_metadata.hyper_parameters.data.spk2id.items():
                if id == local_id:
                    old_key = key
                    break
            # 存在すれば新しい話者名をキーとして追加
            if old_key is not None:
                new_spk2id[speaker.name] = local_id
            else:
                # 必ず AivmManifest.speakers[].local_id の値が spk2id に存在しなければならない
                raise AivmValidationError(f'Speaker ID "{local_id}" of speaker "{speaker.name}" is not found in hyper-parameters.')
        aivm_metadata.hyper_parameters.data.spk2id = new_spk2id
        aivm_metadata.hyper_parameters.data.n_speakers = len(new_spk2id)

        # スタイル名を反映
        new_style2id: dict[str, int] = {}
        for speaker in aivm_metadata.manifest.speakers:
            for style in speaker.styles:
                local_id = style.local_id
                # スタイルのローカル ID が元のハイパーパラメータに存在するかチェック
                old_key = None
                for key, id in aivm_metadata.hyper_parameters.data.style2id.items():
                    if id == local_id:
                        old_key = key
                        break
                # 存在すれば新しいスタイル名をキーとして追加
                if old_key is not None:
                    new_style2id[style.name] = local_id
                else:
                    # 必ず AivmManifest.speakers[].styles[].local_id の値が style2id に存在しなければならない
                    raise AivmValidationError(f'Style ID "{local_id}" of style "{style.name}" is not found in hyper-parameters.')
        aivm_metadata.hyper_parameters.data.style2id = new_style2id
        aivm_metadata.hyper_parameters.data.num_styles = len(new_style2id)


class AivmValidationError(Exception):
    """
    AIVM / AIVMX ファイルの読み取り中にエラーが発生したときに発生する例外
    """

    pass
