
from __future__ import annotations

from enum import StrEnum
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field, constr
from uuid import UUID
from typing import Annotated

from aivmlib.schemas.aivm_manifest_constants import DEFAULT_ICON_DATA_URL
from aivmlib.schemas.style_bert_vits2 import StyleBertVITS2HyperParameters


class ModelArchitecture(StrEnum):
    StyleBertVITS2 = 'Style-Bert-VITS2'
    StyleBertVITS2JPExtra = 'Style-Bert-VITS2 (JP-Extra)'


@dataclass
class AivmMetadata:
    """ AIVM ファイルに含まれる全てのメタデータ """
    # AIVM マニフェストの情報
    manifest: AivmManifest
    # ハイパーパラメータの情報
    hyper_parameters: StyleBertVITS2HyperParameters
    # スタイルベクトルの情報
    style_vectors: bytes | None = None


class AivmManifest(BaseModel):
    """ AIVM マニフェストのスキーマ """
    # AIVM マニフェストのバージョン (ex: 1.0)
    # 現在は 1.0 のみサポート
    manifest_version: Annotated[str, constr(pattern=r'^1\.0$')]
    # 音声合成モデルの名前
    name: Annotated[str, constr(min_length=1)]
    # 音声合成モデルの説明 (省略時は空文字列になる)
    description: str = ''
    # 音声合成モデルの利用規約 (Markdown 形式 / 省略時は空文字列になる)
    terms_of_use: str = ''
    # 音声合成モデルのアーキテクチャ (音声合成技術の種類)
    model_architecture: ModelArchitecture
    # 音声合成モデルを一意に識別する UUID
    uuid: UUID
    # 音声合成モデルのバージョン (SemVer 2.0 準拠 / ex: 1.0.0)
    version: Annotated[str, constr(pattern=r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$')]
    # 音声合成モデルの話者情報 (最低 1 人以上の話者が必要)
    speakers: list[AivmManifestSpeaker]

    # model_ 以下を Pydantic の保護対象から除外する
    model_config = ConfigDict(protected_namespaces=())

class AivmManifestSpeaker(BaseModel):
    """ AIVM マニフェストの話者情報 """
    # 話者の名前
    name: Annotated[str, constr(min_length=1)]
    # 話者の対応言語のリスト (ja, en, zh のような ISO 639-1 言語コード)
    supported_languages: list[Annotated[str, constr(min_length=2, max_length=2)]]
    # 話者を一意に識別する UUID
    uuid: UUID
    # 話者のローカル ID (この音声合成モデル内で話者を識別するための一意なローカル ID で、uuid とは異なる)
    local_id: int = Field(ge=0)
    # 話者のバージョン (SemVer 2.0 準拠 / ex: 1.0.0)
    version: Annotated[str, constr(pattern=r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$')]
    # 話者のスタイル情報 (最低 1 つ以上のスタイルが必要)
    styles: list[AivmManifestSpeakerStyle]

class AivmManifestSpeakerStyle(BaseModel):
    """ AIVM マニフェストの話者スタイル情報 """
    # スタイルの名前
    name: Annotated[str, constr(min_length=1)]
    # スタイルのアイコン画像 (Data URL)
    # 最初のスタイルのアイコン画像が話者単体のアイコン画像として使用される
    icon: str
    # スタイルのボイスサンプル (省略時は空配列になる)
    voice_samples: list[AivmManifestVoiceSample]
    # スタイルの ID (この話者内でスタイルを識別するための一意なローカル ID で、uuid とは異なる)
    local_id: int = Field(ge=0, le=31)

class AivmManifestVoiceSample(BaseModel):
    """ AIVM マニフェストのボイスサンプル情報 """
    # ボイスサンプルの音声ファイル (Data URL)
    audio: str
    # ボイスサンプルの書き起こし文
    transcript: str


# デフォルト表示用の AIVM マニフェスト
DEFAULT_AIVM_MANIFEST = AivmManifest(
    manifest_version = '1.0',
    name = 'Model Name',
    description = '',
    terms_of_use = '',
    model_architecture = ModelArchitecture.StyleBertVITS2JPExtra,
    uuid = UUID('00000000-0000-0000-0000-000000000000'),
    version = '1.0.0',
    speakers = [
        AivmManifestSpeaker(
            name = 'Speaker Name',
            supported_languages = ['ja'],
            uuid = UUID('00000000-0000-0000-0000-000000000000'),
            local_id = 0,
            version = '1.0.0',
            styles = [
                AivmManifestSpeakerStyle(
                    name = 'ノーマル',
                    icon = DEFAULT_ICON_DATA_URL,
                    voice_samples = [],
                    local_id = 0,
                ),
            ],
        ),
    ],
)
