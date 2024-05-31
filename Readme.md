
# aivmlib

💠 **aivmlib**: **Ai**vis **V**oice **M**odel File (.aivm) Utility **Lib**rary

**AIVM** (**Ai**vis **V**oice **M**odel) は、学習済みモデル・ハイパーパラメータ・スタイルベクトル・話者メタデータ（名前 / 概要 / アイコン / ボイスサンプル など）を 1 つのファイルにギュッとまとめた、AI 音声合成モデル用オープンファイルフォーマットです。  
AivisSpeech をはじめとした対応ソフトウェアに AIVM ファイルを追加することで、AI 音声合成モデルを簡単に利用できます。

このライブラリでは、AIVM ファイルのメタデータを読み書きするためのユーティリティを提供します。

## Usage

[`__main__.py`](aivmlib/__main__.py) に実装されている CLI ツールを参照してください。

## AIVM 1.0 Specification

AIVM は、Safetensors 形式の学習済み音声合成モデルのヘッダー領域の中に、カスタムメタデータとして話者メタデータ・ハイパーパラメータ・スタイルベクトルといった各種情報を JSON 文字列として格納したファイルフォーマットである。

### Safetensors 形式との互換性

Safetensors 形式の拡張仕様のため、そのまま通常の Safetensors ファイルとしてロードできる。  
Safetensors 同様、先頭 8bytes の符号なし Little-Endian 64bit 整数がヘッダーサイズ、その後ろにヘッダーサイズの長さだけ UTF-8 の JSON 文字列が続く。  
Safetensors のヘッダー JSON にはテンソルのオフセット等が格納されているが、`__metadata__` キーには string から string への map を自由に設定可能な仕様である。

この仕様を活用し、AIVM は `__metadata__` 内の以下のキーに次のデータを JSON シリアライズして格納する。

- `aivm_manifest` : AIVM マニフェスト
  - JSON 文字列で格納される
  - マニフェストバージョンや話者メタデータを含む大半の情報が含まれる
- `aivm_hyper_parameters` : ハイパーパラメータ
  - 格納フォーマットはモデルアーキテクチャの実装依存
  - `Style-Bert-VITS2`・`Style-Bert-VITS2 (JP-Extra)` モデルアーキテクチャでは JSON 文字列で格納される
- `aivm_style_vectors` : Base64 エンコードされたスタイルベクトル (バイナリ)
  - Base64 デコード後のフォーマットはモデルアーキテクチャの実装依存
  - `Style-Bert-VITS2`・`Style-Bert-VITS2 (JP-Extra)` モデルアーキテクチャでは NumPy 配列 (.npy) を Base64 エンコードした文字列で格納される
  - モデルアーキテクチャ次第では省略されうる

### AIVM マニフェストの仕様

AIVM マニフェストは JSON 形式とする。

AIVM マニフェストには、マニフェストバージョン (= AIVM ファイルバージョン)・モデルアーキテクチャ・モデル名・話者メタデータ・スタイル情報などが含まれる。  
JSON 形式の都合上、画像や音声データは Base64 エンコードされた文字列で格納される。

AIVM マニフェストのフィールド定義は以下の通り ([Pydantic モデル定義](aivmlib/schemas/aivm_manifest.py) より抜粋) 。

```python
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
    styles: list[AivmManifestStyle]

class AivmManifestStyle(BaseModel):
    """ AIVM マニフェストのスタイル情報 """
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
```

### 参考資料

- [Safetensors](https://github.com/huggingface/safetensors)
- [Safetensors Metadata Parsing](https://huggingface.co/docs/safetensors/main/en/metadata_parsing)

## License

[MIT License](License.txt)
