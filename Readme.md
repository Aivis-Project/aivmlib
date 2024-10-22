
# aivmlib

💠 **aivmlib**: **Ai**vis **V**oice **M**odel File (.aivm/.aivmx) Utility **Lib**rary

**AIVM** (**Ai**vis **V**oice **M**odel) / **AIVMX** (**Ai**vis **V**oice **M**odel for ONN**X**) は、学習済みモデル・ハイパーパラメータ・スタイルベクトル・話者メタデータ（名前・概要・ライセンス・アイコン・ボイスサンプル など）を 1 つのファイルにギュッとまとめた、AI 音声合成モデル用オープンファイルフォーマットです。

> [!NOTE]  
> **「AIVM」は、AIVM / AIVMX 両方のフォーマット仕様・メタデータ仕様の総称でもあります。**  
> 具体的には、AIVM ファイルは「AIVM メタデータを追加した Safetensors 形式」、AIVMX ファイルは「AIVM メタデータを追加した ONNX 形式」のモデルファイルです。  
> 「AIVM メタデータ」とは、AIVM 仕様に定義されている、学習済みモデルに紐づく各種メタデータのことをいいます。

[AivisSpeech](https://github.com/Aivis-Project/AivisSpeech) / [AivisSpeech-Engine](https://github.com/Aivis-Project/AivisSpeech-Engine) をはじめとした AIVM 仕様に対応したソフトウェアに AIVM / AIVMX ファイルを追加することで、AI 音声合成モデルを簡単に利用できます。

**aivmlib では、AIVM / AIVMX ファイル内のメタデータを読み書きするためのユーティリティを提供します。**  
この Readme の後半では、aivmlib で対応している各 AIVM 仕様についても説明しています。

> [!TIP]  
> **[AIVM Generator](https://aivm-generator.aivis-project.com/) では、ブラウザ上の GUI で AIVM / AIVMX ファイルを生成・編集できます。**  
> 機能的には aivmlib の上位互換です。手動で AIVM / AIVMX ファイルを生成・編集する際は AIVM Generator の利用をおすすめします。

- [aivmlib](#aivmlib)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)
- [AIVM Specification](#aivm-specification)
  - [Overview](#overview)
  - [AIVM File Format Specification](#aivm-file-format-specification)
    - [Safetensors 形式との互換性](#safetensors-形式との互換性)
    - [参考文献](#参考文献)
  - [AIVMX File Format Specification](#aivmx-file-format-specification)
    - [ONNX 形式との互換性](#onnx-形式との互換性)
    - [参考文献](#参考文献-1)
  - [AIVM Manifest Specification (Version 1.0)](#aivm-manifest-specification-version-10)
    - [サポートされるモデルアーキテクチャ](#サポートされるモデルアーキテクチャ)
    - [AIVM マニフェストのフィールド定義](#aivm-マニフェストのフィールド定義)

## Installation

pip でインストールすると、コマンドラインツール `aivmlib` も自動的にインストールされます。

```bash
pip install aivmlib
```

## Usage

以下に CLI ツール自体の使い方を示します。

```bash
$ aivmlib --help

 Usage: aivmlib [OPTIONS] COMMAND [ARGS]...

 Aivis Voice Model File (.aivm) Utility Library

╭─ Options ─────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.           │
│ --show-completion             Show completion for the current shell, to copy it   │
│                               or customize the installation.                      │
│ --help                        Show this message and exit.                         │
╰───────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────╮
│ create-aivm    与えられたアーキテクチャ, 学習済みモデル, ハイパーパラメータ,      │
│                スタイルベクトルから AIVM メタデータを生成した上で、               │
│                それを書き込んだ仮の AIVM ファイルを生成する                       │
│ create-aivmx   与えられたアーキテクチャ, 学習済みモデル, ハイパーパラメータ,      │
│                スタイルベクトルから AIVM メタデータを生成した上で、               │
│                それを書き込んだ仮の AIVMX ファイルを生成する                      │
│ show-metadata  指定されたパスの AIVM / AIVMX ファイル内に記録されている AIVM      │
│                メタデータを見やすく出力する                                       │
╰───────────────────────────────────────────────────────────────────────────────────╯

$ aivmlib show-metadata --help

 Usage: aivmlib show-metadata [OPTIONS] FILE_PATH

 指定されたパスの AIVM / AIVMX ファイル内に記録されている AIVM メタデータを見やすく出力する

╭─ Arguments ───────────────────────────────────────────────────────────────────────╮
│ *    file_path      PATH  Path to the AIVM / AIVMX file [default: None]           │
│                           [required]                                              │
╰───────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                       │
╰───────────────────────────────────────────────────────────────────────────────────╯

$ aivmlib create-aivm --help

 Usage: aivmlib create-aivm [OPTIONS]

 与えられたアーキテクチャ, 学習済みモデル, ハイパーパラメータ, スタイルベクトルから
 AIVM メタデータを生成した上で、それを書き込んだ仮の AIVM ファイルを生成する

╭─ Options ─────────────────────────────────────────────────────────────────────────╮
│ *  --output              -o      PATH                    Path to the output AIVM  │
│                                                          file                     │
│                                                          [default: None]          │
│                                                          [required]               │
│ *  --model               -m      PATH                    Path to the Safetensors  │
│                                                          model file               │
│                                                          [default: None]          │
│                                                          [required]               │
│    --hyper-parameters    -h      PATH                    Path to the hyper        │
│                                                          parameters file          │
│                                                          (optional)               │
│                                                          [default: None]          │
│    --style-vectors       -s      PATH                    Path to the style        │
│                                                          vectors file (optional)  │
│                                                          [default: None]          │
│    --model-architecture  -a      [Style-Bert-VITS2|Styl  Model architecture       │
│                                  e-Bert-VITS2            [default:                │
│                                  (JP-Extra)]             Style-Bert-VITS2         │
│                                                          (JP-Extra)]              │
│    --help                                                Show this message and    │
│                                                          exit.                    │
╰───────────────────────────────────────────────────────────────────────────────────╯

$ aivmlib create-aivmx --help

 Usage: aivmlib create-aivmx [OPTIONS]

 与えられたアーキテクチャ, 学習済みモデル, ハイパーパラメータ, スタイルベクトルから
 AIVM メタデータを生成した上で、それを書き込んだ仮の AIVMX ファイルを生成する

╭─ Options ─────────────────────────────────────────────────────────────────────────╮
│ *  --output              -o      PATH                    Path to the output AIVMX │
│                                                          file                     │
│                                                          [default: None]          │
│                                                          [required]               │
│ *  --model               -m      PATH                    Path to the ONNX model   │
│                                                          file                     │
│                                                          [default: None]          │
│                                                          [required]               │
│    --hyper-parameters    -h      PATH                    Path to the hyper        │
│                                                          parameters file          │
│                                                          (optional)               │
│                                                          [default: None]          │
│    --style-vectors       -s      PATH                    Path to the style        │
│                                                          vectors file (optional)  │
│                                                          [default: None]          │
│    --model-architecture  -a      [Style-Bert-VITS2|Styl  Model architecture       │
│                                  e-Bert-VITS2            [default:                │
│                                  (JP-Extra)]             Style-Bert-VITS2         │
│                                                          (JP-Extra)]              │
│    --help                                                Show this message and    │
│                                                          exit.                    │
╰───────────────────────────────────────────────────────────────────────────────────╯
```

```bash
# Safetensors 形式で保存された "Style-Bert-VITS2 (JP-Extra)" モデルアーキテクチャの学習済みモデルから AIVM ファイルを生成
# .safetensors と同じディレクトリに config.json と style_vectors.npy があることが前提
# -a オプションを省略した場合、既定で "Style-Bert-VITS2 (JP-Extra)" の学習済みモデルと判定される
$ aivmlib create-aivm -o ./output.aivm -m ./model.safetensors

# 明示的にハイパーパラメータとスタイルベクトルのパスを指定して生成
$ aivmlib create-aivm -o ./output.aivm -m ./model.safetensors -h ./config.json -s ./style-vectors.npy

# ONNX 形式で保存された "Style-Bert-VITS2" モデルアーキテクチャの学習済みモデルから AIVMX ファイルを生成
# .onnx と同じディレクトリに config.json と style_vectors.npy があることが前提
$ aivmlib create-aivmx -o ./output.aivmx -m ./model.onnx -a "Style-Bert-VITS2"

# 明示的にハイパーパラメータとスタイルベクトルのパスを指定して生成
$ aivmlib create-aivmx -o ./output.aivmx -m ./model.onnx -a "Style-Bert-VITS2" -h ./config.json -s ./style-vectors.npy

# AIVM ファイルに格納された AIVM メタデータを確認
$ aivmlib show-metadata ./output.aivm

# AIVMX ファイルに格納された AIVM メタデータを確認
$ aivmlib show-metadata ./output.aivmx
```

> [!TIP]  
> **ライブラリとしての使い方は、[`__main__.py`](aivmlib/__main__.py) に実装されている CLI ツールの実装を参照してください。**

> [!IMPORTANT]  
> **aivmlib は、AIVM ファイルフォーマットの読み込み/書き込み機能のみを有するライブラリです。**  
> 各モデルアーキテクチャごとの AI 音声合成モデルの推論ロジックや、aivmlib から取得したデータをどのようにユーザーに提示するかは、すべてライブラリの利用者に委ねられています。

## License

[MIT License](LICENSE)

<br>

# AIVM Specification

このセクションでは、「AIVM 仕様」に含まれる、下記の技術仕様を定義する。

<!-- no toc -->
- [AIVM File Format Specification](#aivm-file-format-specification)
- [AIVMX File Format Specification](#aivmx-file-format-specification)
- [AIVM Manifest Specification (Version 1.0)](#aivm-manifest-specification-version-10)

## Overview

学習済み AI 音声合成モデルと、その利用に必要な各種メタデータを単一ファイルにまとめることで、**ファイルの散逸や混乱を防ぎ、モデルの利用や共有を容易にすることを目的としている。**

**AIVM 仕様は、音声合成モデルのモデルアーキテクチャに依存しない。**  
異なるモデルアーキテクチャの音声合成モデルを共通のファイルフォーマットで扱えるよう、拡張性や汎用性を考慮して設計されている。

大元の学習済みモデルが単一の Safetensors または ONNX 形式で保存されているならば、原則どのようなモデルアーキテクチャであっても、メタデータを追加して AIVM / AIVMX ファイルを生成できる。  
設計にあたっては、変換処理を挟むことなく通常の Safetensors ファイルや ONNX ファイルとしてロードできるよう、既存エコシステムとの互換性を重視した。

> [!IMPORTANT]  
> **AIVM 仕様は、各モデルアーキテクチャごとの推論方法を定義しない。あくまでも「AI 音声合成モデルのメタデータをまとめたファイル」としての仕様のみを定義する。**  
> たとえば AIVM ファイルの場合、格納されている AI 音声合成モデルは PyTorch 用かもしれないし、TensorFlow 用かもしれない。  
> どのように AI 音声合成モデルの推論を行うかは、AIVM ファイルをサポートするソフトウェアの実装に委ねられている。

## AIVM File Format Specification

以下に、AIVM ファイルフォーマットの仕様を示す。

**AIVM** (**Ai**vis **V**oice **M**odel) は、[Safetensors](https://github.com/huggingface/safetensors) (.safetensors) 形式で保存された学習済みモデルのヘッダー領域の中に、カスタムメタデータとして話者メタデータ ([AIVM マニフェスト](#aivm-manifest-specification-version-10)) ・ハイパーパラメータ・スタイルベクトルといった各種情報を格納した、Safetensors 形式の拡張仕様である。

**「Safetensors 形式で保存された AI 音声合成モデル向けの、共通メタデータ記述仕様」** とも言える。

### Safetensors 形式との互換性

Safetensors 形式の拡張仕様のため、そのまま通常の Safetensors ファイルとしてロードできる。

Safetensors 同様、先頭 8bytes の符号なし Little-Endian 64bit 整数がヘッダーサイズ、その後ろにヘッダーサイズの長さだけ UTF-8 の JSON 文字列が続く。  
Safetensors のヘッダー JSON にはテンソルのオフセット等が格納されているが、`__metadata__` キーには string から string への map を自由に設定可能な仕様である。

この仕様を活用し、AIVM は `__metadata__` 内の以下のキーに、次のデータを JSON 文字列にシリアライズした上で格納する。

- **`aivm_manifest` : [AIVM マニフェスト](#aivm-manifest-specification-version-10)**
  - JSON 文字列として格納される
  - マニフェストバージョンや話者メタデータを含む大半の情報が含まれる
- **`aivm_hyper_parameters` : 音声合成モデルのハイパーパラメータ**
  - 格納フォーマットはモデルアーキテクチャ依存
  - `Style-Bert-VITS2`・`Style-Bert-VITS2 (JP-Extra)` モデルアーキテクチャでは JSON 文字列が格納される
- **`aivm_style_vectors` : Base64 エンコードされた音声合成モデルのスタイルベクトル (バイナリ)**
  - Base64 デコード後のフォーマットはモデルアーキテクチャ依存
  - `Style-Bert-VITS2`・`Style-Bert-VITS2 (JP-Extra)` モデルアーキテクチャでは NumPy 配列 (.npy) を Base64 エンコードした文字列が格納される
  - モデルアーキテクチャ次第では省略されうる

### 参考文献

- [Safetensors](https://github.com/huggingface/safetensors)
- [Safetensors Metadata Parsing](https://huggingface.co/docs/safetensors/main/en/metadata_parsing)

## AIVMX File Format Specification

以下に、AIVMX ファイルフォーマットの仕様を示す。

**AIVMX** (**Ai**vis **V**oice **M**odel for ONN**X**) は、[ONNX](https://onnx.ai/) 形式で保存された学習済みモデルのメタデータ領域の中に、カスタムメタデータとして話者メタデータ ([AIVM マニフェスト](#aivm-manifest-specification-version-10)) ・ハイパーパラメータ・スタイルベクトルといった各種情報を格納した、ONNX 形式の拡張仕様である。

**「ONNX 形式で保存された AI 音声合成モデル向けの、共通メタデータ記述仕様」** とも言える。

### ONNX 形式との互換性

執筆中...

### 参考文献

- [ONNX](https://onnx.ai/)
- [Open Neural Network Exchange Intermediate Representation (ONNX IR) Specification](https://onnx.ai/onnx/repo-docs/IR.html#optional-metadata)
- [ONNX Metadata](https://onnx.ai/onnx/repo-docs/MetadataProps.html)
- [How to populate onnx model with custom meta data map ?](https://github.com/onnx/sklearn-onnx/issues/214)

## AIVM Manifest Specification (Version 1.0)

以下に、AIVM ファイルフォーマットに含まれる、AIVM マニフェスト (Version 1.0) の仕様を示す。

**AIVM マニフェストには、マニフェストバージョン (= AIVM ファイルバージョン)・モデルアーキテクチャ・モデル名・話者メタデータ・スタイル情報などの、音声合成モデルの利用に必要となる様々な情報が含まれる。**

AIVM マニフェストのデータ形式は、JSON フォーマットで記述された UTF-8 文字列である。  
JSON フォーマットの都合上、画像や音声データは Base64 エンコードされた文字列で格納される。

### サポートされるモデルアーキテクチャ

- `Style-Bert-VITS2`
- `Style-Bert-VITS2 (JP-Extra)`

> [!IMPORTANT]  
> **AIVM ファイルをサポートするソフトウェアでは、自ソフトウェアではサポート対象外のモデルアーキテクチャの AIVM ファイルを、適切にバリデーションする必要がある。**  
> たとえば `Style-Bert-VITS2 (JP-Extra)` 以外のモデルアーキテクチャをサポートしないソフトウェアでは、`Style-Bert-VITS2` モデルアーキテクチャの AIVM ファイルのインストールを求められた際に「このモデルアーキテクチャには対応していません」とアラートを表示し、インストールを中止するよう実装すべき。

> [!IMPORTANT]  
> **技術的には上記以外のモデルアーキテクチャの音声合成モデルも格納可能だが、AIVM マニフェスト (Version 1.0) 仕様で公式に定義されているモデルアーキテクチャ文字列は上記のみ。**  
> 独自にモデルアーキテクチャ文字列を定義する場合は、既存のモデルアーキテクチャとの名前衝突や異なるソフト間での表記揺れが発生しないよう、細心の注意を払う必要がある。  
> なるべくこのリポジトリにプルリクエストを送信し、公式に AIVM 仕様に新しいモデルアーキテクチャのサポートを追加する形を取ることを推奨する。

### AIVM マニフェストのフィールド定義

以下は AIVM マニフェスト (Version 1.0) 仕様時点での AIVM マニフェストのフィールド定義を示す ([aivmlib の Pydantic スキーマ定義](aivmlib/schemas/aivm_manifest.py) より抜粋) 。

> [!IMPORTANT]  
> **AIVM マニフェスト内のフィールドは、今後 AIVM 仕様が更新された際に追加・拡張・削除される可能性がある。**  
> 今後のバージョン更新や追加のモデルアーキテクチャのサポートにより、AIVM マニフェストや AIVM ファイルフォーマット自体に新しいメタデータが追加されることも十分考えられる。  
> 現在有効な AIVM マニフェストバージョンは 1.0 のみ。

```python
class ModelArchitecture(StrEnum):
    StyleBertVITS2 = 'Style-Bert-VITS2'
    StyleBertVITS2JPExtra = 'Style-Bert-VITS2 (JP-Extra)'

class ModelFormat(StrEnum):
    Safetensors = 'Safetensors'
    ONNX = 'ONNX'

class AivmManifest(BaseModel):
    """ AIVM マニフェストのスキーマ """
    # AIVM マニフェストのバージョン (ex: 1.0)
    # 現在は 1.0 のみサポート
    manifest_version: Annotated[str, constr(pattern=r'^1\.0$')]
    # 音声合成モデルの名前
    name: Annotated[str, constr(min_length=1)]
    # 音声合成モデルの説明 (省略時は空文字列になる)
    description: str = ''
    # 音声合成モデルの作成者名のリスト (省略時は空リストになる)
    creators: list[str] = []
    # 音声合成モデルの利用規約 (Markdown 形式 / 省略時は空文字列になる)
    # カスタム利用規約を設定する場合を除き、原則各ライセンスへの URL リンクのみを記述する
    # 例: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
    terms_of_use: str = ''
    # 音声合成モデルのアーキテクチャ (音声合成技術の種類)
    model_architecture: ModelArchitecture
    # 音声合成モデルのモデル形式 (Safetensors または ONNX)
    # AIVM ファイル (.aivm) のモデル形式は Safetensors 、AIVMX ファイル (.aivmx) のモデル形式は ONNX である
    model_format: ModelFormat
    # 音声合成モデル学習時のエポック数 (省略時は None になる)
    training_epochs: Annotated[int, Field(ge=0)] | None = None
    # 音声合成モデル学習時のステップ数 (省略時は None になる)
    training_steps: Annotated[int, Field(ge=0)] | None = None
    # 音声合成モデルを一意に識別する UUID
    uuid: UUID
    # 音声合成モデルのバージョン (SemVer 2.0 準拠 / ex: 1.0.0)
    version: Annotated[str, constr(pattern=r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$')]
    # 音声合成モデルの話者情報 (最低 1 人以上の話者が必要)
    speakers: list[AivmManifestSpeaker]

class AivmManifestSpeaker(BaseModel):
    """ AIVM マニフェストの話者情報 """
    # 話者の名前
    name: Annotated[str, constr(min_length=1)]
    # 話者のアイコン画像 (Data URL)
    # 画像ファイル形式は 512×512 の JPEG (image/jpeg)・PNG (image/png) のいずれか (JPEG を推奨)
    icon: Annotated[str, constr(pattern=r'^data:image/(jpeg|png);base64,[A-Za-z0-9+/=]+$')]
    # 話者の対応言語のリスト (ja, en, zh のような ISO 639-1 言語コード)
    supported_languages: list[Annotated[str, constr(min_length=2, max_length=2)]]
    # 話者を一意に識別する UUID
    uuid: UUID
    # 話者のローカル ID (この音声合成モデル内で話者を識別するための一意なローカル ID で、uuid とは異なる)
    local_id: Annotated[int, Field(ge=0)]
    # 話者のスタイル情報 (最低 1 つ以上のスタイルが必要)
    styles: list[AivmManifestSpeakerStyle]

class AivmManifestSpeakerStyle(BaseModel):
    """ AIVM マニフェストの話者スタイル情報 """
    # スタイルの名前
    name: Annotated[str, constr(min_length=1)]
    # スタイルのアイコン画像 (Data URL, 省略可能)
    # 省略時は話者のアイコン画像がスタイルのアイコン画像として使われる想定
    # 画像ファイル形式は 512×512 の JPEG (image/jpeg)・PNG (image/png) のいずれか (JPEG を推奨)
    icon: Annotated[str, constr(pattern=r'^data:image/(jpeg|png);base64,[A-Za-z0-9+/=]+$')] | None = None
    # スタイルの ID (この話者内でスタイルを識別するための一意なローカル ID で、uuid とは異なる)
    local_id: Annotated[int, Field(ge=0, le=31)]  # 最大 32 スタイルまでサポート
    # スタイルのボイスサンプル (省略時は空配列になる)
    voice_samples: list[AivmManifestVoiceSample] = []

class AivmManifestVoiceSample(BaseModel):
    """ AIVM マニフェストのボイスサンプル情報 """
    # ボイスサンプルの音声ファイル (Data URL)
    # 音声ファイル形式は WAV (audio/wav, Codec: PCM 16bit)・M4A (audio/mp4, Codec: AAC-LC) のいずれか (M4A を推奨)
    audio: Annotated[str, constr(pattern=r'^data:audio/(wav|mp4);base64,[A-Za-z0-9+/=]+$')]
    # ボイスサンプルの書き起こし文
    # 書き起こし文は音声ファイルの発話内容と一致している必要がある
    transcript: Annotated[str, constr(min_length=1)]
```
