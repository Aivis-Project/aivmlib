
# aivmlib

![PyPI version](https://badge.fury.io/py/aivmlib.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/aivmlib.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

💠 **aivmlib**: **Ai**vis **V**oice **M**odel File (.aivm/.aivmx) Utility **Lib**rary

-----

**AIVM** (**Ai**vis **V**oice **M**odel) / **AIVMX** (**Ai**vis **V**oice **M**odel for ONN**X**) は、**学習済みモデル・ハイパーパラメータ・スタイルベクトル・話者メタデータ（名前・概要・ライセンス・アイコン・ボイスサンプル など）を 1 つのファイルにギュッとまとめた、AI 音声合成モデル用オープンファイルフォーマット**です。

> [!NOTE]  
> **「AIVM」は、AIVM / AIVMX 両方のフォーマット仕様・メタデータ仕様の総称でもあります。**  
> 具体的には、AIVM ファイルは「AIVM メタデータを追加した Safetensors 形式」、AIVMX ファイルは「AIVM メタデータを追加した ONNX 形式」のモデルファイルです。  
> 「AIVM メタデータ」とは、AIVM 仕様に定義されている、学習済みモデルに紐づく各種メタデータのことをいいます。

[AivisSpeech](https://github.com/Aivis-Project/AivisSpeech) / [AivisSpeech-Engine](https://github.com/Aivis-Project/AivisSpeech-Engine) をはじめとした AIVM 仕様に対応するソフトウェアに AIVM / AIVMX ファイルを追加することで、AI 音声合成モデルを簡単に利用できます。

**aivmlib / [aivmlib-web](https://github.com/Aivis-Project/aivmlib-web) では、AIVM / AIVMX ファイル内のメタデータを読み書きするためのユーティリティを提供します。**  
この aivmlib は、Python で書かれた AIVM 仕様のリファレンス実装です。Web ブラウザで利用する場合は [aivmlib-web](https://github.com/Aivis-Project/aivmlib-web) をご利用ください。

> [!TIP]  
> **[AIVM Generator](https://aivm-generator.aivis-project.com/) では、ブラウザ上の GUI でかんたんに AIVM / AIVMX ファイルを生成・編集できます。**  
> 手動で AIVM / AIVMX ファイルを生成・編集する際は AIVM Generator の利用をおすすめします。

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
  - [FAQ](#faq)
    - [Q. AIVM と AIVMX という 2 つのフォーマットが定義されているのはなぜですか？](#q-aivm-と-aivmx-という-2-つのフォーマットが定義されているのはなぜですか)
    - [Q. AIVM / AIVMX ファイルを既存のツールで読み込むことはできますか？](#q-aivm--aivmx-ファイルを既存のツールで読み込むことはできますか)
    - [Q. 既存の AI 音声合成モデルを AIVM / AIVMX に変換するにはどうすればよいですか？](#q-既存の-ai-音声合成モデルを-aivm--aivmx-に変換するにはどうすればよいですか)
    - [Q. AIVM マニフェストのバージョン管理はどのように行われますか？](#q-aivm-マニフェストのバージョン管理はどのように行われますか)
    - [Q. aivmlib と aivmlib-web の違いはなんですか？](#q-aivmlib-と-aivmlib-web-の違いはなんですか)
    - [Q. 新しいモデルアーキテクチャのサポートを追加するにはどうすればよいですか？](#q-新しいモデルアーキテクチャのサポートを追加するにはどうすればよいですか)
    - [Q. ライセンス情報はどのように記述すべきですか？](#q-ライセンス情報はどのように記述すべきですか)
    - [Q. 画像・音声データのサイズ制限はありますか？](#q-画像音声データのサイズ制限はありますか)
    - [Q. メタデータは手動で編集できますか？](#q-メタデータは手動で編集できますか)

## Installation

pip でインストールすると、コマンドラインツール `aivmlib` も自動的にインストールされます。  
Python 3.11 以上が必要です。

```bash
pip install aivmlib
```

開発時は Poetry を利用しています。

```bash
pip install poetry
git clone https://github.com/Aivis-Project/aivmlib.git
cd aivmlib
poetry install --with dev
poetry run aivmlib --help
```

## Usage

以下に CLI ツール自体の使い方を示します。

```bash
$ aivmlib --help

 Usage: aivmlib [OPTIONS] COMMAND [ARGS]...

 Aivis Voice Model File (.aivm/.aivmx) Utility Library

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

以下にコマンドの実行例を示します。

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
> **aivmlib / [aivmlib-web](https://github.com/Aivis-Project/aivmlib-web) は、AIVM / AIVMX ファイルフォーマットの読み込み/書き込み機能のみを有するライブラリです。**  
> 各モデルアーキテクチャごとの AI 音声合成モデルの推論ロジックや、aivmlib / aivmlib-web から取得したデータをどのようにユーザーに提示するかは、すべてライブラリの利用者に委ねられています。

## License

[MIT License](LICENSE)

<br>

# AIVM Specification

このセクションでは、「AIVM 仕様」に含まれる、下記の技術仕様を定義する。

<!-- no toc -->
- [AIVM File Format Specification](#aivm-file-format-specification)
- [AIVMX File Format Specification](#aivmx-file-format-specification)
- [AIVM Manifest Specification (Version 1.0)](#aivm-manifest-specification-version-10)
- [FAQ](#faq)

## Overview

学習済み AI 音声合成モデルと、その利用に必要な各種メタデータを単一ファイルにまとめることで、**ファイルの散逸や混乱を防ぎ、モデルの利用や共有を容易にすることを目的としている。**

> [!TIP]  
> 単一ファイルにまとめることで、AIVM / AIVMX ファイルをダウンロードして所定のフォルダに配置するだけで、対応ソフトウェアですぐに音声合成モデルを利用できるようになるといった、シンプルな運用が可能になる。  
> 圧縮ファイルではないため、展開の必要がないのもメリット。

**AIVM 仕様は、音声合成モデルのモデルアーキテクチャに依存しない。**  
異なるモデルアーキテクチャの音声合成モデルを共通のファイルフォーマットで扱えるよう、将来的な拡張性や汎用性を考慮して設計されている。

大元の学習済みモデルが単一の Safetensors または ONNX 形式で保存されているならば、原則どのようなモデルアーキテクチャであっても、メタデータを追加して AIVM / AIVMX ファイルを生成できる。  
設計にあたっては、変換処理を挟むことなく通常の Safetensors ファイルや ONNX ファイルとしてロードできるよう、既存エコシステムとの互換性を重視した。

> [!IMPORTANT]  
> **AIVM 仕様は、各モデルアーキテクチャごとの推論方法を定義しない。あくまでも「AI 音声合成モデルのメタデータをまとめたファイル」としての仕様のみを定義する。**  
> たとえば AIVM ファイルの場合、格納されている AI 音声合成モデルは PyTorch 用かもしれないし、TensorFlow 用かもしれない。  
> どのように AI 音声合成モデルの推論を行うかは、AIVM / AIVMX ファイルをサポートするソフトウェアの実装に委ねられている。

## AIVM File Format Specification

以下に、AIVM ファイルフォーマットの仕様を示す。

**AIVM** (**Ai**vis **V**oice **M**odel) は、[Safetensors](https://github.com/huggingface/safetensors) (.safetensors) 形式で保存された学習済みモデルのヘッダー領域の中に、カスタムメタデータとして話者メタデータ ([AIVM マニフェスト](#aivm-manifest-specification-version-10)) ・ハイパーパラメータ・スタイルベクトルといった各種情報を格納した、Safetensors 形式の拡張仕様である。

**「Safetensors 形式で保存された AI 音声合成モデル向けの、共通メタデータ記述仕様」** とも言える。

### Safetensors 形式との互換性

Safetensors 形式の拡張仕様のため、そのまま通常の Safetensors ファイルとしてロードできる。

Safetensors 同様、先頭 8bytes の符号なし Little-Endian 64bit 整数がヘッダーサイズ、その後ろにヘッダーサイズの長さだけ UTF-8 の JSON 文字列が続く。  
Safetensors のヘッダー JSON にはテンソルのオフセット等が格納されているが、`__metadata__` キーには string から string への map を自由に設定可能な仕様である。

この仕様を活用し、AIVM は `__metadata__` 内の以下のキーに、次の文字列データを格納する：

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

ONNX 形式の拡張仕様のため、そのまま通常の ONNX ファイルとしてロードできる。

ONNX ファイルは Protocol Buffers 形式で定義されており、ルートである `ModelProto` メッセージの `metadata_props` フィールドに、`StringStringEntryProto` のリストとしてメタデータを格納できる仕様となっている。

この仕様を活用し、AIVMX は `metadata_props` 内の以下のキーに、次の文字列データを格納する：

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

- [ONNX](https://onnx.ai/)
- [Open Neural Network Exchange Intermediate Representation (ONNX IR) Specification](https://onnx.ai/onnx/repo-docs/IR.html#optional-metadata)
- [ONNX Metadata](https://onnx.ai/onnx/repo-docs/MetadataProps.html)
- [How to populate onnx model with custom meta data map ?](https://github.com/onnx/sklearn-onnx/issues/214)

## AIVM Manifest Specification (Version 1.0)

以下に、AIVM / AIVMX ファイルフォーマットに含まれる、AIVM マニフェスト (Version 1.0) の仕様を示す。

**AIVM マニフェストには、マニフェストバージョン・モデルアーキテクチャ・モデル名・話者メタデータ・スタイル情報などの、音声合成モデルの利用に必要となる様々な情報が含まれる。**

AIVM マニフェストのデータ形式は、JSON フォーマットで記述された UTF-8 文字列である。  
JSON フォーマットの都合上、画像や音声データは Base64 エンコードされた文字列で格納される。

> [!NOTE]  
> 現在 AIVM マニフェストのコンテナフォーマットとして定義されている AIVM (Safetensors) ・AIVMX (ONNX) のメタデータ領域はネストなしの string 型から string 型への key-value でなければならないため、すべてのメタデータを文字列にシリアライズして格納する仕様となっている。  
> 画像や音声などのバイナリデータについては、Base64 エンコードを施した上で文字列として格納する。

### サポートされるモデルアーキテクチャ

- `Style-Bert-VITS2`
- `Style-Bert-VITS2 (JP-Extra)`

> [!IMPORTANT]  
> **AIVM / AIVMX ファイルをサポートするソフトウェアでは、自ソフトウェアではサポート対象外のモデルアーキテクチャの AIVM / AIVMX ファイルを、適切にバリデーションする必要がある。**  
> たとえば `Style-Bert-VITS2 (JP-Extra)` 以外のモデルアーキテクチャをサポートしないソフトウェアでは、`Style-Bert-VITS2` モデルアーキテクチャの AIVM / AIVMX ファイルのインストールを求められた際に「このモデルアーキテクチャには対応していません」とアラートを表示し、インストールを中止するよう実装すべき。

> [!IMPORTANT]  
> **技術的には上記以外のモデルアーキテクチャの音声合成モデルも格納可能だが、AIVM マニフェスト (Version 1.0) 仕様で公式に定義されているモデルアーキテクチャ文字列は上記のみ。**  
> 独自にモデルアーキテクチャ文字列を定義する場合は、既存のモデルアーキテクチャとの名前衝突や異なるソフト間での表記揺れが発生しないよう、細心の注意を払う必要がある。  
> なるべくこのリポジトリにプルリクエストを送信し、公式に AIVM 仕様に新しいモデルアーキテクチャのサポートを追加する形を取ることを推奨する。

### AIVM マニフェストのフィールド定義

以下は AIVM マニフェスト (Version 1.0) 仕様時点での AIVM マニフェストのフィールド定義を示す ([aivmlib の Pydantic スキーマ定義](aivmlib/schemas/aivm_manifest.py) より抜粋) 。

> [!IMPORTANT]  
> **AIVM マニフェスト内のフィールドは、今後 AIVM 仕様が更新された際に追加・拡張・削除される可能性がある。**  
> 今後のバージョン更新や追加のモデルアーキテクチャのサポートにより、AIVM マニフェストや AIVM / AIVMX ファイルフォーマット自体に新しいメタデータが追加されることも十分考えられる。  
> 現在有効な AIVM マニフェストバージョンは `1.0` のみ。

```python
class ModelArchitecture(StrEnum):
    StyleBertVITS2 = 'Style-Bert-VITS2'  # 対応言語: "ja", "en-US", "zh-CN"
    StyleBertVITS2JPExtra = 'Style-Bert-VITS2 (JP-Extra)'  # 対応言語: "ja"

class ModelFormat(StrEnum):
    Safetensors = 'Safetensors'
    ONNX = 'ONNX'

class AivmManifest(BaseModel):
    """ AIVM マニフェストのスキーマ """
    # AIVM マニフェストのバージョン (ex: 1.0)
    # 現在は 1.0 のみサポート
    manifest_version: Literal['1.0']
    # 音声合成モデルの名前 (最大 80 文字)
    # 音声合成モデル内の話者が 1 名の場合は話者名と同じ値を設定すべき
    name: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    # 音声合成モデルの簡潔な説明 (最大 140 文字 / 省略時は空文字列を設定)
    description: Annotated[str, StringConstraints(max_length=140)] = ''
    # 音声合成モデルの制作者名のリスト (省略時は空リストを設定)
    # 制作者名には npm package.json の "author", "contributors" に指定できるものと同じ書式を利用できる
    # 例: ["John Doe", "Jane Doe <jane.doe@example.com>", "John Doe <john.doe@example.com> (https://example.com)"]
    creators: list[Annotated[str, StringConstraints(min_length=1, max_length=255)]] = []
    # 音声合成モデルのライセンス情報 (Markdown 形式またはプレーンテキスト / 省略時は None を設定)
    # AIVM 仕様に対応するソフトでライセンス情報を表示できるよう、Markdown 形式またはプレーンテキストでライセンスの全文を設定する想定
    # 社内のみでの利用など、この音声合成モデルの公開・配布を行わない場合は None を設定する
    license: Annotated[str, StringConstraints(min_length=1)] | None = None
    # 音声合成モデルのアーキテクチャ (音声合成技術の種類)
    model_architecture: ModelArchitecture
    # 音声合成モデルのモデル形式 (Safetensors または ONNX)
    # AIVM ファイル (.aivm) のモデル形式は Safetensors 、AIVMX ファイル (.aivmx) のモデル形式は ONNX である
    model_format: ModelFormat
    # 音声合成モデル学習時のエポック数 (省略時は None を設定)
    training_epochs: Annotated[int, Field(ge=0)] | None = None
    # 音声合成モデル学習時のステップ数 (省略時は None を設定)
    training_steps: Annotated[int, Field(ge=0)] | None = None
    # 音声合成モデルを一意に識別する UUID
    uuid: UUID
    # 音声合成モデルのバージョン (SemVer 2.0 準拠 / ex: 1.0.0)
    version: Annotated[str, StringConstraints(pattern=r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$')]
    # 音声合成モデルの話者情報 (最低 1 人以上の話者が必要)
    speakers: list[AivmManifestSpeaker]

class AivmManifestSpeaker(BaseModel):
    """ AIVM マニフェストの話者情報 """
    # 話者の名前 (最大 80 文字)
    # 音声合成モデル内の話者が 1 名の場合は音声合成モデル名と同じ値を設定すべき
    name: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    # 話者のアイコン画像 (Data URL)
    # 画像ファイル形式は 512×512 の JPEG (image/jpeg)・PNG (image/png) のいずれか (JPEG を推奨)
    icon: Annotated[str, StringConstraints(pattern=r'^data:image/(jpeg|png);base64,[A-Za-z0-9+/=]+$')]
    # 話者の対応言語のリスト (BCP 47 言語タグ)
    # 例: 日本語: "ja", アメリカ英語: "en-US", 標準中国語: "zh-CN"
    supported_languages: list[Annotated[str, StringConstraints(pattern=r'^[a-z]{2,3}(?:-[A-Z]{4})?(?:-(?:[A-Z]{2}|\d{3}))?(?:-(?:[A-Za-z0-9]{5,8}|\d[A-Za-z0-9]{3}))*(?:-[A-Za-z](?:-[A-Za-z0-9]{2,8})+)*(?:-x(?:-[A-Za-z0-9]{1,8})+)?$')]]
    # 話者を一意に識別する UUID
    uuid: UUID
    # 話者のローカル ID (この音声合成モデル内で話者を識別するための一意なローカル ID で、uuid とは異なる)
    local_id: Annotated[int, Field(ge=0)]
    # 話者のスタイル情報 (最低 1 つ以上のスタイルが必要)
    styles: list[AivmManifestSpeakerStyle]

class AivmManifestSpeakerStyle(BaseModel):
    """ AIVM マニフェストの話者スタイル情報 """
    # スタイルの名前 (最大 20 文字)
    name: Annotated[str, StringConstraints(min_length=1, max_length=20)]
    # スタイルのアイコン画像 (Data URL, 省略可能)
    # 省略時は話者のアイコン画像がスタイルのアイコン画像として使われる想定
    # 画像ファイル形式は 512×512 の JPEG (image/jpeg)・PNG (image/png) のいずれか (JPEG を推奨)
    icon: Annotated[str, StringConstraints(pattern=r'^data:image/(jpeg|png);base64,[A-Za-z0-9+/=]+$')] | None = None
    # スタイルの ID (この話者内でスタイルを識別するための一意なローカル ID で、uuid とは異なる)
    local_id: Annotated[int, Field(ge=0, le=31)]  # 最大 32 スタイルまでサポート
    # スタイルごとのボイスサンプル (省略時は空リストを設定)
    voice_samples: list[AivmManifestVoiceSample] = []

class AivmManifestVoiceSample(BaseModel):
    """ AIVM マニフェストのボイスサンプル情報 """
    # ボイスサンプルの音声ファイル (Data URL)
    # 音声ファイル形式は WAV (audio/wav, Codec: PCM 16bit)・M4A (audio/mp4, Codec: AAC-LC) のいずれか (M4A を推奨)
    audio: Annotated[str, StringConstraints(pattern=r'^data:audio/(wav|mp4);base64,[A-Za-z0-9+/=]+$')]
    # ボイスサンプルの書き起こし文
    # 書き起こし文は音声ファイルでの発話内容と一致している必要がある
    transcript: Annotated[str, StringConstraints(min_length=1)]
```

## FAQ

### Q. AIVM と AIVMX という 2 つのフォーマットが定義されているのはなぜですか？

**A. 異なる用途や環境に最適化された 2 つのフォーマットを提供することで、より柔軟な利用を可能にするためです。**

- **AIVM (.aivm): PyTorch などの機械学習フレームワークで直接利用できる [Safetensors](https://huggingface.co/docs/safetensors/index) 形式をベースとしたフォーマットです。**
  - 研究開発やモデルの Fine-Tuning 、モデルマージによる新たな声質生成などに適しています。
  - 一般に、NVIDIA GPU (CUDA / TensorRT など) での高速推論に特化しています。
  - PyTorch には .pth (pickle) 形式もありますが、Python コードをそのままシリアライズする pickle の特性上、任意コードの実行が可能な脆弱性があります。したがって、AIVM 仕様では対応予定はありません。
- **AIVMX (.aivmx): 様々な環境で高速な推論が可能な [ONNX](https://onnx.ai/) 形式をベースとしたフォーマットです。**
  - 特に CPU での推論や、エッジデバイスでの利用に適しています。また Web ブラウザでも推論可能です。
  - 2024 年時点では、一般的な PC ユーザーの多くは NVIDIA GPU や NPU を搭載していない PC を使用しています。
    - ONNX 形式は CPU での推論性能に優れているため、GPU や NPU がなくても快適に音声合成を実行できます。
    - さらに ONNX 形式は DirectML 推論をサポートしており、Windows では AMD Radeon / Intel Arc GPU でも高速推論が可能です。
  - **AIVM 仕様対応ソフトウェアのリファレンス実装でもある [AivisSpeech](https://github.com/Aivis-Project/AivisSpeech) は AIVMX ファイルにのみ対応しています。**
    - PyTorch 依存を排除してインストールサイズを削減し、同時に CPU 推論速度を向上させるためです。

### Q. AIVM / AIVMX ファイルを既存のツールで読み込むことはできますか？

**A. はい、可能です。**

AIVM は Safetensors 形式の、AIVMX は ONNX 形式の拡張仕様として設計されているため、それぞれ通常の Safetensors ファイル・ONNX ファイルとして読み込むことができます。  
AIVM メタデータは既存のモデルフォーマット仕様で定められたメタデータ領域に格納されているため、既存のツールの動作に影響を与えることはありません。

### Q. 既存の AI 音声合成モデルを AIVM / AIVMX に変換するにはどうすればよいですか？

**A. 以下の2つの方法があります。**

1. **[AIVM Generator](https://aivm-generator.aivis-project.com/) (推奨)**: ブラウザ上の GUI で簡単に AIVM / AIVMX ファイルを生成・編集できます。
2. **aivmlib**: このライブラリが提供する CLI ツールを使用して、コマンドラインから最低限のメタデータが設定された AIVM / AIVMX ファイルを生成できます。
   - ハイパーパラメータなどから変換した最低限のメタデータのみ設定されているため、実際に配布する際は別途メタデータを編集する必要があります。

なお、変換元のモデルは単一の Safetensors または ONNX 形式で保存されている必要があります。

### Q. AIVM マニフェストのバージョン管理はどのように行われますか？

**A. AIVM マニフェストのバージョン管理は以下の方針で行われます。**

- **マイナーバージョンアップ (ex: 1.0 -> 1.1)**: 新しいフィールドの追加など、後方互換性が保たれた変更
- **メジャーバージョンアップ (ex: 1.1 -> 2.0)**: 既存フィールドの削除や構造変更など、後方互換性のない変更

現在は 1.0 が最新です。

### Q. aivmlib と [aivmlib-web](https://github.com/Aivis-Project/aivmlib-web) の違いはなんですか？

**A. aivmlib と aivmlib-web は、同じ AIVM 仕様を異なる言語/動作環境向けに実装したライブラリになります。**

- **aivmlib**: Python 実装。デスクトップアプリケーションやサーバーサイドでの利用を想定しています。
  - NVIDIA GPU 搭載の高火力サーバーで運用するケースでは、モデルアーキテクチャや推論環境次第ですが、AIVMX (ONNX) 形式よりも AIVM (Safetensors) 形式の方が高速な可能性もあります。
  - aivmlib は aivmlib-web のリファレンス実装でもあります。新しい仕様を実装する際は、まず aivmlib に実装してから、aivmlib-web に移植する形をとっています。
- **aivmlib-web**: TypeScript 実装。Web ブラウザ上での利用を想定しています。
  - [AIVM Generator](https://aivm-generator.aivis-project.com/) や Web ブラウザ上で音声合成を行うサービスで使うことを前提に設計・開発されています。
  - AIVM と AIVMX 両方のファイルを扱えます（主に AIVM Generator 向け）。
    - Web ブラウザで推論可能なモデルフォーマットは基本的に ONNX 形式に限られるため、実運用上は AIVMX ファイルのみ扱うケースがほとんどのはずです。
  - Python の `BinaryIO` が JavaScript Web API の `File` (Blob) になるなど、Web ブラウザの特性に応じた実装の違いはありますが、基本的な API 設計は aivmlib と同様です。
    - Node.js や Deno などのサーバーサイド JavaScript 環境への対応予定はありません。

> [!TIP]  
> 現時点では、aivmlib / [aivmlib-web](https://github.com/Aivis-Project/aivmlib-web) 以外に、公式にメンテナンスされる AIVM 仕様対応ライブラリはありません。  
> 今後、サードパーティーの他言語向けライブラリが登場する可能性はあります。

> [!IMPORTANT]  
> **新しいモデルアーキテクチャのサポートを追加する際は、aivmlib と [aivmlib-web](https://github.com/Aivis-Project/aivmlib-web) の両方に実装を追加する必要があります。**  
> AIVM Generator は aivmlib-web を使用しているため、エンドユーザーに新機能を提供するためには両方のライブラリをアップデートする必要があります。

### Q. 新しいモデルアーキテクチャのサポートを追加するにはどうすればよいですか？

**A. AIVM 仕様はモデルアーキテクチャの実装詳細を規定しないため、比較的容易に新しいモデルアーキテクチャを追加できます。**

1. **AIVM マニフェスト内のメタデータだけで対応できる場合**: `ModelArchitecture` に新しい種類（例: `GPT-SoVITS2`）を追加するプルリクエストを送信するだけで対応可能です。
   - その際、なるべく同時に `generate_aivm_metadata()` 関数に新しいモデルアーキテクチャのサポートを追加してください。
2. **モデルアーキテクチャ固有のメタデータの追加が必要な場合**: `aivm_style_vectors` フィールドのように、AIVM マニフェストとは別のメタデータキーを新設する仕様を策定した上で、プルリクエストを送信してください。
   - なるべく aivmlib (Python) と [aivmlib-web](https://github.com/Aivis-Project/aivmlib-web) (TypeScript Web) の両方でサポートできる仕様が望ましいです。
   - 当該モデルアーキテクチャが Web で推論可能な ONNX 形式に対応していない場合、推論時のみ必要なメタデータに関しては、aivmlib-web で対応可能な仕様である必要はありません。
   - AIVM マニフェストに追加するメタデータは、aivmlib-web でもサポートできるものでなければなりません（後述）。

> [!IMPORTANT]
> **提出される AIVM マニフェスト仕様は、技術的に aivmlib (Python) と [aivmlib-web](https://github.com/Aivis-Project/aivmlib-web) (TypeScript Web) の両方でサポートできるものでなければなりません。**  
> AIVM Generator の内部では aivmlib-web が利用されています。  
> **aivmlib にサポートを追加したら、aivmlib-web にも同様にサポートを追加してください。**

> [!NOTE]  
> **AIVM マニフェストは、モデルアーキテクチャに依存しない共通のメタデータのみを定義する設計としています。**  
> 実装固有のハイパーパラメータは、`aivm_hyper_parameters` フィールドに格納してください。  
> ハイパーパラメータの Pydantic スキーマ定義の追加も受け付けます。現在は `Style-Bert-VITS2` 系アーキテクチャのハイパーパラメータのスキーマのみが定義されています。

> [!NOTE]  
> 当然ですが、AIVM / AIVMX への変換元モデルは単一の Safetensors または ONNX 形式で保存されている必要があります。  
> **したがって、複数のモデルファイルにまたがるモデルアーキテクチャはサポートされていません。**  
> モデルファイルを一つに結合する、不要なモデルファイルを削るなどの対応をご検討ください。

### Q. ライセンス情報はどのように記述すべきですか？

**A. ライセンス情報は Markdown 形式またはプレーンテキストで、ライセンスの全文のコピーを AIVM / AIVMX ファイルに直接埋め込む形で設定します。**

URL 指定ではなくライセンス全文を埋め込む理由は以下の通りです。

- URL の永続性が保証できない
- URL だけだとライセンス名が分からない
- カスタムライセンスの規定が難しい
- AIVM 仕様対応ソフトウェアでライセンス情報を直接表示できる必要がある

### Q. 画像・音声データのサイズ制限はありますか？

**A. 具体的なサイズ制限は規定していませんが、一般にモデルファイル自体のファイルサイズが巨大なため、メタデータによるさらなるファイルサイズ増加は最小限に抑えるべきです。**

- 画像ファイル: 512×512 の JPEG または PNG (JPEG を推奨)
- 音声ファイル: WAV (PCM 16bit) または M4A (AAC-LC) (M4A を推奨)

> [!TIP]  
> リファレンス実装である [AIVM Generator](https://aivm-generator.aivis-project.com/) では、これらのガイドラインに従って適切なサイズ最適化を行っています。

### Q. メタデータは手動で編集できますか？

**A. メタデータは直接バイナリに埋め込まれるため、手動編集は推奨されません。**  
エンドユーザーの方は [AIVM Generator](https://aivm-generator.aivis-project.com/) をご利用ください。

> [!TIP]  
> **開発者は、aivmlib / [aivmlib-web](https://github.com/Aivis-Project/aivmlib-web) を使用して独自のアプリケーションを作成できます。**  
> aivmlib CLI は、最低限のメタデータを持つ AIVM / AIVMX ファイルの生成と、メタデータの確認機能のみを提供します。
