# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

中型GPTをトレーニング/ファインチューニングするための、最もシンプルで高速なリポジトリです。これは[minGPT](https://github.com/karpathy/minGPT)の書き換えで、教育よりも歯を優先しています。まだアクティブな開発中ですが、現在 `train.py` ファイルは OpenWebText 上で GPT-2 (124M) を再現しています。コード自体は平易で読みやすいです： train.py`は300行程度の訓練ループで、`model.py`は300行程度のGPTモデル定義です。以上です。

![repro124m](assets/gpt2_124M_loss.png)

コードはとてもシンプルなので、ニーズに合わせてハックしたり、ゼロから新しいモデルを訓練したり、事前に訓練したチェックポイントを微調整したりするのはとても簡単です（例えば、現在出発点として利用できる最大のものは、OpenAIのGPT-2 1.3Bモデルです）。

## インストール

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

依存関係

- pytorch](https://pytorch.org) <3
- numpy](https://numpy.org/install/) <3
- トランスフォーマー `transformers` for huggingface transformers <3 (GPT-2チェックポイントをロードする)
- datasets` for huggingface datasets <3 (OpenWebTextをダウンロード+前処理したい場合)
- tiktoken`（OpenAIの高速BPEコード用） <3
- wandb` オプションのロギング <3
- プログレスバー用の `tqdm` <3

## クイックスタート

もしあなたがディープラーニングの専門家ではなく、ただ魔法を感じて足を濡らしたいだけなら、シェイクスピアの作品で文字レベルのGPTを訓練するのが一番手っ取り早い方法です。まず、シェイクスピアを1つの（1MBの）ファイルとしてダウンロードし、生のテキストから1つの大きな整数のストリームに変える：

sh
python data/shakespeare_char/prepare.py
```

これで data ディレクトリに `train.bin` と `val.bin` が作成されます。いよいよGPTを学習させます。GPTのサイズはシステムの計算リソースに大きく依存します：

**私はGPUを持っています。素晴らしいことに、[config/train_shakespeare_char.py](config/train_shakespeare_char.py)設定ファイルで提供される設定で、すぐにベビーGPTを訓練することができます：

設定ファイル: ``sh
python train.py config/train_shakespeare_char.py
```

中身を覗いてみると、最大256文字のコンテキストサイズ、384の特徴チャンネルを持つGPTをトレーニングしており、各レイヤーに6つのヘッドを持つ6レイヤートランスフォーマーであることがわかります。1つのA100 GPUで、このトレーニングの実行には約3分かかり、最高の検証損失は1.4697です。設定に基づき、モデルのチェックポイントは `--out_dir` ディレクトリ `out-shakespeare-char` に書き込まれる。そのため、学習が終了したら、サンプリングスクリプトをこのディレクトリに向けることで、最適なモデルからサンプリングすることができます：

sh
python sample.py --out_dir=out-shakespeare-char
```

これでいくつかのサンプルが生成されます：

```
ANGELO：
そして臆病者は私のベッドに縛りつけられる、
そして私の脅威の門を突き刺す、
なぜなら、彼は逃げ去り、吊るされたからだ。
彼と一緒に

ヴィンセンティオ公爵
あなたの目に感謝します。

ヴィンチェンティオ公爵
それなら、マームを救うために彼に答えよう：
「暴君 "がこんなことを？

ヴィンチェンティオ公爵：
「悪の限りを尽くせば
その力を終わらせるために、庶民のために突き進むのだ。
私は去る、過度の好意と戦うために
薔薇男に急がせる
```

笑 `¯_(ツ)_/¯`. GPUで3分間トレーニングしたキャラクターレベルのモデルとしては悪くない。このデータセットで事前に訓練されたGPT-2モデルをファインチューニングすることで、より良い結果が得られる可能性が高い（後のファインチューニングのセクションを参照）。

**私はmacbook**（または他の安いコンピュータ）しか持っていません。大丈夫、GPTを訓練することはできる。最先端のPyTorchナイトリー（インストール時に[ここで選択](https://pytorch.org/get-started/locally/)）を入手することをお勧めします。PyTorchを使わなくても、単純な列車を走らせるだけなら次のようになります：

DeepL.com（無料版）で翻訳しました。






sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

ここでは、GPUではなくCPUで実行しているので、--device=cpu`と、--compile=False`でPyTorch 2.0のコンパイルをオフにする必要がある。コンテキストサイズは256文字から64文字になり、バッチサイズは64から12になります。また、より小さなTransformer（4層、4ヘッド、128エンベッディングサイズ）を使用し、反復回数を2000回に減らす（それに対応して、通常は学習レートを`--lr_decay_iters`でmax_iters程度まで減衰させる）。我々のネットワークは非常に小さいので、正則化も緩和する（--dropout=0.0`）。これでも約~3分で実行されるが、損失は1.88しかなく、したがってサンプルも悪くなるが、それでも楽しい：

しかし、それでも楽しい。
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
こんな感じでサンプルが生成されます：

```
gleorken vinghard iii：

「そして「」高くない場所で「」揚げ物をする
汝は汝を唸らせるために、その中にbechiveを求めた、
「汝は挫折しない「 」汝は挫折しない「 」汝は挫折しない
```

CPUで～3分、正しいキャラクターのゲシュタルトのヒントを得るには悪くない。もっと長く待ってもいいのであれば、ハイパーパラメーターを調整したり、ネットワークのサイズやコンテキストの長さ（`--block_size`）、トレーニングの長さなどを自由に増やしてほしい。

PyTorchはオンチップGPUを使うので、学習を*かなり*高速化(2-3倍)し、より大きなネットワークを使うことができます。詳しくは [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) を参照してください。

## GPT-2の再現

より本格的なディープラーニングの専門家は、GPT-2の結果を再現することに興味があるかもしれません。そこで、まずデータセットをトークン化します。この場合、[OpenWebText](https://openwebtext2.readthedocs.io/en/latest/) は、OpenAIの（非公開の）WebTextをオープンに複製したものです：

sh
python data/openwebtext/prepare.py
```

これは [OpenWebText](https://huggingface.co/datasets/openwebtext) データセットをダウンロードし、トークン化します。train.bin`とval.bin`が作成され、GPT2 BPEのトークンIDが1つのシーケンスに格納されます。これでトレーニングを開始する準備ができた。GPT-2 (124M)を再現するには、少なくとも8X A100 40GBノードが必要で、以下を実行する：

sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py を実行します。
```

これはPyTorchの分散データ並列(DDP)を使って約4日間実行され、~2.85の損失まで下がります。今、OWTで評価したGPT-2モデルは、約3.11のval損失を得ますが、それを微調整すると、（明らかなドメインギャップにより）～2.85の領域まで下がり、2つのモデルは～一致します。

クラスタ環境で複数のGPUノードに恵まれている場合、GPUを例えば2つのノードにまたがってブルブルさせることができる：

以下のようにします。
# IP 123.456.123.456を例に最初の（マスター）ノードで実行します：
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# ワーカーノードで実行
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

インターコネクトのベンチマーク(iperf3など)を行うことをお勧めします。特に、Infinibandを使用していない場合は、上記の起動の前に`NCCL_IB_DISABLE=1`を追加してください。マルチノードのトレーニングは動作しますが、ほとんどの場合_crawl_になります。デフォルトではチェックポイントは定期的に `--out_dir` に書き込まれる。単純に`python sample.py`でモデルからサンプリングできます。

最後に、単一のGPUで学習するには、`python train.py`スクリプトを実行します。スクリプトは非常に読みやすく、ハックしやすく、透過的であるように心がけています。スクリプトは非常に読みやすく、ハックしやすく、透明であろうとします。

## ベースライン

OpenAI GPT-2 のチェックポイントで、openwebtext のベースラインを取得することができます。以下のようにして数値を得ることができます：

以下のようにします。
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

を実行し、trainとvalで以下の損失を確認します：






| モデル｜パラメータ｜トレーニングロス｜損失｜値
| ------| ------ | ---------- | -------- |
| gpt2｜124M｜3.11｜3.12｜｜です。
| gpt2-medium｜350M｜2.85｜2.84｜｜です。
| gpt2 ラージ｜774M｜2.66｜2.67
| gpt2-xl｜1558M｜2.56｜2.54｜である。

しかし、GPT-2は（クローズドでリリースされていない）WebTextで学習されたものであり、OpenWebTextはこのデータセットをベストエフォートでオープンな形で再現したものに過ぎないことに注意しなければならない。これは、データセット・ドメインのギャップがあることを意味する。実際、GPT-2 (124M)のチェックポイントを取得し、しばらくOWTで直接ファインチューニングを行うと、損失は~2.85まで減少する。これは再現性に関してより適切なベースラインとなる。

## ファインチューニング

ファインチューニングはトレーニングと変わらないが、事前に学習させたモデルから初期化し、より小さな学習率でトレーニングすることを確認するだけである。新しいテキストで GPT を微調整する方法の例として、`data/shakespeare` に移動して `prepare.py` を実行し、小さな shakespeare データセットをダウンロードして `train.bin` と `val.bin` にレンダリングします。OpenWebTextとは異なり、これは数秒で実行されます。ファインチューニングにかかる時間は非常に短く、例えばシングルGPUでは数分です。次のような微調整の例を実行してみてください：

以下のように実行します。python train.py config/finetune_shakespeare.py```

これは`config/finetune_shakespeare.py`にある設定パラメータのオーバーライドをロードします（私はあまりチューニングしていませんが）。基本的には、`init_from`でGPT2のチェックポイントから初期化し、通常通り学習します。メモリ不足の場合は、モデルサイズを小さくするか（`{'gpt2'、'gpt2-medium'、'gpt2-large'、'gpt2-xl'}`）、`block_size`（コンテキストの長さ）を小さくする。最適なチェックポイント (検証の損失が最も少ない場所) は `out_dir` ディレクトリになり、デフォルトでは `out-shakespeare` になる。そして、`sample.py --out_dir=out-shakespeare` のコードを実行します：

```
THEODORE：
もし私が死んだら、
もし私が死んだら、最初の者にあなたを売る、
嘘をつけば、
もし私が嘘をつけば、私は3番目に汝を売る、
汝を売るか、買うか、
汝に告げよう。
私の所有物を売ってはならない。

ジュリエット：
もし盗むなら、汝自身を売るな。

セオドア：
私は盗みません。盗品を売ります。

セオドア：
汝は何を売っているのか知らない、汝は女である、
汝は常に犠牲者であり、何の価値もないものである：
汝には何の権利もない、何の権利もない、ただ売られるだけだ。
```

おっと、GPT、あそこは暗い場所だ。コンフィグのハイパーパラメーターはあまり調整しなかった！





## サンプリング/推論

sample.py`スクリプトを使って、OpenAIが公開している学習済みのGPT-2モデルからサンプリングするか、自分で学習したモデルからサンプリングします。例えば、利用可能な最大の `gpt2-xl` モデルからサンプリングする方法を示します：

shpython sample.py    --init_from=gpt2-xl ˶="What answer?
    --start=「What is the answer to life, the universe, and everything?」 生命、宇宙、そして全てに対する答えは何ですか？\
    --num_samples=5 --max_new_tokens=100
```トレーニングしたモデルからサンプリングしたい場合は、`--out_dir` を使ってコードを適切に指定する。また、例えば ``python sample.py --start=FILE:prompt.txt`` のように、ファイルにあるテキストをモデルに入力することもできます。

## 効率化に関するメモ

単純なモデルのベンチマークやプロファイリングには `bench.py` が便利です。train.py`の学習ループで行われることと同じですが、他の複雑な処理は省かれています。

このコードはデフォルトで[PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/)を使用していることに注意してください。執筆時点(2022年12月29日)では、ナイトリーリリースで `torch.compile()` が利用できるようになっています。この1行のコードによる改善は顕著で、例えば繰り返し処理時間が ~250ms / iter から 135ms / iter に短縮されました。PyTorch チームはよくやった！## todos

- DDPの代わりにFSDPを調査して追加する。
- 標準的な検証(例えばLAMBADA? HELM?)でゼロショットの当惑度を評価する。
- ファインチューニングスクリプトを微調整する。
- 訓練中にバッチサイズが直線的に増加するようにスケジュールを組む
- 他のエンベッディング（ロータリー、アリバイ）を組み込む。- チェックポイントで、最適化バッファとモデルパラメータを分離する。- ネットワークの健全性に関する追加ロギング（勾配クリップイベント、マグニチュードなど）- より良いinitなどに関する調査をもう少し。

## トラブルシューティング

デフォルトでは PyTorch 2.0 (`torch.compile`) を使っていることに注意してください。これはかなり新しく実験的なもので、まだすべてのプラットフォーム(例えばWindows)で利用できるわけではありません。もし関連するエラーメッセージが表示される場合は、`--compile=False`フラグを追加してこれを無効にしてみてください。これはコードを遅くしますが、少なくとも実行はできます。このリポジトリ、GPT、言語モデリングに関するいくつかのコンテキストについては、私の[Zero To Heroシリーズ](https://karpathy.ai/zero-to-hero.html)を見ると役に立つかもしれません。特に、[GPTのビデオ](https://www.youtube.com/watch?v=kCc8FmEb1nY)は、言語モデリングの文脈を事前に知っていれば、人気があります。その他の質問や議論については、Discordの**#nanoGPT**までお気軽にどうぞ：

[!](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)## 謝辞

すべてのnanoGPT実験は、私のお気に入りのクラウドGPUプロバイダーである[Lambda labs](https://lambdalabs.com)のGPUで動いています。nanoGPTのスポンサーになってくれたLambda labsに感謝します！

