# EM-algorithm
EM-algorithmとは未知のデータの分布を混合ガウス分布で近似する際に使用される。 <br>
詳細は　https://qiita.com/kenmatsu4/items/59ea3e5dfa3d4c161efb　にまとまっているので参照をお勧めする。  <br>
理論自体ややこしいが、実装する内容は基本的に負担率の計算と負担率を用いたパラメータの更新である。  <br>

# DEMO

# Features

実装のポイントとしては二次元の場合、行列の計算が複雑となるため、テンソルを扱うことができるnp.einsumを使用していることである。  <br>
これにより、for文を用いることなく計算でき、計算時間を短縮することができる。  

# Requirement

"hoge"を動かすのに必要なライブラリなどを列挙する

* huga 3.5.2
* hogehuga 1.0.2

# Installation

EM-algorithmの説明をする

```bash
pip install huga_package
```

# 使い方

DEMOの実行方法など、"hoge"の基本的な使い方を説明する

```bash
git clone https://github.com/hoge/~
cd examples
python demo.py
```

# Note

注意点などがあれば書く

# Author

作成情報を列挙する

* 作成者
* 所属
* E-mail

# License
ライセンスを明示する

"hoge" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

社内向けなら社外秘であることを明示してる

"hoge" is Confidential.
