# gaussian_saliency_map
本プログラムの内容は既存の顕著性マップをガウス分布を用いて近似することである<br>
顕著性マップを近似する理由は推定に用いる場合，顕著性マップのパラメータ数が非常に多いことがあげられる．<br>
顕著性マップを近似することでパラメータ数が1920×1080から60に減少する．<br>

# DEMO
以下に示すものは顕著性マップ(左)と混合ガウス分布を用いた顕著性マップの近似結果(右)である．<br>
![SaliencyMapExample01](https://github.com/tempolin/forme/assets/168509729/a9486594-9611-419d-aa3c-92263abc3a70)
![GaussianSaliencyMapExample01](https://github.com/tempolin/forme/assets/168509729/5644359d-ad6d-4ffd-bae0-c94ef5bb77e8)<br>
動画の顕著性が存在する領域に対して大幅に情報を失うことなく，近似を行えているといえる．<br>

# Features
動的顕著性マップ：動画のうち注視しやすい領域をカラーマップなどの形で表したもの<br>
通常、機械学習などを用いて作成される。<br>
提案手法では "Droste et al(2020)"の手法を用いて作成した<br>
URL：https://github.com/rdroste/unisal<br>
"hoge"のセールスポイントや差別化などを説明する

# Requirement

"hoge"を動かすのに必要なライブラリなどを列挙する

* huga 3.5.2
* hogehuga 1.0.2

# Installation

Requirementで列挙したライブラリなどのインストール方法を説明する

```bash
pip install huga_package
```

# Usage

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
講義動画のURL<br>
動画の作成者に資料等で使用する許可は得ているものの、本編の二次配布はできないため以下のリンクを参照<br>
フラクタルとは：https://www.youtube.com/watch?v=fnofGm_IHUw


"hoge" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

社内向けなら社外秘であることを明示してる

"hoge" is Confidential.


