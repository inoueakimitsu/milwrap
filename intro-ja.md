# カウント ベース仮定下での Multiple Instance Learning で scikit-learn のモデルを使用する方法

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inoueakimitsu/milwrap/blob/master/milwrap-experiment.ipynb)

- [カウント ベース仮定下での Multiple Instance Learning で scikit-learn のモデルを使用する方法](#カウント-ベース仮定下での-multiple-instance-learning-で-scikit-learn-のモデルを使用する方法)
  - [概要](#概要)
  - [Multiple Instance Learning の問題設定と先行研究](#multiple-instance-learning-の問題設定と先行研究)
    - [Multiple Instance Learning の概要](#multiple-instance-learning-の概要)
    - [通常の教師あり学習と MIL のラベル形式の違い](#通常の教師あり学習と-mil-のラベル形式の違い)
      - [応用分野](#応用分野)
    - [Multiple Instance Learning に対する2つの視点](#multiple-instance-learning-に対する2つの視点)
      - [正解ラベルの粒度](#正解ラベルの粒度)
        - [Presence Assumption](#presence-assumption)
        - [Threshold Assumption](#threshold-assumption)
        - [Count-based Assumption](#count-based-assumption)
        - [異なる粒度間の互換性](#異なる粒度間の互換性)
      - [インスタンスの性質と正しいラベルの性質の関係に基づく視点](#インスタンスの性質と正しいラベルの性質の関係に基づく視点)
  - [milwrap の紹介](#milwrap-の紹介)
    - [MIL法の概要](#mil法の概要)
      - [MILの手法](#milの手法)
      - [SIL（Single Instance Learning）アプローチ](#silsingle-instance-learningアプローチ)
      - [MIL に特化したアプローチ](#mil-に特化したアプローチ)
    - [mi-SVM](#mi-svm)
      - [アルゴリズム](#アルゴリズム)
      - [mi-SVM の特徴](#mi-svm-の特徴)
    - [milwrap](#milwrap)
      - [SVM 以外の学習デバイスの利用](#svm-以外の学習デバイスの利用)
      - [多クラス分類への対応](#多クラス分類への対応)
      - [Multiple Instance Learningの追加仮定をサポート](#multiple-instance-learningの追加仮定をサポート)
  - [精度評価実験](#精度評価実験)
    - [方法](#方法)
      - [既存手法](#既存手法)
      - [提案手法](#提案手法)
      - [オラクル](#オラクル)
      - [ハイパー パラメータ](#ハイパー-パラメータ)
    - [データセット](#データセット)
    - [結果](#結果)
    - [考察](#考察)
  - [milwrap の使い方](#milwrap-の使い方)
    - [インストール](#インストール)
    - [API](#api)
  - [参考文献](#参考文献)

## 概要

本記事では、Multiple Instance Learning（MIL）の概要と、MIL のための Python ライブラリ milwrap を紹介します。

milwrap を利用するメリットは 3 つあります。

- マルチクラス分類をサポートしています。
- カウント ベースの仮定をサポートをしています。
- scikit-learn スタイルの API を持つ任意の教師あり学習モデルを MIL のフレームワークに適応させることができます。

本レポートの実験では、以下のことを示しました。

- milwrap を用いた場合、条件によっては Multiple Instance Learning でも Single Instance Learning と同程度の精度を得ることが可能です。
- データの粒度をできるだけ落とさず、Count-based Assumption に従うデータの情報を利用することで、より高い精度が得られる。

本論文の構成は以下の通りです。

- Multiple Instance Learning の問題設定と先行研究
- milwrap アルゴリズム
- 精度評価実験
- milwrap ライブラリの使用方法

## Multiple Instance Learning の問題設定と先行研究

### Multiple Instance Learning の概要

### 通常の教師あり学習と MIL のラベル形式の違い

Multiple Instance Learning は弱教師付き学習タスクです。
簡単に言えば、MIL はサンプルごとに正しいラベルが与えられず、サンプル群ごとに正しいラベルが与えられるタスクです。
例えば、サンプル 1、2、3 には少なくとも 1 つの正例が含まれるが、サンプル 4、5、6、7 には負例しか含まれないというような、粗視化された情報しか与えられない状況で学習する必要があります。


#### 応用分野

Herrera, Francisco, et al. (2016) の 2.4 項によると、我々は以下の場面で MIL を利用できます。

- 代替的な表現
  - 同じオブジェクトの異なる見方，見え方，説明がある場合
  - 例えば、薬物活性予測で使用されます。
- 複合的なオブジェクト
  - 複数の部品からなる複合的なオブジェクトに対して使用します。
  - 画像分類タスクの例では、「イギリスの朝食」というタイトルの写真に、イギリスの食事の要素に加えて、様々なオブジェクトが含まれている。
- 変化するオブジェクト
  - 時間の経過とともに何度もサンプリングされるオブジェクトがある場合
  - 例えば、倒産の予測などに利用されます。

同資料では、以下の応用分野が挙げられています。

- バイオインフォマティクス
  - 薬物活性予測
    - 化合物分子の変異原性予測
    - 抗がん剤としての分子の活性予測
  - タンパク質同定
    - チオレドキシン フォールド タンパク質の認識
    - カルモジュリン タンパク質の結合タンパク質の同定
    - 遺伝子発現パターン アノテーション
- 画像の分類と検索
  - このタスクは、部分画像を含む画像を扱います。
  - 画像分類
  - 顔認識
  - 画像検索
- ウェブマイニングとテキスト分類
  - ウェブマイニング（バッグ＝インデックスページ、インスタンス＝そのページがリンクしている他の Web サイト）
  - 文書分類 (バッグ = 文書、インスタンス = 通過)
- 物体検出・追跡
  - 馬の検出
  - 歩行者検知
  - レーダー画像に基づく地雷の検出
- 医療診断・画像処理
  - 腫瘍検出
  - 心電図記録による心筋梗塞の検出
  - センサデータを用いた高齢者の虚弱・認知症検出
  - 映像分類を用いた大腸ポリープの異常増殖の検出
- その他の分類応用
  - 生徒の成績予測（バッグ＝生徒、インスタンス＝生徒が行った課題）
  - 強化学習におけるサブゴールの自動発見（バッグ＝エージェントの軌跡、インスタンス＝軌跡を観測したもの）
    - ラベルは軌道が成功したか失敗したかを示します。
  - 銘柄選択問題

このように、MIL の応用は多岐に渡ります。

### Multiple Instance Learning に対する2つの視点

MIL をもう少し詳しく見てみましょう。
MIL は 2 つの観点から詳細化することができます。

- 正解ラベルの粒度
- インスタンスとラベルの関係

#### 正解ラベルの粒度

これは Weidmann ら (2003) が提示したフレームワークです。
MIL は正解ラベルの粒度により、粒度の粗い順に Presence Assumption、Threshold Assumption、Count-based Assumption に分類されます。


##### Presence Assumption

各バッグが 1 つ以上の正例を含むかどうかをラベル付けする場合です。
多くの MIL の文献ではこの粒度のタスクを扱っています。

##### Threshold Assumption

各バッグが N 個以上のポジティブな例のインスタンスを含むかどうかラベル付けされる場合です。

##### Count-based Assumption

各バッグが下限 L、上限 U の正例のインスタンスを含むかどうかラベル付けされるケースです。
本記事で紹介した milwrap を適用することで、この仮定に従うデータを扱うことができます。

##### 異なる粒度間の互換性

粗視化されたタスクは、細粒化されたタスクの表現に書き換えることができます。したがって、細粒度のタスクを扱える技術は、粗粒度のデータを持つタスクも扱うことができます。例えば、Presence Assumption に従うデータを持つタスクは、count-based Assumption タスクを扱える技術でカバーすることができます。逆は真ではありません。

#### インスタンスの性質と正しいラベルの性質の関係に基づく視点

この記事では深く議論しませんが、観測できないインスタンスごとのラベルを所与としたときの、バッグのラベルを生成する確率的なプロセスを考える視点があります。

例えば、バッグの中で正例の数が多ければ多いほど、袋のラベルが正になる確率が高くなるというモデルがあります。

このような観点を集合的仮定（Collective Assumption）と呼びます。

milwrap はまだ集合的仮定をサポートしていません。

## milwrap の紹介

ここまで、MILの 概要とタスクの下位分類について解説を行いました。

ここでは、MIL のタスクを解くために便利に使える milwrap ライブラリについて説明します。

まず、MIL を行う手法の概説を行います。
次に、milwrap のベースとなっている mi-SVM について説明します。
そして、milwrap の mi-SVM への拡張内容を説明します。

### MIL法の概要

#### MILの手法

MIL には、推論時にインスタンス単位での分類を可能とするアプローチと、バッグ単位での分類のみが可能なアプローチの 2 つがあります。

この視点は重要で、たとえば、薬理活性を予測するためには、インスタンス単位で予測することが重要です。
いっぽう、画像分類のようにバッグ単位での予測のみで問題ない場合もあります。

なお、これらのアプローチは明確に区別できるものではありません。例えば、深層学習系の手法には、大局的にはバッグ単位の判断しかできないが、Attention の仕組みによってインスタンス単位の結果を得ることができるため、2 つのアプローチの中間的なアプローチと見ることができます。
インスタンス単位の予測は本当に必要なのか、必ず検討しましょう。もし、インスタンス単位の予測がバッグ単位の予測の解釈可能性を高めるためだけに必要なのであれば、モデル化の自由度が大きい上記の Collective Assumption をサポートした手法のほうがより柔軟で適している可能性があります。

#### SIL（Single Instance Learning）アプローチ

MIL のアルゴリズムの一つとして、バッグごとのラベルをそのバッグに属する全てのインスタンスのラベルとして割り当てる SIL (Single Instance Learning) があります。バッグには否定的な例が多く含まれるため、ラベルが間違っていることが想定されますが、意外とこれで許容されるケースも少なくありません。

ぜひ、SIL を基準として、誤りを許容するアプローチを検討してください。


#### MIL に特化したアプローチ

MIL タスクを解決するために多くの手法が提案されています。
近年では、新しい手法も多く提案されています。

例えば．

- Iterative Discrimination
- Diverse Density

詳しくは参考文献 2 をご覧ください。

### mi-SVM

mi-SVM は MIL に特化した手法の一つで、SVM を Presence Assumption の 2 値クラス分類タスクに適用したものです。

#### アルゴリズム

mi-SVM の疑似コードを以下に引用する。

```
initialize y_i = Y_I for i \in I
REPEAT
  compute SVM solution w, b for data set with  imputed labels
  compute outputs f_i ~ <w, x_i> + b for all x_i in positive bags
  set y_i = sgn(f_i) for every i _in I, Y_I = 1
  FOR (every positive bg B_I)
    IF (\sum_{i \in I}(1 + y_i)/2 == 0)
      compute i* = argmax_{i \in I} f_i
      set y_{i*} = 1
    END
  END
WHILE (imputed labels have changed)
OUTPUT (w, b)
```

mi-SVM は、擬似ラベルを生成し、その擬似ラベルから学習し、SVM の推定出力スコアを用いて擬似ラベルを更新するアプローチです。

#### mi-SVM の特徴

ユーザー目線でみた mi-SVM の利点は以下の通りです。

- 推論時は通常の SIL の SVM となり、実装が容易です。
- Presence を仮定した場合の精度が良いです。

課題としては、以下が挙げられます。

- SVM をベースにしているため、サンプル数が多いデータには不向きです。
  - SIL タスクではサブ サンプリングで十分ですが、MIL の設定ではサブ サンプリングの方法が自明ではありません（バッグごとにサブ サンプリングを行うか、インスタンスごとに行うか決定する必要がある）
- Threshold Assumption、Count-based Assumptionはサポートしていません。

### milwrap

#### SVM 以外の学習デバイスの利用

milwrapでは、上記の mi-SVM アルゴリズムで SVM が使われている部分を、SVM 以外の予測スコアを計算できる手法に置き換えます。

例えば、決定木、ロジスティック回帰、ラッソ回帰、深層学習などが利用できます。

SVM では、大量のデータを使用する場合、グラム行列が大きくなってしまいます。
(milwrap 開発の動機は、scikit-learn の SGDClassifier とカーネル近似を利用したいというところにありました)

milwrap Version 0.1.3 では、scikit-learn の教師あり学習器 API を使った学習器を使うことができます。

ちなみに scikit-learn のように PyTorch モデルをラップできる skorch を使用すると、PyTorch のモデルを MIL に対応させることができます。

推定時には、milwrap は不要です。
モデルの形式は純粋な scikit-learn モデル（または PyTorch のモデル）となります。

#### 多クラス分類への対応

mi-SVM は二値分類にしか対応していなかったので、多値分類ができるようにアルゴリズムを開発しました。

ラベルを修正する場合、修正が必要なラベルをどのクラスに割り当てるかが重要なポイントになります。
実際には、これはタスクに特化したものと考えるべきです。

#### Multiple Instance Learningの追加仮定をサポート

milwrap は、最も細かい仮定であるカウント ベースの仮定をサポートしています。
milwrap で学習する場合、各バッグには各クラスの最小数と最大数の範囲設定が与えられます。
これらの範囲外のインスタンス数が最小になるように学習します。

## 精度評価実験

ここでは、既存手法と提案手法の性能を比較し、milwrap の拡張が精度向上に寄与しているかどうかを確認します。
本実験では、多クラス分類かつカウント ベース仮定のタスクのデータを用いることで、
多クラス分類部の拡張と、カウント ベース仮定の拡張の両方の有効性を確認します。
また、インスタンス単位のラベルが得られるケースと比較し、MIL でどの程度の性能が得られるのかを確認し、MIL の応用可能性を確認します。

### 方法

#### 既存手法

既存手法は mi-SVM です。
mi-SVM は、2 値クラス分類と Presence Assumption にのみ対応しています。
そのため，各クラスに対して 2 値クラス分類器をそれぞれ学習させます。
Presence Assumption に対応するため、ラベルの粒度を落とし、バッグ内に正例が 1 個以上あるかどうかの情報のみを利用します。

#### 提案手法

提案手法では、学習器として milwrap と SVM を用います。
提案手法の 2 つのバリエーションを用いて、多クラス化が有効かどうかを確認しました。
マルチクラス分類アルゴリズムは `milwrap-multiclass` と呼び、milwrap OVR 戦略は `milwrap-ovr` と呼びます。

#### オラクル

比較のために、シングル インスタンス SVM を用いて、インスタンス毎のラベルが入手可能な場合の学習を行います。これをオラクル (Oracle) と呼びます。


#### ハイパー パラメータ

モデルのハイパー パラメータはクロス バリデーションによって選択します。
ハイパー パラメータの候補は以下の通りです。

- C: 0.01, 0.1, 1.0, 10.0
- gamma: 0.01, 0.1, 1.0, 10.0

### データセット

データは以下のように人工的に生成します。

1. 多変量一様分布から、各クラスの各特徴の平均ベクトルを生成します。クラス数は 5、次元数は 7、最小値は 0、最大値は 1 です。
2. 各バッグのインスタンス総数を離散一様分布で生成します。バッグの数はトレーニング用 100 個、テスト用 100 個とします。各バッグには最小で 1 個、最大で 100 個のインスタンスが与えられます。
3. 各バッグの各クラスについて、ゼロ過剰 Dirichlet 分布で比率を生成します。0.2, 0.5, 0.7, 1.0 の各 `prob_of_non_zero` の場合、各クラスは 1 - `prob_of_non_zero` の確率で構成比が 0 になると仮定します。構成比が 0 でないクラスについては、α=1.0 の Dirichlet 分布で構成比を決めます。
4. 多項分布で、各インスタンスについて対応するクラスを決定します。なお、`prob_of_non_zero` が 1.0 であっても、多項分布に割り当てられるインスタンス数は 0 であってもよい点に注意してください。
5. 各粒子の特徴量は等方的な多変量正規分布で生成されます。各特徴の標準偏差は 0.5 とします。
6. バッグの正解ラベルは、あらかじめ設定された適応的な幅の区間（区間は `[0, 5, 10, 20, 50, 100]`）でカウント ベースの仮定に従って生成されます。

### 結果

以下の表は、各手法（`milwrap-multiclass`, `milwrap-ovr`, `misvm`, `oracle`）の最適なハイパー パラメータ設定の RMSE を表しています。

| prob_of_non_zero | milwrap-multiclass | milwrap-ovr | misvm-ovr | oracle |
| :--------------- | :----------------- | :---------- | :-------- | :----- |
| 20%              | 8.33               | 8.90        | 8.72      | 8.50   |
| 50%              | 7.88               | 8.30        | 8.11      | 8.07   |
| 70%              | 5.63               | 8.65        | 22.48     | 5.47   |
| 100%             | 4.17               | 5.92        | 20.77     | 4.08   |

全体として、`milwrap-multiclass` は `milwrap-ovr` や `misvm-ovr` よりも優れています。
特に `prob_of_non_zero` が高いほど、 `milwrap-multiclass` の方が `misvm-ovr` よりも相対的に有利になります。

これは、`prob_of_non_zero` が高い場合、Presence Assumption の情報量が減少し、Count-based Assumption をサポートする milwrap の優位性が増すためです。

`milwrap-multiclass` は `milwrap-ovr` よりも正確です。

`milwrap-multiclass` は `oracle` にかなり近いです。`prob_of_non_misvm` の 50% までは、 `misvm-ovr` はオラクルとほぼ同じ精度です。この事実は、misvm のアルゴリズムが有効に機能していることを表しています。

`prob_of_non_zero` が 50% 以上の状況では、 `misvm-ovr` はより困難であり、このような状況では `milwrap-multiclass` を使用する必要があります。

### 考察

本実験では、多クラス、カウント ベース仮定の Multiple Instance Learning タスクにおいて、既存手法と提案手法の性能を比較し、mi-SVM の milwrap への拡張が推定精度の向上に寄与することを明らかにしました。この実験により、以下のことを示しました。

- milwrap を用いた場合、条件によっては Multiple Instance Learning でも Single Instance Learning と同程度の精度を得ることが可能です。
- データの粒度をできるだけ落とさず、Count-based Assumption に従うデータの情報を利用することで、より高い精度が得られます。

milwrap の弱点は、milwrap の学習が通常の学習と比較して計算量が多く、時間がかかることです。この問題を解決するために、学習効率向上の観点から、初期段階では高速だが精度の低い学習器を用い、後半では低速だが精度の高い学習器に切り替える方法が考えられます。これらの方向性での開発は今後の課題です。


## milwrap の使い方

### インストール

以下のコマンドを実行します。

```shell
pip install milwrap
```

### API

シングル インスタンスの教師あり学習アルゴリズムを用意します。
`predict_proba()` メソッドをもつクラスを使用して下さい。
例えば、ロジスティック回帰を利用する場合は以下のようになります。

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
```

学習器を `MilCountBasedMultiClassLearner` でラップします。

```python
from milwrap import MilCountBasedMultiClassLearner 
mil_learner = MilCountBasedMultiClassLearner(clf)
```

以下のフォーマットでデータを用意します。

```python
# 以下のデータセットを用意します。
#
# - バッグ ... np.ndarray のリスト
# (num_instance_in_the_bag * num_features) のリスト。
# - 閾値の下限 ... np.ndarray (num_bags * num_classes)
# - 閾値の上限 ... np.ndarray (num_bags * num_classes)
#
# bags[i_bag] は lower_threshold[i_bag, i_class] よりも少ない数のインスタンスを含んでいます。
# i_class はインスタンスに対応します。
```

以下のフォーマットでデータを用意します。:

```python
# Multiple Instance Learning を実行します。
clf_mil, y_mil = learner.fit(
    bags,
    lower_threshold,
    upper_threshold,
    n_classes,
    max_iter=10)
```

学習が完了すると、通常のシングル インスタンス学習済みモデルとして利用することができます。

推論には `milwrap` ライブラリは不要です。


```python
# Multiple Instance Learning 後、インスタンスラベルを予測することができる。
clf_mil.predict([instance_feature])
```

## 参考文献

1. Andrews, Stuart, Ioannis Tsochantaridis, and Thomas Hofmann. "Support vector machines for multiple-instance learning." Advances in neural information processing systems 15 (2002). https://proceedings.neurips.cc/paper/2002/file/3e6260b81898beacda3d16db379ed329-Paper.pdf
2. Herrera, Francisco et al. “Multiple Instance Learning - Foundations and Algorithms.” (2016). http://www.amazon.co.jp/dp/3319477587
3. Weidmann, Nils, Eibe Frank, and Bernhard Pfahringer. "A two-level learning method for generalized multi-instance problems." European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2003. https://link.springer.com/content/pdf/10.1007/978-3-540-39857-8_42.pdf
