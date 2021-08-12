import pandas as pd
import numpy as np
# データ可視化のライブラリ
import matplotlib.pyplot as plt
# 機械学習ライブラリ
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# graphvizのインポート
import graphviz
#grid searchとcross validation用
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def probability_command(file_name, input_date, input_max_depth, input_ratio):
    """
    TODO input_max_depthの使用
    """
    # CSVファイルの読み込み。
    df = pd.read_csv(file_name)

    # 翌日終値 - 当日終値で差分を計算
    #shift(-1)でCloseを上に1つずらす
    df['Close+1'] = df.Close.shift(-1)
    df['diff'] = df['Close+1'] - df['Close']
    #最終日はClose+1がNaNになるので削る
    df = df[:-1]

    # 上昇と下降のデータ割合を確認
    m = len(df['Close'])

    df.rename(columns={"diff" : "Target"}, inplace=True)

    # 不要なカラムを削除
    del df['Close+1']
    del df['Date']
    # カラムの並び替え
    df = df[['Target', 'Open', 'High', 'Low', 'Close']]
    # 前日より大きくなったいたら1,小さくなっていたら0
    df.loc[df['Target'] > 0, "Target"] = 1
    df.loc[df['Target'] < 0, "Target"] = 0

    #移動平均の計算、5日、25日、50日、75日
    #ついでにstdも計算する。（=ボリンジャーバンドと同等の情報を持ってる）
    #75日分のデータ確保
    for i in range(1, 75):
        df['Close-'+str(i)] = df.Close.shift(+i)
    #移動平均の値とstdを計算する, skipnaの設定で一つでもNanがあるやつはNanを返すようにする
    nclose = 5
    df['MA5'] = df.iloc[:, np.arange(nclose, nclose+5)].mean(axis='columns', skipna=False)
    df['MA25'] = df.iloc[:, np.arange(nclose, nclose+25)].mean(axis='columns', skipna=False)
    df['MA50'] = df.iloc[:, np.arange(nclose, nclose+50)].mean(axis='columns', skipna=False)
    df['MA75'] = df.iloc[:, np.arange(nclose, nclose+75)].mean(axis='columns', skipna=False)

    df['STD5'] = df.iloc[:, np.arange(nclose, nclose+5)].std(axis='columns', skipna=False)
    df['STD25'] = df.iloc[:, np.arange(nclose, nclose+25)].std(axis='columns', skipna=False)
    df['STD50'] = df.iloc[:, np.arange(nclose, nclose+50)].std(axis='columns', skipna=False)
    df['STD75'] = df.iloc[:, np.arange(nclose, nclose+75)].std(axis='columns', skipna=False)
    #計算終わったら余分な列は削除
    for i in range(1, 75):
        del df['Close-'+str(i)]
    #それぞれの平均線の前日からの変化（移動平均線が上向か、下向きかわかる）
    #shift(-1)でCloseを上に1つずらす
    df['diff_MA5'] = df['MA5'] - df.MA5.shift(1)
    df['diff_MA25'] = df['MA25'] - df.MA25.shift(1)
    df['diff_MA50'] = df['MA50'] - df.MA50.shift(1)
    df['diff_MA75'] = df['MA50'] - df.MA50.shift(1)
    #3日前までのOpen, Close, High, Lowも素性に加えたい
    for i in range(1, 4):
        df['Close-'+str(i)] = df.Close.shift(+i)
        df['Open-'+str(i)] = df.Open.shift(+i)
        df['High-'+str(i)] = df.High.shift(+i)
        df['Low-'+str(i)] = df.Low.shift(+i)
    #NaNを含む行を削除
    df = df.dropna()
    #何日分使うか決める
    nday = int(input_date)
    df = df[-nday:]

    n = df.shape[0]
    p = df.shape[1]
    # 訓練データとテストデータへ分割。シャッフルはしない
    train_start = 0
    ratio = float(int(input_ratio)/100)
    train_end = int(np.floor(ratio*n))
    test_start = train_end + 1
    test_end = n
    data_train = np.arange(train_start, train_end)
    data_train = df.iloc[np.arange(train_start, train_end), :]
    data_test = df.iloc[np.arange(test_start, test_end), :]

    #targetを分離
    X_train = data_train.iloc[:, 1:]
    y_train = data_train.iloc[:, 0]
    X_test = data_test.iloc[:, 1:]
    y_test = data_test.iloc[:, 0]
    # 決定技モデルの訓練
    clf_2 = DecisionTreeClassifier(max_depth=5)

    # grid searchでmax_depthの最適なパラメータを決める
    #k=10のk分割交差検証も行う
    params = {'max_depth': [2, 5, 10, 20]}

    grid = GridSearchCV(estimator=clf_2,
                        param_grid=params,
                        cv=10,
                        scoring='roc_auc')
    grid.fit(X_train, y_train)

    res_text = ''
    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        res_text += "%0.3f +/- %0.2f %r <br>" % (
            grid.cv_results_['mean_test_score'][r],
            grid.cv_results_['std_test_score'][r] / 2.0,
            grid.cv_results_['params'][r]
        )
    res_text += 'Best parameters: %s <br>' % grid.best_params_
    res_text += 'Accuracy: %.2f <br>' % grid.best_score_

    #grid searchで最適だったパラメータを使って学習する
    clf_2 = grid.best_estimator_
    clf_2 = clf_2.fit(X_train, y_train)
    # clf_2
    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0,
                       random_state=None, splitter='best')
    pred_test_2 = clf_2.predict(X_test)
    #テストデータ 正解率
    res_text += "%s <br>"%(str(accuracy_score(y_test, pred_test_2)))

    #重要度の高い素性を表示
    importances = clf_2.feature_importances_
    indices = np.argsort(importances)[::-1]

    for f in range(X_train.shape[1]):
        res_text += "%2d) %-*s %f <br>" % (f + 1, 30,
                                df.columns[1+indices[f]],
                                importances[indices[f]])

    return res_text


def probability_ans(file_names, input_date, input_max_depth, input_ratio):
    response = ''
    for file_name in file_names:
        response += '%s<br>'%file_name
        response += probability_command(file_name, input_date, input_max_depth, input_ratio)
        response += '<br><br>'
    return response
    # try:
    #     response = probability_command(csv_file, input_date, input_max_depth, input_ratio)
    #     return response
    # except Exception as e:
    #     print('エラー')
    #     print('* 種類:', type(e))
    #     print('* 内容:', e)
