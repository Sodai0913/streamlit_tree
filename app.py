import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

#タイトル
st.title("機械学習アプリ")
st.write("streamlitで実装")

# 以下をサイドバーに表示
st.sidebar.markdown("### 機械学習に用いるcsvファイルを入力してください")
#ファイルアップロード
uploaded_files = st.sidebar.file_uploader("ファイルをアップロード", accept_multiple_files= False)
#ファイルがアップロードされたら以下が実行される

if uploaded_files:
    df = pd.read_csv(uploaded_files)
    df_columns = df.columns
    #データフレームを表示
    st.markdown("### 入力データ")
    st.dataframe(df.style.highlight_max(axis=0))

    st.divider()
    st.title("データの準備")

    #説明変数
    feature = st.multiselect("特長量の選択", df_columns.tolist(),df_columns.tolist())

    #目的変数
    target = st.selectbox("ターゲットの選択", df_columns)

    # テストデータを分ける
    test_size = st.slider("テストデータのサイズ", 0.0, 1.0, 0.5)

    df_train, df_test = train_test_split(df,
                                         test_size=test_size,
                                         random_state=1)

    st.divider()
    st.title("モデリング")

    #機械学習のタイプを選択する。
    model_option = st.selectbox("実行する機械学習のタイプ", ["ブースティング決定木","決定木","ランダムフォレスト"])
    
    #機械学習のタイプにより以下の処理が分岐
    if model_option == "ブースティング決定木":
            st.markdown("#### 機械学習を実行")
            execute = st.button("実行")

            model = GradientBoostingClassifier(n_estimators=100,random_state=1,max_depth=4)
            #実行ボタンを押したら下記が進む
            if execute:
                  model.fit(df_train[feature],df_train[target])

                  st.divider()
                  st.title("評価")

                  #予測
                  pred_train = model.predict(df_train[feature])
                  pred_test = model.predict(df_test[feature])

                  st.write("F1スコア:1に近いほど良い")
                  st.write("学習データ:",f1_score(df_train[target],pred_train))
                  st.write("テストデータ:",f1_score(df_test[target],pred_test))


    elif model_option == "決定木":
            st.markdown("#### 機械学習を実行") 
            execute = st.button("実行")
            
            model = DecisionTreeClassifier(max_depth=4)
            
            if execute:
                  model.fit(df_train[feature],df_train[target])
                  st.divider()
                  st.title("評価")

                  #予測
                  pred_train = model.predict(df_train[feature])
                  pred_test = model.predict(df_test[feature])

                  st.write("F1スコア:1に近いほど良い")
                  st.write("学習データ:",f1_score(df_train[target],pred_train))
                  st.write("テストデータ:",f1_score(df_test[target],pred_test))


    elif model_option == "ランダムフォレスト":
            st.markdown("#### 機械学習を実行") 
            execute = st.button("実行")

            model = RandomForestClassifier(n_estimators=200,
                                      max_depth=4,
                                      random_state=1)

            if execute:
                  model.fit(df_train[feature],df_train[target])
                  st.divider()
                  st.title("評価")

                  #予測
                  pred_train = model.predict(df_train[feature])
                  pred_test = model.predict(df_test[feature])

                  st.write("F1スコア:1に近いほど良い")
                  st.write("学習データ:",f1_score(df_train[target],pred_train))
                  st.write("テストデータ:",f1_score(df_test[target],pred_test))
