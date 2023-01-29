import os
import sys
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import *
from PyQt5 import uic

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# 빌드 시 경로 설정
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path("Learning.ui")
form_learning = uic.loadUiType(form)[0]

# 학습 윈도우 클래스
class Learning(QMainWindow, QWidget, form_learning):
    def __init__(self):
        super(Learning, self).__init__()
        self.setupUi(self)
        self.show()

        self.df_train = ''
        self.df_test = ''

        self.btn_load_1.clicked.connect(lambda state: self.loadTrainFile(state))
        self.btn_learn_start.clicked.connect(lambda state: self.startLearning(state, self.df_train))
        self.btn_main.clicked.connect(self.close)

    # 학습시킬 데이터 파일 불러오기 함수
    def loadTrainFile(self, state):
        fname = QFileDialog.getOpenFileName(self, "파일 열기", "", "CSV files (*.csv)")
        #print(fname[0])
        if fname[0]:
            self.textEdit_1.setText(str(fname[0]))
            self.df_train = pd.read_csv(fname[0])

    # 데이터 프레임 생성 함수
    def create_dataframe(self, widget, df):
        widget.setRowCount(len(df.index))
        widget.setColumnCount(len(df.columns))
        widget.setHorizontalHeaderLabels(df.columns)
        widget.setVerticalHeaderLabels(df.index)

        for row_index, row in enumerate(df.index):
            for col_index, col in enumerate(df.columns):
                widget.setItem(row_index, col_index, QTableWidgetItem(str(df.loc[row][col])))

    # 학습 시작 함수(버튼 누를 시 시작)
    def startLearning(self, state, df):
        # preprocessing
        ## GOOD / NOT GOOD에 대하여 0과 1로 이진화 수행
        state = np.where(df['RESULT'] == 'GOOD', 0, 1)  # 0: 정상 / 1: 불량
        df = df.assign(RESULT=state)
        result_col = df['RESULT']

        ## 결측치 제거
        df.dropna(axis=1, inplace=True)

        ## 유효한 column만 남겨서 df 재정의
        df = df.loc[:, ['HPTransPrs', 'InjPress', 'SPHUMIDITY', 'UvTemp1', 'UvTemp2', 'WH3INTEMP', 'WH3OUTTEMP', 'supportHumidity']]

        ## 정규화 수행
        scaler = StandardScaler()
        scaler.fit(df.iloc[:, :-1])

        s_df = scaler.transform(df.iloc[:, :-1])
        s_df = pd.DataFrame(s_df, index=df.index, columns=df.columns[:-1])

        ## X, y 정의
        X = s_df
        y = result_col

        ## train-test split 수행
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        ## oversampling 수행(train 셋에 대해서만)
        os = SMOTE(random_state=2022)
        X_train, y_train = os.fit_resample(X_train, y_train)

        # fitting
        ## XGBClassifier 모델 생성/학습
        model_xgb = XGBClassifier()
        model_xgb.fit(X_train, y_train)

        # prediction
        pred = model_xgb.predict(X_test)

        # Show data
        ## Description
        #self.create_dataframe(self.table_desc, np.round(df.describe(), 2))

        ## History
        # self.textEdit_hist.clear()
        # for idx, loss in enumerate(hist.history['loss']):
        #     self.textEdit_hist.append(
        #         "Epoch: {:2} - loss: {:.3f} - val_loss: {:.3f}".format(idx + 1, loss, hist.history['val_loss'][idx]))

        ## Result
        self.textEdit_acc.setText(classification_report(y_test, pred))
