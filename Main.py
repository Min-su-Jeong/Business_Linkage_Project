import sys
import os
import pymssql
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIntValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from Data_analysis import Data_analysis
from Analysis_1d import Analysis_1d
from Analysis_2d import Analysis_2d
from Learning import Learning

# 빌드 시 경로 설정
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path("Main.ui")
form_main = uic.loadUiType(form)[0]

# 메인 윈도우 클래스
class MainWindow(QMainWindow, QWidget, form_main):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.show()

        # 데이터 프레임
        self.DF = pd.DataFrame()
        self.DF_sample = pd.DataFrame()
        self.scroll = QScrollArea()
        self.topFiller = QWidget()  # 위젯 생성

        # 메뉴 이벤트
        self.action_open_api.triggered.connect(lambda state, widget=self.table_dataframe: self.openApiFunction(state, widget)) # 데이터베이스 연결
        self.action_open_local.triggered.connect(lambda state, widget=self.table_dataframe: self.openLocalFunction(state, widget))
        self.action_exit.triggered.connect(lambda state: self.exitFunction())

        # 버튼 클릭 이벤트
        self.btn_data_analysis.clicked.connect(self.btnClicked_data_analysis)
        self.btn_analysis_1d.clicked.connect(self.btnClicked_analysis_1d)
        self.btn_analysis_2d.clicked.connect(self.btnClicked_analysis_2d)
        self.btn_learning.clicked.connect(self.btnClicked_learning)
        self.pushButton.clicked.connect(self.dialog_open)

        # plot 적용을 위한 초기화
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout_datachart.addWidget(self.canvas)
        self.layout_datachart.addWidget(self.toolbar)

    # 데이터 프레임 내용 GET
    def getDF(self):
        return self.DF

    # 열기(API) 함수 - 서버 데이터 가져오기
    def openApiFunction(self, state, widget):
        float_64 = ['float64_type_columns']
        int_64 = ['int64_type_columns']

        server = 'IP_address'
        db = 'db_name'
        user = 'user_name'
        pw = 'password'

        cnxn = pymssql.connect(server, user, pw, db)
        sql = 'query'
        df = pd.read_sql(sql, cnxn)

        # 전처리(dtype 변경)
        for i, col in enumerate(df.columns):
            if col in float_64:
                df[col] = df[col].astype('float64')
            elif col in int_64:
                df[col] = df[col].astype('int64')

        self.DF = df.copy()
        self.create_dataframe(widget, df.head(10))
        self.show_datainfo(df)
        self.show_datachart(df)

    # 열기(Local) 함수 - 자신의 local 환경에서 데이터 가져오기
    def openLocalFunction(self, state, widget):
        fname = QFileDialog.getOpenFileName(self, "파일 열기", "", "CSV files (*.csv)")

        if fname[0]:
            df = pd.read_csv(fname[0])
            self.DF = df.copy()
            self.create_dataframe(widget, df.head(10))
            self.show_datainfo(df)
            self.show_datachart(df)

    # 끝내기 함수
    def exitFunction(self):
        sys.exit(app.exec_())

    # window 창닫기 이벤트(고유)
    def closeEvent(self, QCloseEvent):
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?", QMessageBox.Yes | QMessageBox.No)
        if re == QMessageBox.Yes:
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()

    # 데이터프레임 생성
    def create_dataframe(self, widget, df):
        widget.setRowCount(len(df.index))
        widget.setColumnCount(len(df.columns))
        widget.setHorizontalHeaderLabels(df.columns)

        for row_index, row in enumerate(df.index):
            for col_index, col in enumerate(df.columns):
                widget.setItem(row_index, col_index, QTableWidgetItem(str(df.loc[row][col])))

    # 새로운 dialog 생성 - Option에서 사용
    def dialog_open(self):
        if len(self.getDF()) != 0:
            global dialog, cb, LineEdit

            dialog = QDialog()
            InnerLayOut = QVBoxLayout()
            groupbox = QGroupBox('CheckBox')
            scroll = QScrollArea()
            vbox = QVBoxLayout()
            LineEdit = QLineEdit()

            cb = []
            for i, col in enumerate(self.DF.columns):
                cb.append(QCheckBox("{}".format(col)))
                vbox.addWidget(cb[i])
            groupbox.setLayout(vbox)

            scroll.setWidget(groupbox)
            scroll.setWidgetResizable(True)

            # 라인 텍스트
            LineEdit.setPlaceholderText("시각화할 데이터 개수 입력(1~{})".format(len(self.DF)))
            LineEdit.setValidator(QIntValidator(1, len(self.DF), self)) # (1~샘플의 개수자리)까지 입력가능

            # 버튼 생성
            btnRun = QPushButton("Set Selected Features")
            btnRun.clicked.connect(self.setFeature)

            # Layout에 붙이기
            InnerLayOut.addWidget(scroll)
            InnerLayOut.addWidget(LineEdit)
            InnerLayOut.addWidget(btnRun)

            # 위젯 설정
            dialog.setLayout(InnerLayOut)
            dialog.setWindowTitle('Set Feature')
            dialog.setWindowModality(0)  ## Qt 에러
            dialog.resize(300, 200)
            dialog.show()
        else: # 데이터 파일을 불러오지 않은 경우
            QMessageBox.critical(self, 'Error', '데이터 파일을 먼저 불러와주시길 바랍니다.')

    # feature 선택에 따른 이벤트 처리 함수
    def setFeature(self):
        # LineEdit 텍스트 입력 여부 확인
        if LineEdit.text() == "":
            QMessageBox.critical(dialog, 'Error', '시각화할 데이터의 개수를 입력 바랍니다.')
        else:
            textList = []
            for i, col in enumerate(self.DF.columns):
                if (cb[i].isChecked() == True):
                    textList.append(cb[i].text())

            # feature 체크 여부 확인
            if len(textList) == 0:
                QMessageBox.critical(dialog, 'Error', 'Feature를 최소 1개 이상 선택 바랍니다.')
            else:
                DF_sample = self.DF.loc[:, textList]
                self.create_dataframe(self.table_dataframe, DF_sample.head(int(LineEdit.text())))
                dialog.close()

    # 샘플의 개수 정보를 출력하는 함수
    def show_datainfo(self, df):
        self.textEdit_sample.setText(str(len(df)))
        self.textEdit_good.setText(str(len(df[df['RESULT'] == 'GOOD'])))
        self.textEdit_bad.setText(str(len(df[df['RESULT'] != 'GOOD'])))

    # 데이터에 대한 결과 정보를 출력하는 함수
    def show_datachart(self, df):
        self.fig.clear()
        vc = df['RESULT'].value_counts().to_frame().reset_index()
        vc = vc.drop(0)
        self.ax = self.fig.add_subplot(111)
        sns.barplot(ax=self.ax, x=vc['RESULT'], y=vc['index'], data=vc)
        self.ax.set(xlabel=None, ylabel=None)
        self.ax.grid(True, axis='x', alpha=0.5, linestyle='--')
        self.fig.tight_layout()
        self.canvas.draw()

    # '데이터 분석으로 가기' 버튼 클릭 시 이벤트 처리 함수
    def btnClicked_data_analysis(self):
        if len(self.getDF()) != 0:
            self.dataAnalysis = Data_analysis(self.getDF())
            self.dataAnalysis.show()
        else:
            QMessageBox.critical(self, 'Error', '데이터 파일을 먼저 불러와주시길 바랍니다.')

    # '분석화면(1D)으로 가기' 버튼 클릭 시 이벤트 처리 함수
    def btnClicked_analysis_1d(self):
        if len(self.getDF()) != 0:
            self.analy1d = Analysis_1d(self.getDF())
            self.analy1d.show()
        else:
            QMessageBox.critical(self, 'Error', '데이터 파일을 먼저 불러와주시길 바랍니다.')

    # '분석화면(2D)으로 가기' 버튼 클릭 시 이벤트 처리 함수
    def btnClicked_analysis_2d(self):
        if len(self.getDF()) != 0:
            self.analy2d = Analysis_2d(self.getDF())
            self.analy2d.show()
        else:
            QMessageBox.critical(self, 'Error', '데이터 파일을 먼저 불러와주시길 바랍니다.')

    # '학습화면으로 가기' 버튼 클릭 시 이벤트 처리 함수
    def btnClicked_learning(self):
        self.learn = Learning()
        self.learn.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())