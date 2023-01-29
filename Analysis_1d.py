import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PyQt5 import uic
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# 빌드 시 경로 설정
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path("Analysis_1d.ui")
form_analysis_1d = uic.loadUiType(form)[0]

# 데이터 분석(1D) 윈도우 클래스
class Analysis_1d(QMainWindow, QWidget, form_analysis_1d):
    def __init__(self, df):
        super(Analysis_1d, self).__init__()
        self.setupUi(self)

        # plot 적용을 위한 초기화
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout_graph.addWidget(self.canvas)
        self.layout_graph.addWidget(self.toolbar)

        # 로드한 데이터프레임 정보 사용
        self.df1 = df
        state = np.where(self.df1['RESULT'] == 'GOOD', 'G', 'NG')
        self.df1 = self.df1.assign(RESULT=state)
        self.Gdf, self.NGdf = self.df1[self.df1['RESULT'] == 'G'], self.df1[self.df1['RESULT'] == 'NG']

        # comboBox 초기화(column 삽입)
        self.graph_list = ['Axvline', 'Histplot', 'Pointplot', 'Boxplot', 'Violinplot', 'Stripplot', 'Kdeplot']
        ## comboBox_graph
        for graph in self.graph_list:
            self.comboBox_graph.addItem(graph)

        ## comboBox_col
        for col in self.df1:
            if col == 'LOT' or col == 'TIME_STAMP' or col == 'RESULT':
                continue
            else:
                self.comboBox_col.addItem(col)
        ## comboBox 이름 변경 시 이벤트 처리
        self.comboBox_graph.currentTextChanged.connect(self.comboBox_select)
        self.comboBox_col.currentTextChanged.connect(self.comboBox_select)

        # checkBox 상태 변경 시 이벤트 처리
        self.checkBox_G.toggle()
        self.checkBox_NG.toggle()
        self.checkBox_G.stateChanged.connect(self.comboBox_select)
        self.checkBox_NG.stateChanged.connect(self.comboBox_select)

        # 버튼 클릭 이벤트 처리
        self.btn_main.clicked.connect(self.close)

        self.show()

    # Combobox 이벤트 처리 함수
    def comboBox_select(self):
        if self.comboBox_graph.currentText() == self.graph_list[0]:
            self.doAxvline(self.comboBox_col.currentText())
        elif self.comboBox_graph.currentText() == self.graph_list[1]:
            self.doHistplot(self.comboBox_col.currentText())
        elif self.comboBox_graph.currentText() == self.graph_list[2]:
            self.doPointplot(self.comboBox_col.currentText())
        elif self.comboBox_graph.currentText() == self.graph_list[3]:
            self.doBoxplot(self.comboBox_col.currentText())
        elif self.comboBox_graph.currentText() == self.graph_list[4]:
            self.doViolinplot(self.comboBox_col.currentText())
        elif self.comboBox_graph.currentText() == self.graph_list[5]:
            self.doStripplot(self.comboBox_col.currentText())
        elif self.comboBox_graph.currentText() == self.graph_list[6]:
            self.doKdeplot(self.comboBox_col.currentText())

    # 유형별 그래프 그리기 함수들
    def doAxvline(self, feature):
        self.fig.clear()
        defective_index = []
        for i in range(len(self.df1['RESULT'])):
            if self.df1.iloc[i]['RESULT'] != 'G':
                defective_index.append(i)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(feature)
        if self.checkBox_G.isChecked():
            self.ax.plot(self.df1.loc[:, feature], color='b', label='G')  # good -> blue
        if self.checkBox_NG.isChecked():
            for i, xc in enumerate(defective_index):
                if i == 0:
                    self.ax.axvline(x=xc, color='r', linestyle='-', linewidth=1, alpha=0.2, label='NG')
                else:
                    self.ax.axvline(x=xc, color='r', linestyle='-', linewidth=1, alpha=0.2)  # not good -> red
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def doHistplot(self, feature):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(feature)
        if self.checkBox_G.isChecked() and self.checkBox_NG.isChecked():
            sns.histplot(ax=self.ax, x=feature, data=self.df1, hue='RESULT', legend=True)
        else:
            if self.checkBox_G.isChecked():
                sns.histplot(ax=self.ax, x=feature, data=self.Gdf, hue='RESULT', label='G')
            elif self.checkBox_NG.isChecked():
                sns.histplot(ax=self.ax, x=feature, data=self.NGdf, hue='RESULT', label='NG')
            self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def doPointplot(self, feature):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(feature)
        if self.checkBox_G.isChecked() and self.checkBox_NG.isChecked():
            sns.pointplot(ax=self.ax, x='RESULT', y=feature, data=self.df1, hue='RESULT')
        else:
            if self.checkBox_G.isChecked():
                sns.pointplot(ax=self.ax, x='RESULT', y=feature, data=self.Gdf, hue='RESULT')
            if self.checkBox_NG.isChecked():
                sns.pointplot(ax=self.ax, x='RESULT', y=feature, data=self.NGdf, hue='RESULT')
        self.fig.tight_layout()
        self.canvas.draw()

    def doBoxplot(self, feature):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(feature)
        if self.checkBox_G.isChecked() and self.checkBox_NG.isChecked():
            sns.boxplot(ax=self.ax, x='RESULT', y=feature, data=self.df1, hue='RESULT')
        else:
            if self.checkBox_G.isChecked():
                sns.boxplot(ax=self.ax, x='RESULT', y=feature, data=self.Gdf, hue='RESULT')
            if self.checkBox_NG.isChecked():
                sns.boxplot(ax=self.ax, x='RESULT', y=feature, data=self.NGdf, hue='RESULT')
        self.fig.tight_layout()
        self.canvas.draw()

    def doViolinplot(self, feature):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(feature)
        if self.checkBox_G.isChecked() and self.checkBox_NG.isChecked():
            sns.violinplot(ax=self.ax, x='RESULT', y=feature, data=self.df1, hue='RESULT')
        else:
            if self.checkBox_G.isChecked():
                sns.violinplot(ax=self.ax, x='RESULT', y=feature, data=self.Gdf, hue='RESULT')
            if self.checkBox_NG.isChecked():
                sns.violinplot(ax=self.ax, x='RESULT', y=feature, data=self.NGdf, hue='RESULT')
        self.fig.tight_layout()
        self.canvas.draw()

    def doStripplot(self, feature):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(feature)
        if self.checkBox_G.isChecked() and self.checkBox_NG.isChecked():
            sns.stripplot(ax=self.ax, x='RESULT', y=feature, data=self.df1, hue='RESULT', jitter=True)
        else:
            if self.checkBox_G.isChecked():
                sns.stripplot(ax=self.ax, x='RESULT', y=feature, data=self.Gdf, hue='RESULT', jitter=True)
            if self.checkBox_NG.isChecked():
                sns.stripplot(ax=self.ax, x='RESULT', y=feature, data=self.NGdf, hue='RESULT', jitter=True)
        self.fig.tight_layout()
        self.canvas.draw()

    def doKdeplot(self, feature):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(feature)
        if self.checkBox_G.isChecked():
            sns.kdeplot(ax=self.ax, data=self.df1[self.df1['RESULT'] == 'G'][feature], warn_singular=False, label='G')
        if self.checkBox_NG.isChecked():
            sns.kdeplot(ax=self.ax, data=self.df1[self.df1['RESULT'] == 'NG'][feature], warn_singular=False, label='NG')
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()