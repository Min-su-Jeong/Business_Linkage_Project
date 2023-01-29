import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import *
from PyQt5 import uic

# 빌드 시 경로 설정
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path("Analysis_2d.ui")
form_analysis_2d = uic.loadUiType(form)[0]

# 데이터 분석(2D) 윈도우 클래스
class Analysis_2d(QMainWindow, QWidget, form_analysis_2d):
    def __init__(self, df):
        super(Analysis_2d, self).__init__()
        self.setupUi(self)

        # plot 적용을 위한 초기화
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout_graph.addWidget(self.canvas)
        self.layout_graph.addWidget(self.toolbar)

        # 로드한 데이터 프레임 정보 사용
        self.df = df

        state = np.where(self.df['RESULT'] == 'GOOD', 'G', 'NG')
        self.df = self.df.assign(RESULT=state)
        self.Gdf, self.NGdf = self.df[self.df['RESULT'] == 'G'], self.df[self.df['RESULT'] == 'NG']

        # comboBox 초기화(column 삽입)
        self.graph_list = ['Regplot', 'Heatmap', 'Kdeplot', 'Jointplot']

        ## comboBox_graph
        for graph in self.graph_list:
            self.comboBox_graph.addItem(graph)

        ## comboBox_col
        for col in self.df:
            if col == 'LOT' or col == 'TIME_STAMP' or col == 'RESULT':
                continue
            else:
                self.comboBox_col.addItem(col)
                self.comboBox_col_2.addItem(col)

        ## comboBox 이름 변경 시 이벤트 처리
        self.comboBox_graph.currentTextChanged.connect(self.comboBox_select)
        self.comboBox_col.currentTextChanged.connect(self.comboBox_select)
        self.comboBox_col_2.currentTextChanged.connect(self.comboBox_select)

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
            self.doRegplot(self.comboBox_col.currentText(),self.comboBox_col_2.currentText())
        elif self.comboBox_graph.currentText() == self.graph_list[1]:
            self.doHeatmap(self.comboBox_col.currentText(),self.comboBox_col_2.currentText())
        elif self.comboBox_graph.currentText() == self.graph_list[2]:
            self.doKdeplot(self.comboBox_col.currentText(),self.comboBox_col_2.currentText())
        elif self.comboBox_graph.currentText() == self.graph_list[3]:
            self.doJointplot(self.comboBox_col.currentText(),self.comboBox_col_2.currentText())

    def doRegplot(self, feature,feature2):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        if self.checkBox_G.isChecked() and self.checkBox_NG.isChecked():
            sns.regplot(x=feature, y=feature2, data=self.df, ax=self.ax,
                        scatter_kws={"fc": "b", "ec": "b", "s": 100, "alpha": 0.3}, color="r", line_kws={"lw":3, "ls":"--","alpha":0.5})
        else:
            if self.checkBox_G.isChecked():
                sns.regplot(x=feature, y=feature2, data=self.Gdf, ax=self.ax,
                            scatter_kws={"fc": "b", "ec": "b", "s": 100, "alpha": 0.3}, color="r",
                            line_kws={"lw": 3, "ls": "--", "alpha": 0.5})
            elif self.checkBox_NG.isChecked():
                sns.regplot(x=feature, y=feature2, data=self.NGdf, ax=self.ax,
                            scatter_kws={"fc": "b", "ec": "b", "s": 100, "alpha": 0.3}, color="r",
                            line_kws={"lw": 3, "ls": "--", "alpha": 0.5})
            self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def doHeatmap(self, feature,feature2):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        corr_df = self.df[[feature, feature2]].astype(float).corr()
        sns.heatmap(corr_df, ax=self.ax, linecolor = "white", annot = True, annot_kws = {"size" : 16})
        self.fig.tight_layout()
        self.canvas.draw()

    def doKdeplot(self, feature, feature2):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        if self.checkBox_G.isChecked() and self.checkBox_NG.isChecked():
            sns.kdeplot(ax=self.ax, data=self.Gdf, x=feature, y=feature2, cmap="Greens", fill='fill', bw_adjust=.5, label='G')
        else:
            if self.checkBox_G.isChecked():
                sns.kdeplot(ax=self.ax, data=self.Gdf, x=feature, y=feature2, cmap="Blues", fill='fill', bw_adjust=.5, label='G')
            if self.checkBox_NG.isChecked():
                sns.kdeplot(ax=self.ax, data=self.NGdf, x=feature, y=feature2, cmap="Reds", fill='fill', bw_adjust=.5, label='NG')
        self.fig.tight_layout()
        self.canvas.draw()

    def doJointplot(self, feature, feature2):
        self.fig.clear()
        widths = [4, 1]
        heights = [1, 4]
        spec = self.fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)
        axs = {}
        for i in range(len(heights) * len(widths)):
            axs[i] = self.fig.add_subplot(spec[i // len(widths), i % len(widths)])
        sns.scatterplot(ax=axs[2], x=feature, y=feature2, data=self.df, hue='RESULT', legend=True)
        sns.kdeplot(self.df[feature], ax=axs[0], legend=False)
        axs[0].set_xlim(axs[2].get_xlim())
        axs[0].set_xlabel('')
        axs[0].set_xticklabels([])
        axs[0].spines["left"].set_visible(False)
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        sns.kdeplot(y=self.df[feature2], ax=axs[3], legend=False)
        axs[3].set_ylim(axs[2].get_ylim())
        axs[3].set_ylabel('')
        axs[3].set_yticklabels([])
        axs[3].spines["bottom"].set_visible(False)
        axs[3].spines["top"].set_visible(False)
        axs[3].spines["right"].set_visible(False)
        axs[1].axis("off")
        self.fig.tight_layout()
        self.canvas.draw()