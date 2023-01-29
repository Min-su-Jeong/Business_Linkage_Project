import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy.stats import shapiro, stats, wilcoxon

from PyQt5 import uic
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# 빌드 시 경로 설정
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


form = resource_path("Data_analysis.ui")
form_data_analysis = uic.loadUiType(form)[0]

# 데이터 분석 윈도우 클래스
class Data_analysis(QMainWindow, QWidget, form_data_analysis):
    def __init__(self, df):
        super(Data_analysis, self).__init__()
        self.setupUi(self)

        # 로드한 데이터프레임 정보 사용
        self.df = df
        state = np.where(self.df['RESULT'] == 'GOOD', 0, 1)  # 0: 정상 / 1: 불량
        self.df = self.df.assign(RESULT=state)

        # plot에 적용하기 위한 데이터프레임 변수
        self.df1 = df
        state = np.where(self.df1['RESULT'] == 'GOOD', 'G', 'NG')
        self.df1 = self.df1.assign(RESULT=state)
        self.Gdf, self.NGdf = self.df1[self.df1['RESULT'] == 'G'], self.df1[self.df1['RESULT'] == 'NG']

        # plot 적용을 위한 초기화
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout_graph.addWidget(self.canvas)
        self.layout_graph.addWidget(self.toolbar)

        # 버튼 리스트 만들기
        self.btn_list = [self.btn_stage1, self.btn_stage2, self.btn_stage3, self.btn_stage4, self.btn_stage5,
                         self.btn_stage6]
        # 첫 번째 버튼을 제외한 나머지 버튼 비활성화
        for i in range(len(self.btn_list)):
            if i != 0:
                self.btn_list[i].setEnabled(False)

        # 버튼 클릭 이벤트 처리
        self.btn_stage1.clicked.connect(self.btnClicked_stage)
        self.btn_stage2.clicked.connect(self.btnClicked_stage)
        self.btn_stage3.clicked.connect(self.btnClicked_stage)
        self.btn_stage4.clicked.connect(self.btnClicked_stage)
        self.btn_stage5.clicked.connect(self.btnClicked_stage)
        self.btn_stage6.clicked.connect(self.btnClicked_stage)
        self.btn_main.clicked.connect(self.close)

        # 최종 결과 리스트 변수
        self.result_list = []

        # 콤보 박스 비활성화(단계 모두 수행 시 활성화)
        self.graph_list = ['Axvline', 'Histplot', 'Pointplot', 'Boxplot', 'Violinplot', 'Stripplot', 'Kdeplot']
        ## comboBox_graph
        for graph in self.graph_list:
            self.comboBox_graph.addItem(graph)

        self.comboBox_graph.setEnabled(False)
        self.comboBox_col.setEnabled(False)

        # comboBox 이름 변경 시 이벤트 처리
        self.comboBox_graph.currentTextChanged.connect(self.comboBox_select)
        self.comboBox_col.currentTextChanged.connect(self.comboBox_select)

        # checkBox 상태 변경 시 이벤트 처리
        self.checkBox_G.toggle()
        self.checkBox_NG.toggle()
        self.checkBox_G.stateChanged.connect(self.comboBox_select)
        self.checkBox_NG.stateChanged.connect(self.comboBox_select)

        # 진행바 값 초기화
        self.progress = 0
        self.progressBar.setValue(0)

        self.show()

    # 최종 결과로 나온 feature 목록 반환 함수
    def getResult(self):
        return self.result_list

    # 진행 정도 업데이트 함수
    def addProgress(self, num):
        self.progress += num
        return self.progress

    # 버튼 클릭 시 이벤트 처리 함수
    def btnClicked_stage(self):
        # textEdit & progress & progress 초기화
        self.textEdit_desc.clear()
        self.progress = 0
        self.progressBar.reset()

        # 제거된 feature 목록 변수 정의
        self.remove_list = []

        # 클릭한 버튼 이름 가져오기
        btn_text = self.sender().text()

        # 각 버튼에 따른 이벤트 처리
        if btn_text == '이상/결측치 제거':
            # 표준편차 0 제거
            for col in self.df.columns:
                if self.df[col].dtype != object and self.df[col].std() == 0:
                    self.df.drop(columns=col, axis=1, inplace=True)
                    self.remove_list.append(col)

            # 결측치 제거
            self.df.dropna(axis=1, inplace=True)
            self.progressBar.setValue(50)

            # 텍스트 출력
            self.textEdit_desc.append('[이상/결측치 제거]\n' +
                                      '- 이상치 제거: 표준편차가 0인 경우(데이터 변화X)에 해당하는 경우 제거\n' +
                                      '- 결측치 제거: 데이터에 값이 없는 경우 or 유효하지 않은 경우 or 숫자가 아닌 경우 제거\n')
            self.textEdit_desc.append("[제거된 feature 목록]\n" + str(self.remove_list) + '\n')
            self.textEdit_desc.append("[남은 feature 목록]\n" + str(list(self.df.columns)))

            # 버튼 상태 변경
            self.btn_list[0].setEnabled(False)
            self.btn_list[1].setEnabled(True)
            self.progressBar.setValue(100)

        elif btn_text == 'Object type 제거':
            # Object type 제거
            for col in self.df.columns:
                if self.df[col].dtype == object:
                    self.df.drop(columns=col, axis=1, inplace=True)
                    self.remove_list.append(col)
            self.progressBar.setValue(50)

            # 텍스트 출력
            self.textEdit_desc.append("[Object type 제거]\n" +
                                      "다양한 분석을 하기 위해선 label에 대한 데이터를 제외한 나머지 데이터 유형이 " +
                                      "모두 숫자 데이터여야 하므로 Object type인 feature를 제거합니다.\n")
            self.textEdit_desc.append("[제거된 feature 목록]\n" + str(self.remove_list) + '\n')
            self.textEdit_desc.append("[남은 feature 목록]\n" + str(list(self.df.columns)))

            # 버튼 상태 변경
            self.btn_list[1].setEnabled(False)
            self.btn_list[2].setEnabled(True)
            self.progressBar.setValue(100)

        elif btn_text == '선형 회귀 분석':
            # 선형 회귀 분석
            ## 전진 단계별 선택법
            variables = self.df.columns[:-2].tolist()  ## 설명 변수 리스트
            selected_variables = []  ## 선택된 변수들
            sl_enter = 0.05
            sl_remove = 0.05

            sv_per_step = []  ## 각 스텝별로 선택된 변수들
            adjusted_r_squared = []  ## 각 스텝별 수정된 결정 계수
            steps = []
            step = 0
            self.progressBar.setValue(self.addProgress(10))

            while len(variables) > 0:
                remainder = list(set(variables) - set(selected_variables))
                pval = pd.Series(index=remainder, dtype='float64')  ## 변수의 p-value

                ## 기존에 포함된 변수와 새로운 변수 하나씩 조합하여 선형 모형을 적합한다.
                for col in remainder:
                    y = self.df['RESULT']  ## 반응 변수
                    X = self.df[selected_variables + [col]]
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()
                    pval[col] = model.pvalues[col]

                min_pval = pval.min()
                if min_pval < sl_enter:  ## 최소 p-value 값이 기준 값보다 작으면 포함
                    selected_variables.append(pval.idxmin())
                    ## 선택된 변수들에 대해서 제거할 변수 선택
                    while len(selected_variables) > 0:
                        selected_X = self.df[selected_variables]
                        selected_X = sm.add_constant(selected_X)
                        selected_pval = sm.OLS(y, selected_X).fit().pvalues[1:]  ## 절편 항의 p-value 제거
                        max_pval = selected_pval.max()
                        if max_pval >= sl_remove:  ## 최대 p-value값이 기준 값보다 크거나 같으면 제외
                            remove_variable = selected_pval.idxmax()
                            selected_variables.remove(remove_variable)
                        else:
                            break

                    step += 1
                    steps.append(step)
                    adj_r_squared = sm.OLS(y, sm.add_constant(self.df[selected_variables])).fit().rsquared_adj
                    adjusted_r_squared.append(adj_r_squared)
                    sv_per_step.append(selected_variables.copy())
                    self.progressBar.setValue(self.addProgress(3))
                else:
                    break

            slt = self.df[selected_variables]
            for col in self.df.columns:
                if col not in slt.columns:
                    if col != 'RESULT':
                        self.remove_list.append(col)
            self.df = pd.concat([slt, self.df['RESULT']], axis=1)

            # 텍스트 출력
            self.textEdit_desc.append('[선형 회귀 분석(OLS: Ordinary Least Squares)이란?]\n' +
                                      '잔차제곱합(RSS: Residual Sum of Squares)을 최소화하는 가중치 벡터를 구하는 방법입니다.' +
                                      '각각의 독립변수 xi가 종속변수 y에 영향이 있는지 확인가능하며 p-value 값을 기준으로 무효한 feature들을 제거합니다.\n')
            self.textEdit_desc.append("[제거된 feature 목록]\n" + str(self.remove_list) + '\n')
            self.textEdit_desc.append("[남은 feature 목록]\n" + str(list(self.df.columns)))

            # 버튼 상태 변경
            self.btn_list[2].setEnabled(False)
            self.btn_list[3].setEnabled(True)
            self.progressBar.setValue(100)

        elif btn_text == '정규화 검정':
            global nstd_remove_list

            Sample_ng = []
            nstd_remove_list = []

            Gdf = self.df[self.df['RESULT'] == 0]
            NGdf = self.df[self.df['RESULT'] == 1]
            min_value = min(len(Gdf), len(NGdf))

            epochs = 500
            self.progressBar.setMaximum(epochs + 1)

            # 정규화 검정
            for epoch in range(epochs):
                std_list = []
                nstd_list = []
                Sample_g = []

                g = Gdf.sample(n=min_value, replace=False, axis=0)
                for i, col in enumerate(g.columns):
                    if col != 'RESULT':
                        cnt = 0
                        sample_g = g[col]

                        if min_value <= 100:
                            while (stats.ttest_1samp(sample_g, Gdf[col].mean()).pvalue > 0.05):
                                sample_g = Gdf[col].sample(n=min_value, replace=False)
                        elif min_value > 100:
                            while (stats.ttest_1samp(sample_g, Gdf[col].mean()).pvalue > 0.05):
                                sample_g = Gdf[col].sample(n=min_value, replace=False)

                        Sample_g.append(sample_g)
                        Sample_ng.append(NGdf.sample(n=min_value, replace=False, axis=0)[col])

                for i, col in enumerate(g.columns):
                    if col != 'RESULT':
                        # 정규화 검정 & 독립 t 검정
                        if float(shapiro(Sample_ng[i]).pvalue) < 0.05 and float(shapiro(Sample_g[i]).pvalue) < 0.05:
                            if col not in nstd_list:
                                nstd_list.append(col)
                            if wilcoxon(Sample_g[i], Sample_ng[i], zero_method='pratt').pvalue < 0.05:
                                self.result_list.append(Sample_g[i])
                            else:
                                if col not in nstd_remove_list:
                                    nstd_remove_list.append(col)
                        else:
                            if col not in std_list:
                                std_list.append(col)

                self.progressBar.setValue(self.addProgress(1))

            res = pd.DataFrame(self.result_list).T.columns
            unique, counts = np.unique(res, return_counts=True)
            uniq_cnt_dict = dict(zip(unique, counts))

            self.result_list = []
            for col in uniq_cnt_dict:
                if uniq_cnt_dict[col] >= epochs / 2:
                    self.result_list.append(col)

            # 텍스트 출력
            self.textEdit_desc.append('[정규화 검정(Shapiro-Wilk test)이란?]\n'
                                      '자료의 값들과 표준정규점수와의 선형상관관계를 측정하여 표본이 정규 분포의 가정을 만족하는지 검정하는 방법 입니다.\n')
            self.textEdit_desc.append('[결과]\n' + '정규화를 따르는 feature\n' + str(std_list) +
                                      '\n\n정규화를 따르지 않는 feature\n' + str(nstd_list))
            # 버튼 상태 변경
            self.btn_list[3].setEnabled(False)
            self.btn_list[4].setEnabled(True)

            self.progressBar.setValue(epochs + 1)

        elif btn_text == '독립 t 검정':
            self.progressBar.setMaximum(100)

            # 버튼 상태 변경
            self.btn_list[4].setEnabled(False)
            self.btn_list[5].setEnabled(True)

            # 텍스트 출력
            self.textEdit_desc.append('[독립 t 검정(Independent t-test)이란?]\n'
                                      '두 집단의 평균 차이를 검증하기 위한 방법입니다.\n')
            self.textEdit_desc.append("[제거된 feature 목록]\n" + str(nstd_remove_list) + '\n')
            self.textEdit_desc.append("[남은 feature 목록]\n" + str(self.result_list))
            self.progressBar.setValue(100)

        elif btn_text == '데이터 분석 시작':
            # comboBox_col
            for col in self.result_list:
                if col == 'LOT' or col == 'TIME_STAMP' or col == 'RESULT':
                    continue
                else:
                    self.comboBox_col.addItem(col)

            # combobox 활성화
            self.comboBox_graph.setEnabled(True)
            self.comboBox_col.setEnabled(True)

            # 버튼 상태 변경
            self.btn_list[5].setEnabled(False)
            self.progressBar.setValue(100)

    # combobox 선택에 따른 이벤트 처리 함수
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
        self.showAnalysisResult(self.comboBox_col.currentText())

    # 분석 결과에 대한 설명을 출력하는 함수
    def showAnalysisResult(self, feature):
        self.textEdit_desc.clear()
        self.progressBar.setValue(0)

        result = self.df1['RESULT']
        df_res = pd.concat([self.df1.loc[:, self.result_list], result], axis=1)

        G_list = list(df_res[df_res['RESULT'] == 'G'][feature].values)
        G_list_uni = list(df_res[df_res['RESULT'] == 'G'][feature].unique())
        NG_list = list(df_res[df_res['RESULT'] != 'G'][feature].values)
        NG_list_uni = list(df_res[df_res['RESULT'] != 'G'][feature].unique())

        G_cnt, NG_cnt = [], []
        for G_value, NG_value in zip(G_list_uni, NG_list_uni):
            G_cnt.append(G_list.count(G_value))
            NG_cnt.append(NG_list.count(NG_value))
            self.progressBar.setValue(self.addProgress(2))
        gd, ngd = dict(zip(G_list_uni, G_cnt)), dict(zip(NG_list_uni, NG_cnt))
        gd_sort = sorted(gd.items(), key=lambda item: item[1], reverse=True)
        ngd_sort = sorted(ngd.items(), key=lambda item: item[1], reverse=True)

        # 텍스트 출력
        self.textEdit_desc.append('//데이터 분석 결과//\n\n' +
                                  "[1. 최종 feature 목록]\n" + str(self.result_list) + '\n\n' +
                                  '[2. G/NG 분포 범위]\nG : {} ~ {}\nNG: {} ~ {}\n\n'.format(min(gd), max(gd), min(ngd),
                                                                                         max(ngd)) +
                                  '[3. G/NG 중복X 값]\nG:\n{}\nNG:\n{}\n\n'.format(
                                      sorted([g for g in G_list_uni if g not in NG_list_uni]),
                                      sorted([ng for ng in NG_list_uni if ng not in G_list_uni])) +
                                  '[4. G/NG 최빈값 TOP 5]\nG : {}\nNG: {}\n\n'.format([gd_sort[i][0] for i in range(5)],
                                                                                    [ngd_sort[i][0] for i in
                                                                                     range(5)]) +
                                  '[5. NG 가장 적게 관측되는 값]\n{}'.format(
                                      sorted([key for key in ngd if ngd[key] == min(ngd.values())])))
        self.progressBar.setValue(100)

    # 유형에 따른 그래프 그리기 함수들
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
        self.canvas.draw()

    def doKdeplot(self, feature):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(feature)
        if self.checkBox_G.isChecked():
            sns.kdeplot(ax=self.ax, data=self.df1[self.df1['RESULT'] == 'G'][feature], label='G')
        if self.checkBox_NG.isChecked():
            sns.kdeplot(ax=self.ax, data=self.df1[self.df1['RESULT'] == 'NG'][feature], label='NG')
        self.ax.legend()
        self.canvas.draw()