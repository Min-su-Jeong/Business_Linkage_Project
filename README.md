# Business_Linkage_Project
기업 연계형 프로젝트 개발을 위한 Repository 입니다.
- 사출기 결과물의 양불 판정을 위한 이상치 검출을 위한 GUI 분석 프로그램 개발
<br/><br/>

## Functions
|기능|구현 여부|설명|
|:---:|:---:|---|
|데이터 불러오기(Local)|O|자신의 컴퓨터(Local)에 존재하는 .csv 형식의 파일 데이터를 불러올 수 있다.|
|데이터 불러오기(API)|O|회사 내 데이터베이스(DB)에 존재하는 사출기 결과에 대한 데이터를 불러올 수 있다.(주기적으로 업데이트 됨.)|
|데이터 기본정보 표시|O|불러온 데이터 결과(RESULT)를 기반으로 데이터에 대한 기본 정보를 표시한다.(Dataframe, Bar plot, 양/불 개수)|
|데이터 부분 선택|O|불러온 대량의 데이터 중 사용자가 부분적으로 보고자하는 데이터 feature 종류와 개수 등을 직접 설정하여 필요한 부분 데이터만 표시할 수 있다.|
|데이터 시각화|O|불러온 데이터를 기반으로 다양한 플롯(Plot)을 사용하여 데이터 시각화가 가능하다. 사용자가 보고 싶은 플롯(Plot)을 선택할 수 있으며 특징(Feature)의 개수에 따라 1D, 2D로 나뉜다.|
|데이터 분석|O|총 5단계로 나누어 데이터 분석을 진행한다. 이상/결측치 제거, Object type 제거, 선형 회귀 분석, 정규화 검정, 독립 T 검정 단계가 순차적으로 진행된다.|
|데이터 분석 결과 도출|O|총 5단계의 데이터 분석을 통해 전체 데이터에 대한 결과를 도출해낼 수 있다. 이를 그래프와 결과 설명으로 사용자에게 제공된다.|
|데이터 학습 및 예측|O|양/불 판정을 위해 Anomaly Dectection 기반 딥러닝 모델을 사용하여 사용자가 제공하는 데이터에 대해서 학습 및 예측을 진행한다.|
|GUI 프로그래밍|O|GUI 프로그래밍을 통해 사용자가 좀 더 효과적으로 데이터 분석 과정을 확인할 수 있고 쉽게 이해할 수 있다.|
|이상치 검출 알고리즘 개발 및 정확도 향상|△|딥러닝 모델(XGBoost Classifier)을 활용한 이상치 검출 알고리즘을 통해 사출기 결과물의 양품과 불량을 판정하는 알고리즘을 개발하고 높은 정확도를 달성하기 위해 다양한 실험을 진행 중이다.|

<br/>

## Flowchart
<p align="center">
  <img src="https://user-images.githubusercontent.com/74342121/215761959-da1aaff6-947c-4838-a650-5d950249a2aa.png" width="450" height="500">
</p>

```
👉 실행 흐름
 1. 프로그램 시작 
 2. 메인화면 
 3. 데이터 파일 불러오기(.csv) 
 4. 데이터 부분 선택(option), 데이터 기본정보 출력
 5. 4가지 기능 중 1가지 선택하여 수행(데이터 분석, 데이터 시각화(1D), 데이터 시각화(2D), 학습) 
 6. 메인화면에서 창 닫기
```

<br/>

## System Configuration
<p align="center">
  <img src="https://user-images.githubusercontent.com/74342121/215767736-10bbd4dd-157a-4474-a8e5-c4204e3f17ea.png" width="450" height="230">
</p>

<br/>

## Module Design
```
⦁ Main(메인화면)
  - 프로그램의 실행과 함께 실행되며 사용자가 데이터를 불러올 수 있고 다양한 분석 기능을 수행할 수 있도록 제공한다.
  
⦁ Data_analysis(데이터 분석)
  - 총 5단계의 데이터 분석이 이루어지며 step-by-step 방식으로 사용자가 분석 진행 과정을 직접 볼 수 있도록 설명을 제공한다.
  
⦁ Analysis_1d(데이터 시각화(1D))
  - 사용자가 1개의 feature와 그래프 유형을 선택하여 해당 데이터에 대한 결과를 시각적으로 확인할 수 있도록 기능을 제공한다.
  
⦁ Analysis_2d(데이터 시각화(2D))
  - 사용자가 2개의 feature와 그래프 유형을 선택하여 해당 데이터에 대한 결과를 시각적으로 확인할 수 있도록 기능을 제공한다.
  
⦁ Learning(학습)
  - 사용자가 학습 데이터를 제공하면 내부 구현되어 있는 딥러닝 모델 학습과정에 따라 Anomaly detection 학습을 진행한다. 
    그 후, 예측을 수행하여 사용자에게 데이터에 대한 예측 정확도를 보여주어 얼마나 학습이 잘 되었는지 지표를 제공해준다.
```

<br/>

## Development
### 1. 메인 화면 - 프로그램 실행 및 첫 화면 출력
<p align="center">
  <img src="https://user-images.githubusercontent.com/74342121/215770940-3ec1f822-cbd6-47e3-8094-ed864b1b1557.png" width="500" height="460">
</p>

```
데이터에 대한 기본정보를 사용자에게 제공하기 위해서 데이터에 관해 표현할 수 있는 기본 구조가 제공되어 있다.
```

### 1-1. 메인 화면 - 파일 불러오기 수행 시
<p align="center">
  <img src="https://user-images.githubusercontent.com/74342121/215771542-1189fb7b-2afa-430e-9fac-eec63cd5f6fe.png" width="500" height="460">
</p>

```
데이터를 불러온 후, 데이터에 대한 기본정보를 보여주는 화면이다. 데이터프레임을 통해 .csv 파일 내에 존재하는 정보들을 테이블 형태로 
표시해주며 결과 값에 대한 유형들과 샘플의 개수 등이 표시된다.
```

### 2. 데이터 분석 - 초기화면
<p align="center">
  <img src="https://user-images.githubusercontent.com/74342121/215771954-162ec241-dacc-44ad-9ca3-98997001e076.png" width="500" height="400">
</p>

```
데이터 분석의 초기화면이다. 총 5단계로 데이터 분석이 이루어진다. 단계별로 버튼이 활성화되며 순차적으로 누르게 되면 필요 없는 
feature들은 자동으로 제거되며 최종 데이터 분석 시작 버튼을 통해 데이터 분석 결과를 알아볼 수 있다.
```

### 2-1. 데이터 분석 - 데이터 분석 결과
<p align="center">
  <img src="https://user-images.githubusercontent.com/74342121/215773050-00e99b7d-81fc-4b05-9ebd-6f16b7c9f51d.png" width="500" height="400">
</p>

```
최종 데이터 분석 결과를 보여주는 화면이다. 100개의 feature 중 데이터 분석 과정을 통해 최종적으로 남은 feature들의 목록을 
보여주며 양품과 불량에 대한 범위, 서로 중복되지 않는 값들, 최빈값 등을 보여준다. 이를 통해 각각의 분포도를 알 수 있으며 그래프의
종류 변경을 통해 다양한 관점에서 분석이 가능하다.
```

### 3. 데이터 시각화(1D)
<p align="center">
  <img src="https://user-images.githubusercontent.com/74342121/215773479-c0421a7d-f99b-417b-8fff-51ef96ac5e0c.png" width="500" height="400">
</p>

```
전체 데이터에 대하여 사용자가 선택한 feature에 해당하는 데이터만을 그래프로 시각화하여 보여주는 화면이다. 
그래프의 유형은 약 7가지가 있으며 다양한 관점에서 전체 데이터를 시각화하여 볼 수 있도록 제공한다.
```

### 4. 데이터 시각화(2D)
<p align="center">
  <img src="https://user-images.githubusercontent.com/74342121/215773774-b5935006-580d-403d-951c-bf5a049063e4.png" width="500" height="400">
</p>

```
사용자가 2가지 feature와 그래프 유형을 선택하면 그에 관한 상관관계를 분석할 수 있도록 제공해 주는 기능이다. 
데이터 시각화(1D)와 달리 2가지 feature를 선택하여야 하는 차이가 있으며 그래프의 유형도 1D와는 다르기 때문에 
다양한 방법으로 분석이 가능하도록 한다.
```

### 5. 학습 및 예측
<p align="center">
  <img src="https://user-images.githubusercontent.com/74342121/215774109-6e0d5cce-b715-475a-aca0-3c789773a155.png" width="500" height="400">
</p>

```
내부 구현되어 있는 딥러닝 모델에게 학습시킬 데이터를 로드한 후, 학습 시작을 클릭하게 되면 모델이 양/불에 대한 
Anomaly Detection 과정을 수행한다. 그 후, 결과를 성능 지표를 통해 사용자에게 보여준다. 
학습 데이터와 테스트 데이터의 비율은 7:3 으로 나누어 진행하였다.

※ 아직 실험 단계에 있으며 추후 세부적인 딥러닝 모델 개발과 함께 GUI 기능을 추가할 예정 
```

<br/>

## Conclusion
- 기업 연계형 프로젝트를 통해 사출기 결과물의 양불 판정을 위한 이상치 검출을 위한 GUI 분석 프로그램 개발을 개발하였다. 
- 이러한 분석 프로그램을 통해 다양한 관점에서 이상 데이터를 분석을 가능토록 하였다.(데이터 시각화, 통계적 검정 분석 등) 
- 추후 <strong>양/불 판정을 위한 이상치 검출 모델 정확도를 향상</strong>시키는 데에 노력을 기울일 예정이다. 이는 이전의 [딥러닝 프로젝트](https://github.com/Min-su-Jeong/Deep_Learning_Project)와 연계하여 연구를 진행할 예정이다.

<br/>

## Paper
- J. Jeong, M. Jeong, J. Si and S. Kim, "Development of Data Analysis System for Determining Product Quaility of Injection Molding Machine Results", Proc. Of Korean Institute of Information Technology Conference, pp.329-330, Dec. 2022.
- [Paper Link](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11183818)

- J. Si, J. Jeong, M, Jeong, and S. Kim, "Anomaly Detection of Injection Molding Using Statistics-Based Feature Selection and Generative Adversarial Learning", Journal of Korean Institute of Information Technology(JKIIT), Accepted
- [Journal Link](http://ki-it.com/_common/do.php?a=current&b=21&bidx=3274&aidx=36440)
