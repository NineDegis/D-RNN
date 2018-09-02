# RNN



### RNN의 recurrent 뜻은?

* 반복적인, 되풀이되는 등의 뜻.
### RNN 이란?

* 기존까지의 피드포워드 신경망들은 모든 노드를 한번씩만 지나가기 때문에 시간의 개념을 제외하고 독립적으로 학습한다(현재 들어온 데이터만 고려한다).
* 순한 신경망(RNN)은 반복적인 데이터(순차적인 데이터)를 학습하는데 특화되어 발전된 인공신경망(ANN)의 한 방식이다.
* 과거의 데이터가 미래에 영항을 줄 수 있는 구조.



### RNN의 기본구조는?

[![source: imgur.com](http://i.imgur.com/s8nYcww.png)](http://imgur.com/s8nYcww)

 

### LSTM(Long Short-term Memory models)란?

* RNN이 해결하지 못하는 장기의존성 문제를 해결한 RNN의 종류
* LSTM은 RNN으로 성취할 수 있는 하나의 빅스텝

### forward, backward compute pass란?

- 순전파, 역전파랑 같은 말인듯.

### GRU(Gated Recureent Unit)이란?

* LSTM을 변형시킨 알고리즘으로, Gradient Vanishing 문제를 해결함.
* LSTM과 달리 초기 weight를 지속적으로 업데이트 하지 않고 어떻게 반영할 것인지 결정함.

### hidden state란?

* 네트워크의 "메모리" 부분으로서, 이전 시간 스텝의 hidden state 값과 현재 시간 스텝의 입력값에 의해 계산.


### 방향을 가진 엣지란?

* 히든 노드들이 순환구조를 이루기 위해 엣지가 단방향으로 연결 되었다는 듯.


### 순환구조(directed cycle)란?

* 이전 작업을 현재 작업과 연결한다는 의미

* hidden layer 데이터를 저장하고 있다.

* 하나의 순환은 시간 단위로 여러개로 펼쳐져 있다고 생각할 수 있음.


### 인공 신경망(ANN)이란?

* 생물학의 신경망에서 영감을 얻는 학습 알고리즘.
* 시냅스의 결합으로 네트워크를 형성한 인공 뉴런이 학습을 통해 결합 세기를 변화시켜 문제해결 능력을 가지는 비선형 모델.




### CNN(Convolutional Neural Networks)란?

- 딥러닝의 한 종류로 주로 이미지를 인식하는데 사용.

  

### 활성함수(activation function)이란?

* 입력 신호의 총합을 그대로 사용하지 않고, 출력 신호로 변환하는 함수.

* 입력 신호의 총합이 활성화를 일으키는지 아닌지를 정하는 역할.



### one-hot-vector 란?

* 해당하는 칸의 정보를 1로 표시하고 나머지는 0으로 표시하는 방법.
<img src="https://raw.githubusercontent.com/skrudtn/images/master/D-RNN/one-hot-encoding.PNG" width="50%"/>


### 순전파(foward propagation)란?

![source: imgur.com](http://i.imgur.com/TIdBDTJ.png)

* input layer에서 output layer까지 갱신하는 과정.

### 역전파(back propagation)란?

* RNN이 필요로하는 정답을 알려줘 모델이 parameter를 갱신해내가는 과정.

### 최종 Loss란?

* 순전파를 끝냈을 때 나온 손실 함수

### 그래디언트란?

* 공간에 대한 기울기
* 공간에서 변화가 가장 큰 방향
* gradient의 방향은 함수값이 커지는 방향.
* NN에서 그래디언트의 역할이 뭘까??

### 로컬 그래디언트란?

* 각 단계의 그래디언트인가



### Vanishing gardient problem이란?

* 역전파시 그래디언트가 점차 줄어 학습능력이 크게 저하되는 현상.

* 파라미터가 학습 안된다.

  

### cell-state란?

* LSTM의 전체 체인을 관통하며 필요한 정보를 다음 단계로 넘기는 역할

### forget gate란?

* LSTM 셀 스테이트 에서 어떤 정보를 버릴지 선택하는 게이트.

### 계단함수란?

- x가 0보다 크면 1 아니면 0을 반환하는 함수
- y=x>0?1:0



### sigmoid 함수란?

<img src="https://raw.githubusercontent.com/skrudtn/images/master/D-RNN/sigmoid.PNG" width="30%"/>

<img src="https://raw.githubusercontent.com/skrudtn/images/master/D-RNN/sigmoid-graph.PNG" width="30%"/>
* 0 ~ 1 사이의 값을 갖는다.
### 하이퍼볼릭탄젠트(tanh)란?
* sigmoid와 유사하게 생겼지만 -1 ~ 1 사이의 값을 갖는다.

### ReLU 함수란?

* 입력이 0을 넘으면 그 입력을 그대로 출력.
* 0 이하이면 0을 출력
<img src="https://raw.githubusercontent.com/skrudtn/images/master/D-RNN/ReLu.PNG" width="30%"/>



### LSTM이란?

* RNN의 hidden state에 cell-state를 추가한 구조이다.
* vanishing gradient problem을 극복하기 위해 고안된 것.

### input gate란?

* 셀 스테이트에서 어떤 값을 업데이트할 지를 결정하는 게이트.

### LSTM의 순전파란?

* RNN과 같이 Ht를 구한 다음, 행 기준으로 i, f, o, g로 4등분해 각각에 해당하는 활성함수를 적용하여 i,f,o,g를 구함.

### i, f, o, g 각각에해당하는 활성함수란?

* sigmoid, sigmoid, sigmoid, tanh

