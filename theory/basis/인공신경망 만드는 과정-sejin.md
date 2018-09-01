#### 인공지능

input과 output(학습데이터셋)을 통해 함수를 자동으로 구하는 것

#### 딥러닝 모델 선택하기

https://ratsgo.github.io/deep%20learning/2017/04/22/NNtricks/

1. **적절한 네트워크 선택**

   1) **구조(structure)**: Single words vs Bag of Words, etc.

   2) **활성함수(비선형성(nonlinearity) 획득 방법)**: ReLu vs tanh, etc.

2. **그래디언트 체크**: 네트워크를 구축했는데 그래디언트 계산이 혹시 잘못될 염려가 있으므로 잘됐는지 체크해봅니다

3. **학습 파라미터 초기화**: 초기화 방법에도 여러가지가 있으므로 적절히 선택합니다

4. **학습 파라미터 최적화**: Stochastic Gradient vs Adam, etc.

5. **과적합 방지**: dropout vs regularize, etc.

#### 활성함수

* 정의

  * ~~뉴럴네트워크의 개별 뉴런에 들어오는 입력신호의 총합을 출력신호로 변환하는 함수~~
  * 유의미한 값만을 남기는 함수
  * 
  * 비선형 함수를 써야 한다.
    * 선형 함수(그래프가 직선)는 레이어를 쌓아봐야 아무 이득이 없다. 따라서 활성함수로는 비선형 함수를 써야 한다.
    * 비선형 함수 !== 초월함수이다. ex) $y=x^2$도 비선형함수다.
  * 퍼셉트론과 인공신경망의 차이 = 활성함수의 유무

* 종류

  * sigmoid
    - 출력범위가 0 이상 1 이하인 함수
    - $\sigma(x) = {1 \over {1+e^{-x}}}$
  * hyperbolic tangent ($\tanh$)
    - -1 초과 1 미만의 값을 가진다.
    - hyperbolic 삼각함수
      - 정의
        - 우리가 아는 보통의 삼각함수는 원을 기준으로 정의된다.
        - 얘는 쌍곡선을 기준으로 정의된다.

  * ReLU
  * Leaky ReLU
  * Exponential Linear Units (ELU)
  * Maxout Neurons (MN)

#### 학습 parameter (= weight)

만약 가설함수($h(x)$)가 $ax+b$라고 하면 $a$와 $b$를 구하는 과정

(sum(h(input data set) - output data set))^2이 최소화되게 $a$와 $b$를 감소시켜야 한다. 이 최소화돼야하는 합을 **loss function**이라 한다.

**loss function**의 최솟값을 **graident descent**를 이용한다.

#### Gradient Descent

현재 가설함수에 대한 loss function의 각 파라미터의 편미분값을 가지고 현재 가설함수의 각 파라미터가 커져야 하는지 작아져야 하는지를 알아낼 수 있다.

이를 기반으로 각 파라미터의 값을 조금씩(learning rate 값에 따라) 조정한다.

#### one-hot-vector

- 원소가 하나만 1이고 나머지는 0인 벡터
- 선풍기 버튼 생각하면 된다.