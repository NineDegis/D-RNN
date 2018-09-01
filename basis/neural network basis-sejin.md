 [밑바닥부터 시작하는 딥러닝](http://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)이란 책이 친절하니 이걸 보면 좋을 것 같다(용하는 이미 봤다).

## Perceptron

인공신경망을 이해하기 위해서는 먼저 그 조상격인 퍼셉트론 네트워크에 대해 이해할 필요가 있다. 퍼셉트론 네트워크는 이름 그대로 퍼셉트론으로 구성된 네트워크를 의미한다. 단일 퍼셉트론은 다음 그림과 같이 생겼다.

|<img src="https://raw.githubusercontent.com/3jin/Images/master/perceptron.png" width="50%"/>|
|:--:|
|출처: [http://neuralnetworksanddeeplearning.com/chap1.html](http://neuralnetworksanddeeplearning.com/chap1.html)|

퍼셉트론은 **여러 binary 입력을 받아 0 또는 1을 출력하는 시스템**이다. 퍼셉트론마다 **threshold** 값을 가지고 있어서 입력의 합이 threshold를 넘으면 1을, 넘지 못하면 0을 출력한다.

예를 들어보자. 내가 1교시 수업을 가야 하는데 아침에 일어나니 배가 아프다고 하자. 참고 그냥 등교할지(0) 또는 처리하고 등교할지(1)는 다음과 같은 요소들을 고려해서 결정해야 할 것이다.

1. 시간적 여유가 많은가?
2. 이번에 지각해도 성적이 F로 떨어지지 않는가?
3. 배 아픈 정도가 참을만한가?

threshold가 1이라고 하자. 그럼 1~3 항목 중 2개 이상이 충족되면 싸고 등교하는거다. 하지만 곰곰히 생각해보면 이렇게 단순히 결정할 문제는 아니다. 예를 들어 시간적 여유가 많다면($x_1=1$), 지각 한 번만 더 하면 F를 받고($x_2 = 0$) 그렇게 급하지도 않다 하더라도($x_3 = 0$) 굳이 참고 등교할 필요가 없다. 이 때문에 나온 개념이 **가중치(weight)**다. 고려요소들의 중요도에 따라 결정되는 값이다. 

가중치까지를 고려해서 퍼셉트론을 수식으로 나타내게 되면 다음과 같다.
$$
\begin{eqnarray} \mbox{output} & = & \left\{ \begin{array}{ll} 0 & \mbox{if } \sum_j w_j x_j \leq \mbox{ threshold} \\ 1 & \mbox{if } \sum_j w_j x_j > \mbox{ threshold} \end{array} \right. \end{eqnarray}
$$

### Perceptron Network

|<img src="https://raw.githubusercontent.com/3jin/Images/master/perceptron-network.png" width="80%"/>|
|:--:|
|출처: [http://neuralnetworksanddeeplearning.com/chap1.html](http://neuralnetworksanddeeplearning.com/chap1.html)|

위 그림처럼 퍼셉트론들로 네트워크를 만들면 좀 더 미묘한 결정도 내릴 수 있게 된다. 첫번째 계층에서 세 요소에 대한 결정을 내린 뒤 그 결정 결과를 바탕으로 다시 다른 결정을 내려 결론에 도달하는 식이다. 이 퍼셉트론 네트워크를 이용하면 가산기와 같은 논리회로를 만드는 게 가능하다.

퍼셉트론 하나로 weight와 threshold를 적당히 조작하면($w_1 = -2, w_2 = -2, \mathrm{threshold}=3$) NAND 게이트를 만들 수 있다. [모든 논리함수는 NAND 게이트만으로 구현할 수 있으므로](http://www.ktword.co.kr/abbr_view.php?m_temp1=4560) 가산기와 같은 논리회로를 다음과 같이 구성하는 게 가능하다.

|<img src="https://raw.githubusercontent.com/3jin/Images/master/perceptron-nand.png" width="80%"/>|
|:--:|
|출처: [http://neuralnetworksanddeeplearning.com/chap1.html](http://neuralnetworksanddeeplearning.com/chap1.html)|

## Neural Network

논리회로는 원래 입출력이 0, 1밖에 없으므로 퍼셉트론만 가지고도 원하는 결과를 만들 수 있다. 하지만 현실의 복잡한 문제를 해결하려면 퍼셉트론의 한계가 드러나게 된다.

Perceptron network에서 weight와 bias를 처음에 랜덤하게 주고 입력을 넣어보면 출력이 맞게 나오는 것도 조금 있고, 대부분은 기대한 값과 다른 출력을 뱉을 것이다. 여기서 weight와 bias를 조금씩 조작해서 다시 입력을 넣다 보면 출력값들이 어느 순간 기대한 값들에 가까워질 것이다. 그런데, 퍼셉트론은 이진값만을 뱉기 때문에 그 변화가 불연속적이고 예측하기 어렵다. 이를 해결하기 위해 **활성 함수**라는 개념이 도입된다.

### 활성 함수 (Activation Function)

활성 함수란, 가중치 * 입력 값에 어떤 변형을 가해서 최종적으로 해당 뉴런을 활성화할지 말지를 결정하는 함수다. Perceptron은 활성함수가 계단함수인 뉴런이라고 보면 된다. 계단함수는 다음과 같다.
$$
h(x) = \left\{ \begin{array}{} 0 & \mathrm{if} \;\, x \leq \mathrm{threshold} \\ 1 & \mathrm{if} \;\, x > \mathrm{threshold} \end{array} \right.
$$
아까 threshold값을 넘으면 똥을 싸고 등교하고 넘지 못하면 참고 등교한다고 했는데, 이 'threshold를 넘느냐 마느냐'가 perceptron의 활성함수인 계단함수다.

계단함수 외에 활성함수는 다음과 같이 여러가지가 존재한다.

* **sigmoid**
  $$
  \sigma(x) \equiv \frac{1}{1+e^{-x}}
  $$
  가장 유명한 활성함수다. 보통 $\sigma(x)$와 같이 표현한다. 그래프는 다음과 같이 $\sigma : [-\infin, \infin] \rarr (0, 1)$으로 그려진다.

  | <img src="https://raw.githubusercontent.com/3jin/Images/master/sigmoid-graph.png" width="80%"/> |
  | :----------------------------------------------------------: |
  | 출처: [위키백과](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png) |

  값이 부드럽게 변하고, 0~1의 범위를 갖기 때문에 활성함수로 쓰기에 아주 좋다.

* **hyperbolic tangent**
  $$
  \tanh(x) \equiv {{\sinh(x)} \over {\cosh(x)}} = {{e^{2x} - 1} \over {e^{2x} + 1}}
  $$
  우리의 졸업작품을 위해 쓰일 RNN에서 주로 사용하는 활성함수다. 

  | <img src="https://raw.githubusercontent.com/3jin/Images/master/tanh-graph.gif" width="80%"/> |
  | :----------------------------------------------------------: |
  | 출처: [Wolfram MathWorld](http://mathworld.wolfram.com/images/interactive/TanhReal.gif) |

  -1~1의 범위를 갖는다.

* **ReLU**

  나는 크게 안 궁금해서 얘는 생략하겠다. 필요하신 분들은 구글링 ㄱㄱ.

### Layers

뉴럴 네트워크는 다음 그림처럼 뉴런들을 input layer, hidden layer, output layer로 구분한다. Hidden layer는 뭔가 심오한 뜻이 있는 건 아니고, input layer나 output layer가 아닌 layer라는 뜻이다.

| <img src="https://raw.githubusercontent.com/3jin/Images/master/neural-network-layers.png" width="80%"/> |
| :----------------------------------------------------------: |
| 출처: [https://github.com/drewnoff/spark-notebook-ml-labs/tree/master/labs/DLFramework](https://github.com/drewnoff/spark-notebook-ml-labs/tree/master/labs/DLFramework) |

