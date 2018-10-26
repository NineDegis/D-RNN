# How to configure.

`checkpoints` 라는 이름의 디렉토리가 src root directory 에 생성되어 있어야 합니다. 
참고로 기왕이면 src directory 를 src root 로 지정하는 편이 편합니다. 안그러면 코드를 조금 수정해야 할 거에요.

# How to train.

    $ python3 train.py





# How to run Tensorboard

```
$ tensorboard --logdir=./logs --port=9898
```

텐서보드 전용 포트번호로 9889번을 할당 받았기 때문에  해당 포드로만 접근 가능합니다.



```
117.16.136.52:9898
```

위 주소로 접근하면 tensorboard 그래프를 확인할 수 있습니다.

