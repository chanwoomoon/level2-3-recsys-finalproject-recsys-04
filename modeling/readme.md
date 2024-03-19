cateigry_eda
- 1차로 추천을 하기위한 lgbmclassifier입니다.
- 여기에서는 filtering을 진행하여 product_id의 unique를 줄임 + stratified sampling을 진행하였습니다.
- sweep의 결과 lr가 모델에 성능에 큰 영향을 미치는 것으로 나타났습니다.
- 개성을 추가해서 추가적인 학습을 진행해보는 것도 필요합니다.
- topk recommendation 관련해서 추가적인 코드 수정이 필요합니다.


wdn
- 2차로 최종적으로 product_id를 예측하기 위한 모델링입니다.
- 개성을 추가하였습니다.
    - category_eda에 이를 추가해서 다시 진행해봐야합니다. 
- Wide Deep Neural network의 구조를 사용하였고, 이를 데이터의 구조에 맞게 재구성하였습니다.
- categorical data는 deep 구조에 넣었으며, continuous data는 wide 구조에 넣어 side information으로 사용하도록 하였습니다.

03.10
- 학습을 하는데 성능 개선중입니다.
- topk recommendation 관련해서 추가적인 코드의 수정이 필요합니다.