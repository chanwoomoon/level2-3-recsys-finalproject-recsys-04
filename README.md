# 모델링 개요

Recsys-04 Final Project
# 🎁GiftHub
# Team
| **김세훈** | **문찬우** | **김시윤** | **배건우** | **이승준** |
| :------: |  :------: | :------: | :------: | :------: |
| [<img src="https://avatars.githubusercontent.com/u/8871767?v=4" height=150 width=150>](https://github.com/warpfence) | [<img src="https://avatars.githubusercontent.com/u/95879995?v=4" height=150 width=150> ](https://github.com/chanwoomoon) | [<img src="https://avatars.githubusercontent.com/u/68991530?v=4" height=150 width=150> ](https://github.com/tldbs5026) | [<img src="https://avatars.githubusercontent.com/u/83867930?v=4" height=150 width=150>](https://github.com/gunwoof) | [<img src="https://avatars.githubusercontent.com/u/133944361?v=4" height=150 width=150>](https://github.com/llseungjun) |



# 1. 1st_stage
## lgbm
- backend를 위해 refactoring이 진행된 폴더


## LightGBM
LGBM.py
- 사용자의 input을 받아 후보군을 생성하는 단계입니다.
- lgbm을 이용하여 product_id(후보군)을 추론하는 것이 목적입니다.

boosting_gender_inf.ipynb
- 일부 카테고리에서 성별 및 연령 그룹이 있기에 이를 활용하여 나머지 카테고리에 적용하고자 하였으나 성별-연령 예측 모델의 성능이 좋지 않아 사용하지 않았습니다.

lgbm_gpu.md
- lgbm 라이브러리를 gpu로 학습하기 위한 설정에 대한 markdown입니다.

# 2. 2nd_stage

## bert4rec
- bert4rec 모델을 이용하여 아이템 기반 협업 필터링 추천을 수행하기 위한 폴더입니다.

## EASE
- EASE 모델을 이용하여 아이템 기반 협업 필터링 추천을 수행하기 위한 폴더입니다.

## LightGCN
lightgcn.ipynb
- LightGCN 모델을 이용하여 아이템 기반 협업 필터링 추천을 수행하기 위한 폴더입니다.

LightGCN_Refactoring
- LightGCN의 결과를 리팩토링하여 백엔드에서 사용하기 용이하게 변경하였습니다.
- 학습할 파일에 대한 경로 등의 설정은 main.py에서 수정할 수 있습니다.
- 실행방법

```
cd modeling/2nd_stage/LightGCN/LightGCN_Refactoring

# 재학습
python main.py --train

# 아이템 기반 추천
python main.py --rec='item'
# 유저 기반 추천
python main.py --rec='user' 

```

## WDN_notused
- 네이버 데이터를 바탕으로 product_id를 도출하기 위해 개발하였습니다.
- 프로젝트의 task와 맞지 않아 최종적으로는 사용하지 않았습니다.


# 3. Recbole
- recbole을 이용하여 모델의 성능을 실험하고자 사용하였습니다.

## recbole_
gender_inter.py
- 1st stage에서 성별을 추론하기 위한 모델링 데이터를 만들기 위한 파일입니다.
- 최종적으로 모델의 성능이 좋지않아 사용하지 않았습니다.

make_inter_amazon.py
- 2nd stage에서 아마존 데이터에 대한 모델의 성능을 측정하기 위해 interaction파일을 만드는 python 파일입니다.
- lightgcn, ease, recvae, mf등의 모델에 사용하여 성능을 측정한 후, 이에 대한 모델을 구현해서 모델의 재사용성 및 확장성을 고려하였습니다.
- 실행방법

```
python make_inter_amazon.py
```

requirements.txt
- recbole을 이용하기 위한 최소 requirements가 저장되어있습니다.


## recbole.ipynb
- 모델의 성능을 주피터로 확인할 수 있도록 가벼운 코드로 재구성 하였습니다.