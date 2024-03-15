<template>
    <section class="bg-gray-50 text-center">
        <div class="mx-auto max-w-screen-xl px-4 py-32 lg:flex lg:h-screen lg:items-center">
            <div class="mx-auto">
                <div class="flex flex-wrap justify-center gap-4 items-center text-center">
                    <div class="mr-20"> <!--테이블 layout-->
                        <h1 class="text-3xl font-extrabold sm:text-5xl" style="font-size: 20px;">상품이 마음에 들었다면 좋아요를 눌러주세요.</h1>
                        <table class="mt-8">
                            <tr v-for="(row, rowIndex) in Math.ceil(productListToShow.length / 3)" :key="rowIndex">
                            <td  v-for="(column, columnIndex) in 3" :key="columnIndex" class="p-4 items-center text-center">
                                <template v-if="productListToShow[(row - 1) * 3 + columnIndex]">
                                <img :src="productListToShow[(row - 1) * 3 + columnIndex].image_url" alt="" height="100" width="100" 
                                @click="handleProductClick(productListToShow[(row - 1) * 3 + columnIndex],'interaction')" />
                                <!-- <p class="mt-2 whitespace-wrap w-full text-center" style="font-size: 5px;">{{productListToShow[(row - 1) * 3 + columnIndex].product_name}}</p> -->
                                </template>
                            </td>
                            </tr>
                        </table>
                        <div class="mt-4">
                            <button @click="showNextProducts()" class="rounded-md bg-white px-3.5 py-2.5 text-sm font-semibold text-gray-900 shadow-sm hover:bg-gray-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white">다음 상품 확인하기</button>
                            <button @click="getPrediction()" class="rounded-md bg-white px-3.5 py-2.5 text-sm font-semibold text-gray-900 shadow-sm hover:bg-gray-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white">선물 추천 바로받기</button>
                        </div>
                    </div>
                    <div class="w-1/2 h-1/2"> <!--carousel 효과 적용하는 block, 해당 div에 layout 옵션 추가-->
                        <h1 class="text-3xl font-extrabold sm:text-5xl" style="font-size: 20px;">비슷한 사람들이 많이 구매한 상품</h1>
                        <Carousel :itemsToShow="3.95" :wrapAround="true" :transition="500">
                            <Slide v-for="(item, index) in predictionList" :key="index">
                                <div class="carousel__item">
                                    <img :src="item.image_url" 
                                    @click="handleProductClick(item,'like')"/>
                                    <!-- <p>{{item.product_name}}</p> -->
                                </div>
                            </Slide>
                            
                            <template #addons>
                                <Navigation />
                                <Pagination />
                            </template>
                        </Carousel><br>
                        <Carousel :itemsToShow="3.95" :wrapAround="true" :transition="500">
                            <Slide v-for="slide in 10" :key="slide">
                            <div class="carousel__item">
                                <img :src="'https://shop-phinf.pstatic.net/20230308_89/1678249882534ecdzU_JPEG/79385717218304917_1496244196.jpg?type=f480_480'" />
                            </div>
                            </Slide>
                    
                            <template #addons>
                                <Navigation />
                                <Pagination />
                            </template>
                        </Carousel>
                    </div>
                </div>
            </div>
        </div>
    </section>
</template>

<script>
import { Carousel, Navigation, Pagination, Slide } from 'vue3-carousel'
import { ref, defineComponent, onMounted } from 'vue';
import { userInfoStore } from '../store/index';
import axios from 'axios';
import 'vue3-carousel/dist/carousel.css'

export default defineComponent({
    components: {
        Carousel,
        Slide,
        Pagination,
        Navigation,
    },
    setup() {
        //product
        const productList = ref([]);
        const productListToShow = ref([]);
        //flag 변수
        const phaseIndex = ref(0);
        const isClicked = ref([]);
        //predict 결과
        const predictionList = ref([]);
        //store
        const store = userInfoStore();

        //get으로 상품 27개 가져오기
        const getList = async () => {
            try {
                const user_id = store.getDataAll.user_id
                const response = await axios.get(`/api/user/matched-items/${user_id}`);
                productList.value = response.data;
                store.matchedItems.value = productList.value;
                console.log('productList', productList.value);
                showNextProducts();
                setPredictionList();
            } catch (error) {
                console.error('데이터를 가져오는 중에 오류가 발생했습니다:', error);
            }
        };

        //get으로 prediction 결과 10개 가져오기
        const getPrediction = async() => {
            try {
                const user_id = store.getDataAll.user_id
                const response = await axios.get(`/api/user/items-prediction/${user_id}`);
                predictionList.value = response.data;
                console.log('predictionList', predictionList.value);
            } catch (error) {
                console.error('데이터를 가져오는 중에 오류가 발생했습니다:', error);
            }
        };

        //상품 9개를 보여주는 테이블에서 다음 9개의 값을 보여주는 함수
        const showNextProducts = async() => {
            const startIndex = (phaseIndex.value % 3) * 9; // 각 페이즈의 시작 인덱스 계산
            productListToShow.value = productList.value.slice(startIndex, startIndex + 9);
            phaseIndex.value++;
        };

        //predictionList 초기화 함수
        const setPredictionList = async() => {
            predictionList.value = [{ image_url: 'https://shop-phinf.pstatic.net/20230308_89/1678249882534ecdzU_JPEG/79385717218304917_1496244196.jpg?type=f480_480' }];
            console.log('predictionList', predictionList.value);
        };

        //상품 중 고른 상품에 대해 POST와 DELETE 요청을 처리하는 함수
        const handleProductClick = async (product, type) => { //type : 'like' or 'interation'
            const user_id = store.getDataAll.user_id
            if (!isClicked.value.includes(product.product_id)) {
                try {
                // POST 요청을 보내는 부분
                const response = await axios.post(`/api/user/${type}/`, {
                    "user_id": user_id,
                    "product_id": product.product_id,
                });
                isClicked.value.push(product.product_id);
                console.log('POST request successful:',isClicked.value);
                } catch (error) {
                console.error('Error sending POST request:', error);
                }
            } else {
                try {
                // DELETE 요청을 보내는 부분
                await axios.delete(`/api/user/${type}/`, {
                    data: {
                        "user_id": user_id,
                        "product_id": product.product_id,
                    },
                });
                isClicked.value.splice(isClicked.value.indexOf(product.product_id), 1);
                console.log('DELETE request successful:',isClicked.value);
                } catch (error) {
                console.error('Error sending DELETE request:', error);
                }
            }
        };

        //9개 이미지 랜더링 hook
        onMounted(getList);

        return {
        predictionList,
        productListToShow,
        showNextProducts,
        setPredictionList,
        handleProductClick,
        getPrediction,
        };
    }
});
</script>

<style>
.carousel__item {
    min-height: 200px;
    width: 100%;
    background-color: var(--vc-clr-primary);
    color: var(--vc-clr-white);
    font-size: 20px;
    border-radius: 8px;
    display: flex;
    justify-content: center;
    align-items: center;
}
  
.carousel__slide {
    padding: 5px;
}

.carousel__viewport {
  perspective: 2000px;
}
.carousel__track {
  transform-style: preserve-3d;
}
.carousel__slide--sliding {
  transition: 0.5s;
}
.carousel__slide--active ~ .carousel__slide {
  transform: rotateY(20deg) scale(0.9);
}

.carousel__slide--prev {
  opacity: 1;
  transform: rotateY(-10deg) scale(0.95);
}

.carousel__slide--next {
  opacity: 1;
  transform: rotateY(10deg) scale(0.95);
}

.carousel__slide--active {
  opacity: 1;
  transform: rotateY(0) scale(1.1);
}
.carousel__prev,
.carousel__next {
    box-sizing: content-box;
    border: 5px solid white;
}
</style>


