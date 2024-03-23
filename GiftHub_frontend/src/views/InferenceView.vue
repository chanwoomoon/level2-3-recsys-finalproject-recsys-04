<template>
    <section class="bg-gray-50 text-center">
        <div v-if ="isLoading" class="flex justify-center items-center h-screen">
            <div class="rounded-full h-20 w-20 bg-violet-800 animate-ping"></div>
        </div>
        <div v-if ="!isLoading" class="mx-auto max-w-screen-xl px-4 py-32 lg:flex lg:h-screen lg:items-center">
            <div class="mx-auto">
                <div class="flex flex-wrap justify-center gap-4 items-center text-center">
                    <div class="mr-20">
                        <h1 class="text-3xl font-extrabold sm:text-5xl" style="font-size: 20px;">상품이 마음에 들었다면 좋아요를 눌러주세요.</h1>
                        <table class="mt-4 "> 
                            <tr v-for="(row, rowIndex) in Math.ceil(amazonProductListToShow.length / 3)" :key="rowIndex">
                                <td  v-for="(column, columnIndex) in 3" :key="columnIndex" class="p-4 items-center text-center relative">
                                    <template v-if="amazonProductListToShow[(row - 1) * 3 + columnIndex]">
                                        <!-- 이미지 -->
                                        <img :class="amazonButton === 0 ? 'blur-lg' : ''" :src="amazonProductListToShow[(row - 1) * 3 + columnIndex].image_url" alt="" height="80" width="80" 
                                            @click="handleProductClick(amazonProductListToShow[(row - 1) * 3 + columnIndex],'interaction','amazon')" />
                                        <button v-if = "(amazonButton != 0) & amazonIsClicked.includes(amazonProductListToShow[(row - 1) * 3 + columnIndex].product_id)">                
                                            <svg class="text-red-400 w-6 h-auto fill-current absolute top-0 right-0  " xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
                                                <path d="M0 190.9V185.1C0 115.2 50.52 55.58 119.4 44.1C164.1 36.51 211.4 51.37 244 84.02L256 96L267.1 84.02C300.6 51.37 347 36.51 392.6 44.1C461.5 55.58 512 115.2 512 185.1V190.9C512 232.4 494.8 272.1 464.4 300.4L283.7 469.1C276.2 476.1 266.3 480 256 480C245.7 480 235.8 476.1 228.3 469.1L47.59 300.4C17.23 272.1 .0003 232.4 .0003 190.9L0 190.9z"/>
                                            </svg>
                                        </button>
                                        <button v-if = "amazonButton != 0" @click="handleProductClick(amazonProductListToShow[(row - 1) * 3 + columnIndex],'interaction','amazon')">
                                            <svg class="text-red-400 w-6 h-auto fill-current absolute top-0 right-0" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
                                                <path d="M244 84L255.1 96L267.1 84.02C300.6 51.37 347 36.51 392.6 44.1C461.5 55.58 512 115.2 512 185.1V190.9C512 232.4 494.8 272.1 464.4 300.4L283.7 469.1C276.2 476.1 266.3 480 256 480C245.7 480 235.8 476.1 228.3 469.1L47.59 300.4C17.23 272.1 0 232.4 0 190.9V185.1C0 115.2 50.52 55.58 119.4 44.1C164.1 36.51 211.4 51.37 244 84C243.1 84 244 84.01 244 84L244 84zM255.1 163.9L210.1 117.1C188.4 96.28 157.6 86.4 127.3 91.44C81.55 99.07 48 138.7 48 185.1V190.9C48 219.1 59.71 246.1 80.34 265.3L256 429.3L431.7 265.3C452.3 246.1 464 219.1 464 190.9V185.1C464 138.7 430.4 99.07 384.7 91.44C354.4 86.4 323.6 96.28 301.9 117.1L255.1 163.9z"/>
                                            </svg>
                                        </button>
                                        <!-- 버튼 -->
                                        <button v-if="((row - 1) * 3 + columnIndex === 4) &(amazonButton === 0)"  @click="amazonTrigger()"
                                            class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-blue-500 text-white py-2 px-4 rounded-md">
                                            버튼
                                        </button>
                                    </template>
                                </td>
                            </tr>
                        </table>
                        <div class="mt-4">
                            <button :class="amazonButton === 0 ? 'blur-lg' : ''" @click="showNextProducts()" class="rounded-md bg-white px-3.5 py-2.5 text-sm font-semibold text-gray-900 shadow-sm hover:bg-gray-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white">다음 상품 확인하기</button>
                            <button :class="amazonButton === 0 ? 'blur-lg' : ''" @click="getPrediction('amazon','lightgcn')" class="rounded-md bg-white px-3.5 py-2.5 text-sm font-semibold text-gray-900 shadow-sm hover:bg-gray-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white">선물 추천 바로받기</button>
                        </div>
                    </div>
                    <div v-if="amazonButton === 0" class="w-1/2 h-1/2"> <!--carousel 효과 적용하는 block, 해당 div에 layout 옵션 추가-->
                        <h1 class="text-3xl font-extrabold sm:text-5xl" style="font-size: 20px;">인기도 기반 상품 추천</h1>
                        <Carousel class="mt-4" :itemsToShow="3.95" :wrapAround="true" :transition="500">
                            <Slide v-for="(item, index) in productList" :key="index">
                                <div class="carousel__item">
                                    <img :src="item.image_url" 
                                    @click="handleProductClick(item,'like','naver')"/>
                                    <button v-if="isClicked.includes(item.product_id)">                
                                        <svg class="text-red-400 w-6 h-auto fill-current absolute top-0 right-0  " xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
                                            <path d="M0 190.9V185.1C0 115.2 50.52 55.58 119.4 44.1C164.1 36.51 211.4 51.37 244 84.02L256 96L267.1 84.02C300.6 51.37 347 36.51 392.6 44.1C461.5 55.58 512 115.2 512 185.1V190.9C512 232.4 494.8 272.1 464.4 300.4L283.7 469.1C276.2 476.1 266.3 480 256 480C245.7 480 235.8 476.1 228.3 469.1L47.59 300.4C17.23 272.1 .0003 232.4 .0003 190.9L0 190.9z"/>
                                        </svg>
                                     </button>
                                     <button @click="handleProductClick(item,'like','naver')">
                                        <svg class="text-red-400 w-6 h-auto fill-current absolute top-0 right-0" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
                                            <path d="M244 84L255.1 96L267.1 84.02C300.6 51.37 347 36.51 392.6 44.1C461.5 55.58 512 115.2 512 185.1V190.9C512 232.4 494.8 272.1 464.4 300.4L283.7 469.1C276.2 476.1 266.3 480 256 480C245.7 480 235.8 476.1 228.3 469.1L47.59 300.4C17.23 272.1 0 232.4 0 190.9V185.1C0 115.2 50.52 55.58 119.4 44.1C164.1 36.51 211.4 51.37 244 84C243.1 84 244 84.01 244 84L244 84zM255.1 163.9L210.1 117.1C188.4 96.28 157.6 86.4 127.3 91.44C81.55 99.07 48 138.7 48 185.1V190.9C48 219.1 59.71 246.1 80.34 265.3L256 429.3L431.7 265.3C452.3 246.1 464 219.1 464 190.9V185.1C464 138.7 430.4 99.07 384.7 91.44C354.4 86.4 323.6 96.28 301.9 117.1L255.1 163.9z"/>
                                        </svg>
                                    </button>
                                </div>
                            </Slide>
                            
                            <template #addons>
                                <Navigation />
                                <Pagination />
                            </template>
                        </Carousel><br>
                        <h1 class="mt-4 text-3xl font-extrabold sm:text-5xl" style="font-size: 20px;">LGBM 기반 추천</h1>
                        <Carousel class="mt-4" :itemsToShow="3.95" :wrapAround="true" :transition="500">
                            <Slide v-for="(item, index) in predictionList" :key="index">
                                <div class="carousel__item">
                                    <img :src="item.image_url" 
                                    @click="handleProductClick(item,'like','naver')"/>
                                    <button v-if="isClicked.includes(item.product_id)">                
                                        <svg class="text-red-400 w-6 h-auto fill-current absolute top-0 right-0  " xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
                                            <path d="M0 190.9V185.1C0 115.2 50.52 55.58 119.4 44.1C164.1 36.51 211.4 51.37 244 84.02L256 96L267.1 84.02C300.6 51.37 347 36.51 392.6 44.1C461.5 55.58 512 115.2 512 185.1V190.9C512 232.4 494.8 272.1 464.4 300.4L283.7 469.1C276.2 476.1 266.3 480 256 480C245.7 480 235.8 476.1 228.3 469.1L47.59 300.4C17.23 272.1 .0003 232.4 .0003 190.9L0 190.9z"/>
                                        </svg>
                                     </button>
                                     <button @click="handleProductClick(item,'like','naver')">
                                        <svg class="text-red-400 w-6 h-auto fill-current absolute top-0 right-0" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
                                            <path d="M244 84L255.1 96L267.1 84.02C300.6 51.37 347 36.51 392.6 44.1C461.5 55.58 512 115.2 512 185.1V190.9C512 232.4 494.8 272.1 464.4 300.4L283.7 469.1C276.2 476.1 266.3 480 256 480C245.7 480 235.8 476.1 228.3 469.1L47.59 300.4C17.23 272.1 0 232.4 0 190.9V185.1C0 115.2 50.52 55.58 119.4 44.1C164.1 36.51 211.4 51.37 244 84C243.1 84 244 84.01 244 84L244 84zM255.1 163.9L210.1 117.1C188.4 96.28 157.6 86.4 127.3 91.44C81.55 99.07 48 138.7 48 185.1V190.9C48 219.1 59.71 246.1 80.34 265.3L256 429.3L431.7 265.3C452.3 246.1 464 219.1 464 190.9V185.1C464 138.7 430.4 99.07 384.7 91.44C354.4 86.4 323.6 96.28 301.9 117.1L255.1 163.9z"/>
                                        </svg>
                                    </button>
                                </div>
                            </Slide>
                    
                            <template #addons>
                                <Navigation />
                                <Pagination />
                            </template>
                        </Carousel>
                    </div>
                    <div v-if="amazonButton != 0" class="w-1/2 h-1/2">
                        <h1 class="text-3xl font-extrabold sm:text-5xl" style="font-size: 20px;">아마존 모델 기반 추천</h1>
                        <Carousel class="mt-4" :itemsToShow="3.95" :wrapAround="true" :transition="500">
                            <Slide v-for="(item, index) in amazonPredictionList" :key="index">
                                <div class="carousel__item">
                                    <img :src="item.image_url" 
                                    @click="handleProductClick(item,'like','amazon')"/>
                                    <button v-if="amazonIsClicked.includes(item.product_id)">                
                                        <svg class="text-red-400 w-6 h-auto fill-current absolute top-0 right-0  " xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
                                            <path d="M0 190.9V185.1C0 115.2 50.52 55.58 119.4 44.1C164.1 36.51 211.4 51.37 244 84.02L256 96L267.1 84.02C300.6 51.37 347 36.51 392.6 44.1C461.5 55.58 512 115.2 512 185.1V190.9C512 232.4 494.8 272.1 464.4 300.4L283.7 469.1C276.2 476.1 266.3 480 256 480C245.7 480 235.8 476.1 228.3 469.1L47.59 300.4C17.23 272.1 .0003 232.4 .0003 190.9L0 190.9z"/>
                                        </svg>
                                     </button>
                                     <button @click="handleProductClick(item,'like','amazon')">
                                        <svg class="text-red-400 w-6 h-auto fill-current absolute top-0 right-0" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
                                            <path d="M244 84L255.1 96L267.1 84.02C300.6 51.37 347 36.51 392.6 44.1C461.5 55.58 512 115.2 512 185.1V190.9C512 232.4 494.8 272.1 464.4 300.4L283.7 469.1C276.2 476.1 266.3 480 256 480C245.7 480 235.8 476.1 228.3 469.1L47.59 300.4C17.23 272.1 0 232.4 0 190.9V185.1C0 115.2 50.52 55.58 119.4 44.1C164.1 36.51 211.4 51.37 244 84C243.1 84 244 84.01 244 84L244 84zM255.1 163.9L210.1 117.1C188.4 96.28 157.6 86.4 127.3 91.44C81.55 99.07 48 138.7 48 185.1V190.9C48 219.1 59.71 246.1 80.34 265.3L256 429.3L431.7 265.3C452.3 246.1 464 219.1 464 190.9V185.1C464 138.7 430.4 99.07 384.7 91.44C354.4 86.4 323.6 96.28 301.9 117.1L255.1 163.9z"/>
                                        </svg>
                                    </button>
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
import { computed, ref, defineComponent, onMounted } from 'vue';
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
        const amazonProductList = ref([]);
        const amazonProductListToShow = ref([]);
        //flag 변수
        const phaseIndex = ref(0);
        const isClicked = ref([]);
        const amazonIsClicked = ref([]);
        const amazonButton = ref(0);
        const isLoading = ref(true); 
        //predict 결과
        const predictionList = ref([]);
        const amazonPredictionList = ref([]);
        //store
        const store = userInfoStore();

        //get으로 상품 27개 가져오기(네이버 데이터)
        const getNaverList = async () => {
            try {
                const user_id = store.getDataAll.user_id
                const response = await axios.get(`/api/user/matched-items/${user_id}`);
                productList.value = response.data;
                store.matchedItems.value = productList.value;
                console.log('productList', productList.value);
                console.log('유저 id', user_id);
                setPredictionList();
            } catch (error) {
                console.error('데이터를 가져오는 중에 오류가 발생했습니다:', error);
            }
        };

        //get으로 아마존 상품 27개 가져오기
        const getAmazonList = async () => {
            try {
                const response = await axios.get(`/api/amazon/items-select`);
                amazonProductList.value = response.data;
                store.amazonMatchedItems.value = amazonProductList.value;
                console.log('amazon productList', amazonProductList.value);
                showNextProducts();
                setPredictionList();
            } catch (error) {
                console.error('데이터를 가져오는 중에 오류가 발생했습니다:', error);
            }
        };

        //get으로 prediction 결과 10개 가져오기
        const getPrediction = async(data,model) => {
            if (data == 'naver'){
                try {
                    const user_id = store.getDataAll.user_id
                    const response = await axios.get(`/api/user/items-prediction/${user_id}`);
                    predictionList.value = response.data;
                    console.log('predictionList', predictionList.value);
                } catch (error) {
                    console.error('데이터를 가져오는 중에 오류가 발생했습니다:', error);
                }
            } else {
                try {
                    const user_id = store.getDataAll.user_id
                    const response = await axios.get(`/api/amazon/items-prediction/${model}/${user_id}`);
                    amazonPredictionList.value = response.data;
                    console.log('amazonPredictionList', amazonPredictionList.value);
                } catch (error) {
                    console.error('데이터를 가져오는 중에 오류가 발생했습니다:', error);
                }
            }
                
        };

        // LGBM inference button
        const buttonClass = computed(() => {
            return isClicked.value.length === 0 ? 'blur-lg' : '';
        });


        //상품 9개를 보여주는 테이블에서 다음 9개의 값을 보여주는 함수(amazon)
        const showNextProducts = async() => {
            const startIndex = (phaseIndex.value % 3) * 9; // 각 페이즈의 시작 인덱스 계산
            amazonProductListToShow.value = amazonProductList.value.slice(startIndex, startIndex + 9);
            phaseIndex.value++;
        };


        //predictionList,amazonPredictionList 초기화 함수
        const setPredictionList = async() => {
            amazonPredictionList.value = [{ image_url: 'https://shop-phinf.pstatic.net/20230308_89/1678249882534ecdzU_JPEG/79385717218304917_1496244196.jpg?type=f480_480' }];
            console.log('predictionList', predictionList.value);
        };

        //상품 중 고른 상품에 대해 POST와 DELETE 요청을 처리하는 함수
        const handleProductClick = async (product, type, data) => { //type : 'like' or 'interation', data: 'amazon' or 'naver'
            const user_id = store.getDataAll.user_id
            if (data == 'amazon'){
                if (!amazonIsClicked.value.includes(product.product_id)) {
                    try {
                    // POST 요청을 보내는 부분
                    const response = await axios.post(`/api/amazon/user/${type}/`, {
                        "user_id": user_id,
                        "product_id": product.product_id,
                    });
                    amazonIsClicked.value.push(product.product_id);
                    console.log('amazon POST request successful:',amazonIsClicked.value);
                    } catch (error) {
                    console.error('Error sending POST request:', error);
                    }
                } else {
                    try {
                    // DELETE 요청을 보내는 부분
                    await axios.delete(`/api/amazon/user/${type}/`, {
                        data: {
                            "user_id": user_id,
                            "product_id": product.product_id,
                        },
                    });
                    amazonIsClicked.value.splice(amazonIsClicked.value.indexOf(product.product_id), 1);
                    console.log('amazon DELETE request successful:',amazonIsClicked.value);
                    } catch (error) {
                    console.error('Error sending DELETE request:', error);
                    }
                }
            }
            else{
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
            }
            
        };

        
        //아마존 데이터로 전환
        const amazonTrigger = async () => {
            amazonButton.value++;
        };

        //9개 이미지 랜더링 hook
        onMounted(async () => {
            // 비동기 작업을 수행하여 시간을 늦출 수 있음, userInfoStore에 post로 user_id를 가져오는데 생기는 지연 시간 때문에 생기는 버그 때문에 추가
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // 시간이 지난 후에 실행될 코드
            console.log('2초 후에 실행됩니다.');
            getNaverList();
            getPrediction('naver');
            getAmazonList();

            isClicked.value = [];
            amazonIsClicked.value =[];
            await new Promise(resolve => setTimeout(resolve, 3000));
            isLoading.value = false; 
        });

        return {
        isClicked,
        isLoading,
        amazonIsClicked,
        productList,
        predictionList,
        amazonPredictionList,
        amazonProductList,
        amazonProductListToShow,
        productListToShow,
        showNextProducts,
        buttonClass,
        setPredictionList,
        handleProductClick,
        getPrediction,
        amazonTrigger,
        amazonButton
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


