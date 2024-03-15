<template>
  <form>
    <label>제목<input v-model="postData.postTitle" type="text" /></label><br />
    <label>설명<input v-model="postData.postDescription" type="text" /></label>
    <button @click.prevent="addPostData">제출</button>
  </form>
</template>

<script setup>
import { usePostStore } from '../store/index';
import { ref } from 'vue';

// store에서 사용할 함수를 가져온다
const store = usePostStore();
const { createContents } = store;

// form 제출 시 서버에 전달할 데이터 ref를 만들어 주기
const postData = ref({
  postTitle: '',
  postDescription: '',
});

// 데이터를 추가하는 함수
const addPostData = () => {
  createContents(postData.value);
  
  // input에 입력된 값 초기화
  postData.value.postTitle = '';
  postData.value.postDescription = '';
}
</script>