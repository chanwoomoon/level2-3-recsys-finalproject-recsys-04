<template>
  <section class="bg-gray-50 text-center">
	<div class="flex justify-center items-center h-screen">
		<div v-if="isLoading" class="rounded-full h-20 w-20 bg-violet-800 animate-ping"></div>
	</div>
  </section>
</template>

<script>
export default {
	data () {
        return {
            isLoading: true
        }
    }
}
</script>


