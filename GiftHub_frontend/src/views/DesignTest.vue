// https://www.hyperui.dev/components/marketing/banners
// https://tailwindcss.com/docs/top-right-bottom-left -> tailwind layout 참고 사이트
<template>
    <section class="bg-gray-50 text-center">
      <div class="relative isolate overflow-hidden bg-gray-900 px-6 pt-16 shadow-2xl  sm:px-16 md:pt-24 lg:flex lg:gap-x-20 lg:px-24 lg:pt-0"> <!--보라색배경-->
        <div class="mx-auto max-w-screen-xl px-4 py-32 lg:flex lg:h-screen lg:items-center">
          <div class="mx-auto max-w-xl">
            <svg viewBox="0 0 1024 1024" class="absolute left-1/2 top-1/2 -z-10 h-[64rem] w-[64rem] -translate-y-1/2 [mask-image:radial-gradient(closest-side,white,transparent)] sm:left-full sm:-ml-80 lg:left-1/2 lg:ml-0 lg:-translate-x-1/2 lg:translate-y-0" aria-hidden="true">
              <circle cx="512" cy="512" r="512" fill="url(#759c1415-0410-454c-8f7c-9a820de03641)" fill-opacity="0.7" />
                <defs>
                  <radialGradient id="759c1415-0410-454c-8f7c-9a820de03641">
                    <stop stop-color="#7775D6" />
                    <stop offset="1" stop-color="#E935C1" />
                  </radialGradient>
                </defs>
            </svg>
            <div class="mx-auto max-w-xl ">
              <h1 class="text-3xl font-extrabold sm:text-5xl text-white">선물 받을 사람의 성별은?</h1>
              <h1 class="mt-4 font-extrabold sm:text-xl/relaxed text-white"><br>"선택해주세요."</h1>
              <div class="mt-8 flex flex-wrap justify-center gap-4">
                <button class="block w-full bg-white rounded bg-red-600 px-12 py-3 text-sm font-extrabold  shadow hover:bg-red-700 focus:outline-none focus:ring active:bg-red-500 sm:w-auto" >남</button>
                <button class="block w-full bg-white rounded px-12 py-3 text-sm font-extrabold shadow hover:bg-red-700 focus:outline-none focus:ring active:text-red-500 sm:w-auto" >여</button><br><br>
              </div>
            </div>
            <div class="absolute inset-x-0 bottom-0 h-40 ...">
                <nav class="flex flex-row flex-nowrap justify-between md:justify-center items-center" aria-label="Pagination">
                  <!-- Previous Page Button -->
                  <a class="flex w-10 h-10 mr-1 justify-center items-center rounded-full border border-gray-200 bg-white dark:bg-gray-800 text-black dark:text-white hover:border-gray-300 dark:hover:border-gray-600"
                  href="/" title="Previous Page">
                      <span class="sr-only">Previous Page</span>
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5"
                          stroke="currentColor" class="block w-5 h-5">
                          <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
                      </svg>
                    </a>
                  <!-- Page Buttons (1 to 5) -->
                  <button class="hidden md:flex w-10 h-10 mx-1 justify-center items-center rounded-full border border-gray-200 bg-white dark:bg-gray-700 text-black dark:text-white hover:border-gray-300 dark:hover:border-gray-600"
                      href="/gender" aria-current="page" title="Page 1">
                      1
                  </button>
                  <a class="hidden md:flex w-10 h-10 mx-1 justify-center items-center rounded-full border border-gray-200 bg-white dark:bg-gray-700 text-black dark:text-white hover:border-gray-300 dark:hover:border-gray-600"
                      href="/age" title="Page 2">
                      2
                  </a>
                  <a class="hidden md:flex w-10 h-10 mx-1 justify-center items-center rounded-full border border-gray-200 bg-white dark:bg-gray-700 text-black dark:text-white hover:border-gray-300 dark:hover:border-gray-600"
                      href="/color" title="Page 3">
                      3
                  </a>
                  <a class="hidden md:flex w-10 h-10 mx-1 justify-center items-center rounded-full border border-gray-200 bg-white dark:bg-gray-700 text-black dark:text-white hover:border-gray-300 dark:hover:border-gray-600"
                      href="/price" title="Page 4">
                      4
                  </a>
                  <a class="hidden md:flex w-10 h-10 mx-1 justify-center items-center rounded-full border border-gray-200 bg-white dark:bg-gray-700 text-black dark:text-white hover:border-gray-300 dark:hover:border-gray-600"
                      href="/situation" title="Page 5">
                      5
                  </a>
                  <!-- Next Page Button -->
                  <a class="flex w-10 h-10 ml-1 justify-center items-center rounded-full border border-gray-200 bg-white dark:bg-gray-800 text-black dark:text-white hover:border-gray-300 dark:hover:border-gray-600"
                  href="/age" title="Next Page">
                      <span class="sr-only">Next Page</span>
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5"
                          stroke="currentColor" class="block w-5 h-5">
                          <path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                      </svg>
                  </a>
                </nav>
            </div>         
          </div>
        </div>
      </div>
    </section>
</template>







<!--main-->
<!-- // https://www.hyperui.dev/components/marketing/banners
<template>
    <section class="bg-gray-50 text-center">
      <div class="relative isolate overflow-hidden bg-gray-900 px-6 pt-16 shadow-2xl  sm:px-16 md:pt-24 lg:flex lg:gap-x-20 lg:px-24 lg:pt-0">
        <div class="mx-auto max-w-screen-xl px-4 py-32 lg:flex lg:h-screen lg:items-center">
          <div class="mx-auto max-w-xl">
            <svg viewBox="0 0 1024 1024" class="absolute left-1/2 top-1/2 -z-10 h-[64rem] w-[64rem] -translate-y-1/2 [mask-image:radial-gradient(closest-side,white,transparent)] sm:left-full sm:-ml-80 lg:left-1/2 lg:ml-0 lg:-translate-x-1/2 lg:translate-y-0" aria-hidden="true">
              <circle cx="512" cy="512" r="512" fill="url(#759c1415-0410-454c-8f7c-9a820de03641)" fill-opacity="0.7" />
                <defs>
                  <radialGradient id="759c1415-0410-454c-8f7c-9a820de03641">
                    <stop stop-color="#7775D6" />
                    <stop offset="1" stop-color="#E935C1" />
                  </radialGradient>
                </defs>
            </svg>
            <div class="mx-auto max-w-md text-center lg:mx-0 lg:flex-auto lg:py-32 lg:text-left">
              <h2 class="text-3xl font-bold tracking-tight text-white sm:text-4xl">특별한 날<br />더 특별해지고 싶은 당신을 위한 추천</h2>
              <p class="mt-6 text-lg leading-8 text-gray-300">지금 바로 딱 맞는 선물을 추천 받아보세요. </p>
              <div class="mt-10 flex items-center justify-center gap-x-6 lg:justify-start">
                <a href="/gender" class="rounded-md bg-white px-3.5 py-2.5 text-sm font-semibold text-gray-900 shadow-sm hover:bg-gray-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white">선물 추천 바로가기</a>
                <a href="#" class="text-sm font-semibold leading-6 text-white">Learn more <span aria-hidden="true">→</span></a>
              </div>
            </div>         
          </div>
        </div>
      </div>
    </section>
</template> -->
