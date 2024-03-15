const target = "http://101.79.11.75:8000";

const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  lintOnSave: false,
  devServer:{
    proxy: {
      //port:8080, port 지정 따로 안해줘도 됨
      '^/api' : {
      target,
      changeOrigin: true
      }
    }
  }
})
