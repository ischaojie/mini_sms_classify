import Vue from 'vue'
import App from './App.vue'
import router from './router'

import Antd from "ant-design-vue"
import "ant-design-vue/dist/antd.css"


Vue.config.productionTip = false

Vue.use(Antd)


new Vue({
  router,
  render: function (h) { return h(App) }
}).$mount('#app')
