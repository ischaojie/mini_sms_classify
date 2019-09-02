<style>
  div{
    margin: 12px 0;
  }
</style>

<template>
  <div class="home">
    <h3>垃圾邮件分类 (by 潮戒)</h3>
      <img alt="GitHub stars"
           src="https://img.shields.io/github/stars/zhuzhezhe/mini_sms_classify"
      href="https://github.com/zhuzhezhe/mini_sms_classify">
     <a-row>
      <a-col :span="8" :offset="8">
        <a-textarea placeholder="Basic usage" :rows="4" v-model="email"/>
      </a-col>
    </a-row>
    <a-button type="primary" @click="predict">飞一会儿</a-button>
    <div>分类结果：{{result}}</div>
  </div>
</template>

<script>
    import axios from 'axios'

    export default {
  name: 'home',
  data() {
    return {
      result: '',
      email: ''
    };
  },
  methods: {
    predict() {
      const path = 'http://localhost:5000/predict?email='+this.email;
      axios.get(path)
        .then((res) => {
          this.result = res.data.label;

        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
  },
  created() {
    this.predict();
  },
}
</script>
