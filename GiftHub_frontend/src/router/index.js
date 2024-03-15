import { createWebHistory, createRouter } from "vue-router";
import Main from "@/views/MainView.vue";
import Login from "@/views/LoginView.vue";
import Gender from "@/views/GenderView.vue";
import Age from "@/views/AgeView.vue";
import Color from "@/views/ColorView.vue";
import Price from "@/views/PriceView.vue";
import Inference from "@/views/InferenceView.vue";
import Situation from "@/views/SituationView.vue";
import Test from "@/views/test.vue";
import Design from "@/views/DesignTest.vue"
const routes = [
  {
    path: "/",
    name: "Main",
    component: Main,
  },
  {
    path: "/login",
    name: "Login",
    component: Login,
  },
  {
    path: "/gender",
    name: "Gender",
    component: Gender,
  },
  {
    path: "/age",
    name: "Age",
    component: Age,
  },
  {
    path: "/color",
    name: "Color",
    component: Color,
  },
  {
    path: "/color",
    name: "Color",
    component: Color,
  },
  {
    path: "/price",
    name: "Price",
    component: Price,
  },
  {
    path: "/situation",
    name: "Situation",
    component: Situation,
  },
  {
    path: "/inference",
    name: "Inference",
    component: Inference,
  },
  {
    path: "/test",
    name: "Test",
    component: Test,
  },
  {
    path: "/design",
    name: "DesignTest",
    component: Design,
  },
];

const router = createRouter({
    history: createWebHistory(),
    routes,
});

export default router;