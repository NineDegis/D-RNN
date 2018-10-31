import Router from 'koa-router';

const home = new Router();

home.get('/', (ctx) => {
  ctx.body = 'home';
});

export default home;
