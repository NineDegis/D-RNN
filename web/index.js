import Koa from 'koa';
import Router from 'koa-router';
import home from './api/home';

const params = process.argv;

const testMode = (params[2] === 'test');
const port = params.length > 3 ? Number(params[3]) : (testMode ? 5913 : 9897);
const app = new Koa();
const router = new Router();

router
  .use('/', home.routes());

app
  .use(async (ctx, next) => {
    await next();
    const ip = ctx.request.ip;
    const rt = ctx.response.get('X-Response-Time');
    console.log(`${ctx.method} ${ctx.url} - ${rt} from ${ip}`);
  })
  .use(async (ctx, next) => {
    const start = Date.now();
    await next();
    const ms = Date.now() - start;
    ctx.set('X-Response-Time', `${ms}ms`);
  })
  .use(router.routes())
  .use(router.allowedMethods())
  .listen(port, () => {
    console.log(`Server is listening to port ${port}`);
  });
