import Router from 'koa-router';

const home = new Router();

const movies = ['test',/* mongoose */];

home.get('/', async (ctx) => {
  await ctx.render('home', {
    movies
  });
});

export default home;
