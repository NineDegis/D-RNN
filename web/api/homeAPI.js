import Router from 'koa-router';
import * as movie from '../db/movie';
import reviewAPI from './review';

const homeAPI = new Router();

homeAPI.get('/', async (ctx) => {
  const movieList = await movie.getMovieList();
  await ctx.render('home', {
    movieList
  });
});

homeAPI.use('reviews', reviewAPI.routes());

export default homeAPI;
