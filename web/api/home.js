import Router from 'koa-router';
import * as movie from '../db/movie';
import * as review from '../db/review';

const home = new Router();

home.get('/', async (ctx) => {
  const movieList = await movie.getMovieList();
  await ctx.render('home', {
    movieList
  });
});

home.get('reviews', async (ctx) => {
  const movieID = ctx.request.query['id'];
  const reviewList = await review.getReviewList(movieID);
  ctx.body = reviewList;
});

export default home;
