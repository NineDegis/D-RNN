import Router from 'koa-router';
import * as review from '../db/review';

const reviewAPI = new Router();

reviewAPI.get('/read', async (ctx) => {
  const movieID = ctx.request.query['id'];
  ctx.body = await review.getReviewList(movieID);
});

reviewAPI.post('/write', async (ctx) => {
  const movieID = ctx.request.query['id'];
  const reviewComment = ctx.request.body['comment'];
  await review.insertReview(movieID, reviewComment);
  ctx.body = await review.getReviewList(movieID);
  ctx.res.dataType = 'application/javascript'
});

export default reviewAPI;
