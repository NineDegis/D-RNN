import Router from 'koa-router';
import pool from '../database.js';

const home = new Router();

const convertQuery = (queryResult, attrList) => {
  const numAttrs = attrList.length;
  const convertedQuery = {};
  for(let j=0; j<numAttrs; j++) {
    convertedQuery[attrList[j]] = queryResult[attrList[j]];
  }
  return convertedQuery;
};

home.get('/', async (ctx) => {
  // pool.query('select * from movie', (error, results, fields) => {
  //   if (error) console.log(error);
  //   else {
  //     movies.concat(results);
  //   }
  // });
  const movies = [];
  await pool.query('select * from movie', (error, results, fields) => {
    const attrList = ['title', 'genre', 'play_time', 'post_url'];
    if (error) console.log(error);
    else {
      const numResults = results.length;
      for(let i=0; i<numResults; i++) {
        movies.push(convertQuery(results[i], attrList));
      }
    }
  });
  await ctx.render('home', {
    movies
  });
});

export default home;
