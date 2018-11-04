import pool from './database_pool';
import { convertQuery } from './db_helpers';

const getReviewList = async (movieID) => {
  const reviewList = [];
  const rows = await pool.query('select * from review where movie_id= ? ', movieID);
  try {
    const attrList = ['comment', 'review_datetime'];
    const numResults = rows.length;
    for (let i = 0; i < numResults; i++) {
      reviewList.push(convertQuery(rows[i], attrList));
    }
    return reviewList;
  } catch (err) {
    console.log(err);
    return false;
  }
};

export { getReviewList };
