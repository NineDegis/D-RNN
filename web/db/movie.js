import pool from './database_pool';
import { convertQuery } from './db_helpers';

const getMovieList = async () => {
  const movieList = [];
  const rows = await pool.query('select * from movie');
  try {
    const attrList = ['movie_id', 'title', 'post_url', 'description', 'play_time', 'genre', 'release_year'];
    const numrows = rows.length;
    for (let i = 0; i < numrows; i++) {
      movieList.push(convertQuery(rows[i], attrList));
      const genre = rows[i].genre;
      movieList[movieList.length - 1]['main_genre'] = genre.substr(0, genre.indexOf(','));
      const playTime = rows[i].play_time;
      const hour = Math.floor(playTime / 60);
      const min = playTime % 60;
      movieList[movieList.length - 1]['play_time'] = (hour < 10 ? '0' : '') + hour + ':' + (min < 10 ? '0' : '') + (playTime % 60);
    }
    return movieList;
  } catch (err) {
    console.log(err);
    return false;
  }
};

export { getMovieList };
