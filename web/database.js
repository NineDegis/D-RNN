import mysql from 'mysql';

class DatabasePool {
  constructor() {
    if (!this.instance) this.instance = this.createConnectionPool();
    return this.instance;
  }

  createConnectionPool() {
    // TODO(sejin): Make a separated config file
    // const dbConfig = JSON.parse(dbConfigJson);
    const dbConfig = {
      "host": "172.21.0.3",
      "port": 3306,
      "user": "root",
      "password": "graduation",
      "database": "movie_reviews",
      "connectionLimit": 100
    };

    return mysql.createPool({
      connectionLimit: dbConfig.connectionLimit,
      host: dbConfig.host,
      port: dbConfig.port,
      user: dbConfig.user,
      password: dbConfig.password,
      database: dbConfig.database,
    });
  }

  makeQuery(querySentence, callback) {
    pool.query(querySentence, (error, results, fields) => {
      if (error) console.log(error);
      else {
        callback();
      }
    });
  }
}

const pool = (new DatabasePool());

export default pool;
