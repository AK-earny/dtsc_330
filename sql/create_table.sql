CREATE TABLE IF NOT EXISTS grants (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  application_id VARCHAR(20) NOT NULL,
  start_at DATE,
  grant_type VARCHAR(10),
  total_cost INTEGER
);

CREATE TABLE articles (
    article_id INTEGER PRIMARY KEY,
    title TEXT,
    author TEXT,
    year INTEGER
);

CREATE TABLE articlesgrantsbridge (
    article_id INTEGER,
    grant_id INTEGER,
    PRIMARY KEY (article_id, grant_id),
    FOREIGN KEY (article_id) REFERENCES Articles(article_id),
    FOREIGN KEY (grant_id) REFERENCES Grants(grant_id)
);