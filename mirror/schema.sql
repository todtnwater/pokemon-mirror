-- DB 서버(N150)의 MySQL에서 root로 한 번 실행하세요.
-- '여기에_비밀번호' 부분은 직접 정한 비밀번호로 바꾸세요.
-- 100.79.16.67 은 ssk-server의 Tailscale IP입니다. 이 주소에서만 접속을 허용합니다.

CREATE DATABASE IF NOT EXISTS pokemon_mirror DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'pokemon'@'100.79.16.67' IDENTIFIED BY '여기에_비밀번호';
GRANT SELECT, INSERT, UPDATE ON pokemon_mirror.* TO 'pokemon'@'100.79.16.67';
FLUSH PRIVILEGES;

USE pokemon_mirror;

-- 날짜별·포켓몬별 1등 횟수. 사진이나 IP는 저장하지 않습니다.
CREATE TABLE IF NOT EXISTS pokemon_daily_count (
    match_date  DATE        NOT NULL,
    dex         INT         NOT NULL,
    name        VARCHAR(64) NOT NULL,
    match_count INT         NOT NULL DEFAULT 0,
    PRIMARY KEY (match_date, dex)
);
