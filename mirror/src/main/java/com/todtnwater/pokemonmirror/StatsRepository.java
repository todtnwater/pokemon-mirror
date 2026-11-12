package com.todtnwater.pokemonmirror;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;

import java.sql.Date;
import java.time.LocalDate;
import java.util.List;

/**
 * DB 서버(MySQL)의 pokemon_daily_count 테이블에 날짜별 1등 횟수를 쌓습니다.
 * 저장하는 값은 날짜, 도감 번호, 이름, 횟수뿐입니다. (사진·IP 저장 안 함)
 */
@Repository
public class StatsRepository {

    private static final RowMapper<PokemonCount> COUNT_MAPPER = (rs, rowNum) ->
            new PokemonCount(rs.getInt("dex"), rs.getString("name"), rs.getLong("total"));

    private final JdbcTemplate jdbcTemplate;

    public StatsRepository(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    /** 오늘 날짜의 해당 포켓몬 횟수를 1 올립니다. 오늘 첫 기록이면 새 줄을 만듭니다. */
    public void increment(LocalDate date, int dex, String name) {
        jdbcTemplate.update("""
                INSERT INTO pokemon_daily_count (match_date, dex, name, match_count)
                VALUES (?, ?, ?, 1)
                ON DUPLICATE KEY UPDATE match_count = match_count + 1
                """, Date.valueOf(date), dex, name);
    }

    /** 전체 기간 TOP N. */
    public List<PokemonCount> topAllTime(int limit) {
        return jdbcTemplate.query("""
                SELECT dex, MAX(name) AS name, SUM(match_count) AS total
                FROM pokemon_daily_count
                GROUP BY dex
                ORDER BY total DESC, dex ASC
                LIMIT ?
                """, COUNT_MAPPER, limit);
    }

    /** 특정 날짜 TOP N. */
    public List<PokemonCount> topOfDay(LocalDate date, int limit) {
        return jdbcTemplate.query("""
                SELECT dex, name, match_count AS total
                FROM pokemon_daily_count
                WHERE match_date = ?
                ORDER BY total DESC, dex ASC
                LIMIT ?
                """, COUNT_MAPPER, Date.valueOf(date), limit);
    }

    /** 전체 기간 누적 횟수. */
    public long totalAllTime() {
        Long total = jdbcTemplate.queryForObject(
                "SELECT COALESCE(SUM(match_count), 0) FROM pokemon_daily_count", Long.class);
        return total == null ? 0 : total;
    }
}
