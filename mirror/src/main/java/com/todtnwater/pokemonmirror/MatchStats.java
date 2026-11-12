package com.todtnwater.pokemonmirror;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DataAccessException;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.time.ZoneId;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 어떤 포켓몬이 1등으로 많이 나왔는지 DB에 셉니다. 사진은 저장하지 않고 도감 번호만 셉니다.
 * 매 프레임마다 세면 숫자가 부풀려지므로, 한 접속자에게 같은 포켓몬이 연속으로
 * stable-frames번 1등으로 나왔을 때 한 번만 셉니다.
 * DB에 문제가 생겨도 닮은꼴 찾기 자체는 계속 동작하도록 오류는 로그만 남깁니다.
 */
@Component
public class MatchStats {

    private static final Logger log = LoggerFactory.getLogger(MatchStats.class);
    private static final ZoneId KOREA = ZoneId.of("Asia/Seoul");

    private record ClientState(int candidateDex, int streak, int countedDex) {
    }

    private final StatsRepository statsRepository;
    private final int stableFrames;
    private final ConcurrentHashMap<String, ClientState> clients = new ConcurrentHashMap<>();

    public MatchStats(StatsRepository statsRepository,
                      @Value("${mirror.stats.stable-frames}") int stableFrames) {
        this.statsRepository = statsRepository;
        this.stableFrames = stableFrames;
    }

    public void record(String clientKey, MatchResponse.PokemonMatch top) {
        ClientState state = clients.compute(clientKey, (key, old) -> {
            if (old == null || old.candidateDex() != top.dex()) {
                return new ClientState(top.dex(), 1, old == null ? -1 : old.countedDex());
            }
            return new ClientState(top.dex(), old.streak() + 1, old.countedDex());
        });
        if (state.streak() == stableFrames && state.countedDex() != top.dex()) {
            clients.computeIfPresent(clientKey, (key, old) -> new ClientState(old.candidateDex(), old.streak(), top.dex()));
            try {
                statsRepository.increment(LocalDate.now(KOREA), top.dex(), top.name());
            } catch (DataAccessException exception) {
                log.warn("통계 저장 실패 (닮은꼴 찾기는 계속 동작): {}", exception.getMessage());
            }
        }
    }

    /** 통계 화면용: 전체 누적, 전체 TOP, 오늘 TOP. DB 오류면 빈 결과를 돌려줍니다. */
    public Map<String, Object> summary(int limit) {
        try {
            return Map.of(
                    "total", statsRepository.totalAllTime(),
                    "top", statsRepository.topAllTime(limit),
                    "today", statsRepository.topOfDay(LocalDate.now(KOREA), limit));
        } catch (DataAccessException exception) {
            log.warn("통계 조회 실패: {}", exception.getMessage());
            return Map.of("total", 0L, "top", List.of(), "today", List.of());
        }
    }

    /** 1시간마다 접속자별 상태를 비웁니다. (DB에 쌓인 숫자는 그대로 둡니다) */
    @Scheduled(fixedRate = 3_600_000)
    public void cleanUpClients() {
        clients.clear();
    }
}
