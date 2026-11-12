package com.todtnwater.pokemonmirror;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.concurrent.ConcurrentHashMap;

/**
 * 접속자(IP)마다 1초에 보낼 수 있는 요청 수를 제한합니다.
 * 한 사람이 서버 CPU를 독차지하지 못하게 막는 용도입니다.
 */
@Component
public class RateLimiter {

    private record Window(long second, int count) {
    }

    private final int limitPerSecond;
    private final ConcurrentHashMap<String, Window> windows = new ConcurrentHashMap<>();

    public RateLimiter(@Value("${mirror.rate-limit-per-second}") int limitPerSecond) {
        this.limitPerSecond = limitPerSecond;
    }

    public boolean tryAcquire(String clientKey) {
        long second = System.currentTimeMillis() / 1000;
        Window window = windows.compute(clientKey, (key, old) ->
                (old == null || old.second() != second) ? new Window(second, 1) : new Window(second, old.count() + 1));
        return window.count() <= limitPerSecond;
    }

    /** 1분마다 오래된 기록을 지워 메모리가 계속 늘지 않게 합니다. */
    @Scheduled(fixedRate = 60_000)
    public void cleanUp() {
        long now = System.currentTimeMillis() / 1000;
        windows.entrySet().removeIf(entry -> entry.getValue().second() < now - 5);
    }
}
