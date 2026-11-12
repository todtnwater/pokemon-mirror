package com.todtnwater.pokemonmirror;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

/**
 * 내부 Python AI 서버(127.0.0.1:8091)에 얼굴 사진을 보내고 결과를 받습니다.
 */
@Component
public class AiClient {

    private final RestClient restClient;

    public AiClient(@Value("${mirror.ai.base-url}") String baseUrl,
                    @Value("${mirror.ai.timeout-ms}") int timeoutMs) {
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(timeoutMs);
        requestFactory.setReadTimeout(timeoutMs);
        this.restClient = RestClient.builder()
                .baseUrl(baseUrl)
                .requestFactory(requestFactory)
                .build();
    }

    public MatchResponse match(byte[] jpeg) {
        return restClient.post()
                .uri("/match")
                .contentType(MediaType.IMAGE_JPEG)
                .body(jpeg)
                .retrieve()
                .body(MatchResponse.class);
    }
}
