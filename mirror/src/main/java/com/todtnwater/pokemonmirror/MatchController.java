package com.todtnwater.pokemonmirror;

import jakarta.servlet.http.HttpServletRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestClientException;

import java.util.Map;

@RestController
public class MatchController {

    private static final Logger log = LoggerFactory.getLogger(MatchController.class);

    private final AiClient aiClient;
    private final RateLimiter rateLimiter;
    private final MatchStats matchStats;
    private final int maxUploadBytes;

    public MatchController(AiClient aiClient,
                           RateLimiter rateLimiter,
                           MatchStats matchStats,
                           @Value("${mirror.max-upload-bytes}") int maxUploadBytes) {
        this.aiClient = aiClient;
        this.rateLimiter = rateLimiter;
        this.matchStats = matchStats;
        this.maxUploadBytes = maxUploadBytes;
    }

    /** 브라우저가 보낸 웹캠 프레임(JPEG)을 검사한 뒤 AI 서버로 넘깁니다. 사진은 저장하지 않습니다. */
    @PostMapping(value = "/api/match", consumes = MediaType.IMAGE_JPEG_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<?> match(@RequestBody byte[] jpeg, HttpServletRequest request) {
        String clientKey = clientIp(request);
        if (!rateLimiter.tryAcquire(clientKey)) {
            return error(HttpStatus.TOO_MANY_REQUESTS, "요청이 너무 많아요. 잠시 후 다시 시도해 주세요.");
        }
        if (jpeg.length == 0) {
            return error(HttpStatus.BAD_REQUEST, "이미지가 비어 있습니다.");
        }
        if (jpeg.length > maxUploadBytes) {
            return error(HttpStatus.PAYLOAD_TOO_LARGE, "이미지가 너무 큽니다.");
        }
        if (!isJpeg(jpeg)) {
            return error(HttpStatus.UNSUPPORTED_MEDIA_TYPE, "JPEG 이미지만 받습니다.");
        }

        MatchResponse result = aiClient.match(jpeg);
        if (result == null) {
            return error(HttpStatus.SERVICE_UNAVAILABLE, "AI 서버 응답이 비어 있습니다.");
        }
        MatchResponse.PokemonMatch top = result.top();
        if (top != null) {
            matchStats.record(clientKey, top);
        }
        return ResponseEntity.ok(result);
    }

    /** 1등으로 많이 나온 포켓몬: 전체 누적 TOP 5와 오늘 TOP 5. */
    @GetMapping(value = "/api/stats", produces = MediaType.APPLICATION_JSON_VALUE)
    public Map<String, Object> stats() {
        return matchStats.summary(5);
    }

    @ExceptionHandler(RestClientException.class)
    public ResponseEntity<Map<String, String>> aiUnavailable(RestClientException exception) {
        log.warn("AI 서버 호출 실패: {}", exception.getMessage());
        return error(HttpStatus.SERVICE_UNAVAILABLE, "AI 서버에 연결할 수 없습니다.");
    }

    private static ResponseEntity<Map<String, String>> error(HttpStatus status, String message) {
        return ResponseEntity.status(status).body(Map.of("error", message));
    }

    private static boolean isJpeg(byte[] data) {
        return data.length > 3 && (data[0] & 0xFF) == 0xFF && (data[1] & 0xFF) == 0xD8;
    }

    /** Cloudflare Tunnel을 거쳐 오므로 실제 접속자 IP는 CF-Connecting-IP 헤더에 들어 있습니다. */
    private static String clientIp(HttpServletRequest request) {
        String cloudflareIp = request.getHeader("CF-Connecting-IP");
        if (cloudflareIp != null && !cloudflareIp.isBlank()) {
            return cloudflareIp.trim();
        }
        String forwarded = request.getHeader("X-Forwarded-For");
        if (forwarded != null && !forwarded.isBlank()) {
            return forwarded.split(",")[0].trim();
        }
        return request.getRemoteAddr();
    }
}
