package com.todtnwater.pokemonmirror;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.List;

/**
 * Python AI 서버가 돌려주는 결과.
 * found가 false면 box와 matches는 null입니다.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record MatchResponse(boolean found, List<Double> box, List<PokemonMatch> matches) {

    public record PokemonMatch(int dex, String name, String types, double score) {
    }

    @JsonIgnore
    public PokemonMatch top() {
        if (!found || matches == null || matches.isEmpty()) {
            return null;
        }
        return matches.get(0);
    }
}
