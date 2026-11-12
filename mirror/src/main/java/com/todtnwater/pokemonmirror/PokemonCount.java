package com.todtnwater.pokemonmirror;

/** 통계 화면에 보여 줄 포켓몬 한 마리의 집계 결과. */
public record PokemonCount(int dex, String name, long count) {
}
