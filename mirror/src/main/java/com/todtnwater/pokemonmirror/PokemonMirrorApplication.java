package com.todtnwater.pokemonmirror;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling
public class PokemonMirrorApplication {

    public static void main(String[] args) {
        SpringApplication.run(PokemonMirrorApplication.class, args);
    }
}
