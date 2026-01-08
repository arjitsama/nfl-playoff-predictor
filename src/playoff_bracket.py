# src/playoff_bracket.py

AFC = {
    "seed_to_team": {
        1: "DEN",   # bye
        2: "NE",
        3: "JAX",
        4: "HOU",
        5: "BUF",
        6: "PIT",
        7: "LAC",
    },
    "wildcard": [(2, 7), (3, 6), (4, 5)],
}

NFC = {
    "seed_to_team": {
        1: "SEA",   # bye
        2: "CHI",
        3: "PHI",
        4: "CAR",
        5: "LA",
        6: "SF",
        7: "GB",
    },
    "wildcard": [(2, 7), (3, 6), (4, 5)],
}
