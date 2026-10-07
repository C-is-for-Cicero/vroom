from f1pred import config


def test_race_points_values():
    assert config.race_points(1) == 25
    assert config.race_points(10) == 1
    assert config.race_points(11) == 0


def test_sprint_points_separate_from_race_points():
    assert config.sprint_points(1) == 8
    assert config.sprint_points(8) == 1
    assert config.sprint_points(9) == 0


def test_fastest_lap_bonus_eras():
    # 2019–2024: 1 point if top 10. 2025+: nothing.
    assert config.fastest_lap_bonus(2023, 5) == 1
    assert config.fastest_lap_bonus(2023, 11) == 0
    assert config.fastest_lap_bonus(2025, 1) == 0
    assert config.fastest_lap_bonus(2026, 1) == 0
    assert config.fastest_lap_bonus(2018, 1) == 0


def test_constructor_lineage_2026_reset():
    assert config.canonical_constructor("sauber") == "audi"
    assert config.canonical_constructor("alfa") == "audi"
    assert config.canonical_constructor("toro_rosso") == "rb"
    assert config.canonical_constructor("force_india") == "aston_martin"
    assert config.canonical_constructor("renault") == "alpine"
    assert config.canonical_constructor("cadillac") == "cadillac"
    # unknown ids pass through instead of crashing
    assert config.canonical_constructor("brand_new_team") == "brand_new_team"
