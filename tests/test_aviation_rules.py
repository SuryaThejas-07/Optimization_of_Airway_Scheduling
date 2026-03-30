from agno_runway.utils.aviation_rules import AviationRulesEngine


def test_required_gap_respects_min_constraints() -> None:
    rules = AviationRulesEngine(
        min_time_gap_seconds=30,
        runway_occupancy_seconds=50,
        safety_buffer_seconds=10,
    )

    gap_h_to_l = rules.required_gap("H", "L")
    gap_l_to_l = rules.required_gap("L", "L")

    assert gap_h_to_l >= 30
    assert gap_l_to_l >= 30
    assert gap_h_to_l > gap_l_to_l


def test_is_safe_interval() -> None:
    rules = AviationRulesEngine()

    assert rules.is_safe_interval(
        leader_time=100, follower_time=260, leader_wake="M", follower_wake="L"
    )
    assert not rules.is_safe_interval(
        leader_time=100, follower_time=120, leader_wake="M", follower_wake="L"
    )
