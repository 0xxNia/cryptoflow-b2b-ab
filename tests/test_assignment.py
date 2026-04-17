import pytest

from cryptoflow.assignment import VariantSpec, assign_variant, mutually_exclusive_pick


def test_assign_variant_stable():
    v = (
        VariantSpec("control", 0.5),
        VariantSpec("treatment", 0.5),
    )
    a = assign_variant(salt="prod", experiment_id="exp_fee_bps", unit_id="user_42", variants=v)
    b = assign_variant(salt="prod", experiment_id="exp_fee_bps", unit_id="user_42", variants=v)
    assert a == b
    assert a in {"control", "treatment"}


def test_assign_variant_rejects_non_positive_weight():
    with pytest.raises(ValueError, match="weight"):
        assign_variant(
            salt="prod",
            experiment_id="exp_fee_bps",
            unit_id="user_42",
            variants=(VariantSpec("control", 1.0), VariantSpec("treatment", 0.0)),
        )


def test_mutually_exclusive_pick():
    exps = [
        ("exp_a", (VariantSpec("c", 1), VariantSpec("t", 1))),
        ("exp_b", (VariantSpec("c", 1), VariantSpec("t", 1))),
    ]
    picked = mutually_exclusive_pick(salt="x", unit_id="u1", experiments=exps)
    assert picked is not None
    exp_id, variant = picked
    assert exp_id in {"exp_a", "exp_b"}
    assert variant in {"c", "t"}
