from oscarlic.allocation import Candidate, random_allocation, uniform_text_allocation, utility_allocation
from oscarlic.oracle import greedy_state_dependent_oracle


def candidates():
    return [
        Candidate("000", 3, 3.0, text_mask=True),
        Candidate("001", 2, 2.0, text_mask=False),
        Candidate("002", 4, 10.0, text_mask=True),
    ]


def test_utility_allocation():
    out = utility_allocation(candidates(), 6)
    assert [x.candidate_id for x in out.selected] == ["002", "001"]
    assert out.spent_bytes == 6


def test_uniform_text_is_text_first_and_stable():
    out = uniform_text_allocation(candidates(), 5)
    assert [x.candidate_id for x in out.selected] == ["000", "001"]


def test_random_is_reproducible():
    a = random_allocation(candidates(), 9, 7)
    b = random_allocation(candidates(), 9, 7)
    assert [x.candidate_id for x in a.selected] == [x.candidate_id for x in b.selected]


def test_state_dependent_oracle_additive_case():
    benefit = {"a": 4.0, "b": 2.0, "c": 1.0}
    costs = {"a": 4, "b": 1, "c": 1}
    def loss(selected):
        return 10.0 - sum(benefit[x] for x in selected)
    def cost(selected, candidate):
        return costs[candidate]
    result = greedy_state_dependent_oracle(["a", "b", "c"], 2, loss, cost)
    assert [s.candidate_id for s in result.steps] == ["b", "c"]
    assert result.spent_bytes == 2
    assert result.final_loss == 7.0
