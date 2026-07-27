from __future__ import annotations

from hallugraph_dynamic_typing.graph_spec import MODEL_NODE_IDS, NODE_SPECS, NodeKind
from hallugraph_dynamic_typing.state import AgentState, FORBIDDEN_STATE_KEYS, SourceInputState


def test_node_ids_are_unique_and_model_inventory_is_derived() -> None:
    node_ids = [item.node_id for item in NODE_SPECS]
    assert len(node_ids) == len(set(node_ids))
    assert MODEL_NODE_IDS == {
        item.node_id for item in NODE_SPECS if item.kind in {NodeKind.MODEL, NodeKind.NLI}
    }


def test_source_nodes_cannot_read_answer() -> None:
    offenders = [item.node_id for item in NODE_SPECS if item.phase == "source" and item.may_read_answer]
    assert offenders == []


def test_source_input_and_agent_state_have_no_gold_fields() -> None:
    source_keys = set(SourceInputState.__annotations__)
    state_keys = set(AgentState.__annotations__)
    assert source_keys.isdisjoint(FORBIDDEN_STATE_KEYS)
    assert state_keys.isdisjoint(FORBIDDEN_STATE_KEYS)


def test_all_model_nodes_have_prompt_ids() -> None:
    assert all(
        item.prompt_id
        for item in NODE_SPECS
        if item.kind in {NodeKind.MODEL, NodeKind.NLI}
    )

