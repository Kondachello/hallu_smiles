"""Adapters that expose each detector through graph_eval's shared contract.

This integration layer imports both the shared contract (``graph_eval.types``) and
the existing HalluGraph code (``run`` / ``src``).  Neither GraphEval nor HalluGraph
imports this layer, and this layer does not import the experiment framework.
"""
