"""Compatibility namespace for Outlines' unused airport-code helper.

The PyPI distribution named ``pyairports==0.0.1`` is an unrelated placeholder:
it declares the distribution but does not provide the ``pyairports`` module.
Outlines 0.0.46 imports that module eagerly even when only JSON constrained
decoding is used.  The Job inserts this checked-in directory before
site-packages; only the unused airport list is supplied here.
"""
