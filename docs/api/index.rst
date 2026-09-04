API Reference
=============

This page contains the auto-generated API reference for pyseekdb.

Main Package
------------

.. automodule:: pyseekdb
   :members:
   :undoc-members:
   :show-inheritance:

Embedding Functions
-------------------

Since pyseekdb 2.0 no embedding function implementations are bundled. Users
implement the ``EmbeddingFunction`` / ``SparseEmbeddingFunction`` protocols or
register their own classes via ``@register_embedding_function`` and
``@register_sparse_embedding_function``.

Embedding Protocol
~~~~~~~~~~~~~~~~~~

.. automodule:: pyseekdb.client.embedding_function
   :members:
   :undoc-members:
   :show-inheritance:

Sparse Embedding Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pyseekdb.client.sparse_embedding_function
   :members:
   :undoc-members:
   :show-inheritance:
