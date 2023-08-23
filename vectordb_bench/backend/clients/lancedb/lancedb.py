from ..api import VectorDB, DBConfig, DBCaseConfig, IndexType

import pyarrow as pa
import logging
from typing import Any, Type
from contextlib import contextmanager
from .config import LanceDBConfig, LanceDBIndexConfig

log = logging.getLogger(__name__)

class LanceDB(VectorDB):
    def __init__(
     self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig | None,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        **kwargs,
    ) -> None:
        """Initialize wrapper around the vector database client.

        Please drop the existing collection if drop_old is True. And create collection
        if collection not in the Vector Database

        Args:
            dim(int): the dimension of the dataset
            db_config(dict): configs to establish connections with the vector database
            db_case_config(DBCaseConfig | None): case specific configs for indexing and searching
            drop_old(bool): whether to drop the existing collection of the dataset.
        """
        self.collection_name = collection_name
        self.db_config = db_config
        self.case_config = db_case_config
        self._scalar_field = "id"
        self._vector_field = "vector"

        import lancedb
        db = lancedb.connect(self.db_config.get("uri"))
        if drop_old and self.collection_name in db.table_names():
            db.drop_table(self.collection_name)
        if collection_name not in db.table_names():
            schema = pa.schema([
                (self._scalar_field, pa.int32()),
                (self._vector_field, pa.list_(pa.float32(), dim))
            ])
            db.create_table(self.collection_name, schema=schema)
    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return LanceDBConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return LanceDBIndexConfig

    @contextmanager
    def init(self) -> None:
        """ create and destory connections to database.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        import lancedb
        self.tbl: lancedb.LanceTable | None = None

        db = lancedb.connect(self.db_config.get("uri"))
        self.tbl = db.open_table(self.collection_name)

        yield
        self.tbl = None
        del(self.tbl)

    def ready_to_load(self):
        pass

    def optimize(self):
        assert self.tbl is not None
        print(self.case_config.index_param().get('metric'))
        self.tbl.create_index(self.case_config.index_param().get('metric'), num_partitions=self.case_config.index_param().get('num_partitions'), num_sub_vectors=self.case_config.index_param().get('num_sub_vectors'))

    def ready_to_search(self):
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception):
        """Insert the embeddings to the vector database. The default number of embeddings for
        each insert_embeddings is 5000.

        Args:
            embeddings(list[list[float]]): list of embedding to add to the vector database.
            metadatas(list[int]): metadata associated with the embeddings, for filtering.
            **kwargs(Any): vector database specific parameters.

        Returns:
            int: inserted data count
        """
        assert self.tbl is not None
        assert len(embeddings) == len(metadata)
        try:
            self.tbl.add([{self._vector_field: e[0], self._scalar_field: e[1]} for e in zip(embeddings, metadata)])
        except Exception as e:
            log.info(f"Failed to insert data: {e}")
            return (0, e)
        return (len(embeddings), None)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        """Get k most similar embeddings to query vector.

        Args:
            query(list[float]): query embedding to look up documents similar to.
            k(int): Number of most similar embeddings to return. Defaults to 100.
            filters(dict, optional): filtering expression to filter the data while searching.

        Returns:
            list[int]: list of k most similar embeddings IDs to the query embedding.
        """
        if filters:
            expr = f"{self._scalar_field} {filters.get('metadata')}"
            res = self.tbl \
                .search(query) \
                .metric(self.case_config.index_param().get('metric')) \
                .limit(k) \
                .select([self._scalar_field]) \
                .where(expr) \
                .to_arrow()
        else:
            res = self.tbl \
                .search(query) \
                .metric(self.case_config.search_param().get('metric')) \
                .nprobes(self.case_config.search_param().get('nprobes')) \
                .refine_factor(self.case_config.search_param().get('refine_factor')) \
                .limit(k) \
                .select([self._scalar_field]) \
                .to_arrow()
        return res.column(self._scalar_field).to_pylist()

        
