from pydantic import BaseModel, SecretStr
from ..api import DBConfig, DBCaseConfig, MetricType

class LanceDBCloudConfig(DBConfig):
    uri: SecretStr = "db://test"
    api_key: SecretStr
    region: str

    def to_dict(self) -> dict:
        return { "uri": self.uri.get_secret_value(), "api_key": self.api_key.get_secret_value(), "region": self.region }

class LanceDBCloudIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    num_partitions: int | None = 256
    num_sub_vectors: int | None = 96
    nprobes: int | None = 0

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "L2"
        elif self.metric_type == MetricType.IP:
            return "dot"
        return "cosine"

    def index_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
            "num_partitions": self.num_partitions,
            "num_sub_vectors": self.num_sub_vectors,
        }

    def search_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
            "nprobes": self.nprobes,
        }