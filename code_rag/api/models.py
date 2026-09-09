from pydantic import BaseModel, ConfigDict


class SyncResult(BaseModel):
    status: str
    indexed_files: int


class ApiReport(BaseModel):
    library: str
    language: str
    report: str


class SetupResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_dir: str
    downloaded: list[str]
    skipped: list[str]
