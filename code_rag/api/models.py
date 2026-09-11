from pydantic import BaseModel, ConfigDict, Field


class SyncFileError(BaseModel):
    file: str
    message: str


class SyncResult(BaseModel):
    status: str
    indexed_files: int
    errors: list[SyncFileError] = Field(default_factory=list)


class ApiReport(BaseModel):
    library: str
    language: str
    report: str


class SetupResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_dir: str
    downloaded: list[str]
    skipped: list[str]
