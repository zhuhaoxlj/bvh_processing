from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field


class ProcessBvhRequest(BaseModel):
    action_id: str = Field(alias="actionId", min_length=1)
    original_file_url: AnyHttpUrl = Field(
        alias="originalFileUrl",
        description="可直接下载 BVH 文件的 MinIO 地址",
    )
    handle_options: list[int] = Field(alias="handleOptions")
    callback_url: AnyHttpUrl = Field(alias="callbackUrl")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ProcessBvhResponse(BaseModel):
    success: bool
    task_id: str | None = Field(alias="taskId")
    message: str

    model_config = ConfigDict(populate_by_name=True)


class HealthResponse(BaseModel):
    status: str
