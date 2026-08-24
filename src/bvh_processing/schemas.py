from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field


class ProcessBvhRequest(BaseModel):
    action_id: str = Field(
        alias="actionId",
        min_length=1,
        description="业务后端的动作记录 ID",
    )
    original_file_url: AnyHttpUrl = Field(
        alias="originalFileUrl",
        description="可直接下载 BVH 文件的 MinIO 地址",
    )
    handle_options: list[int] = Field(
        alias="handleOptions",
        description=(
            "按顺序执行的处理选项编号：1 整体去噪，2 整体平滑，"
            "3 脚步锁定校正，4 循环优化"
        ),
    )
    callback_url: AnyHttpUrl = Field(
        alias="callbackUrl",
        description="处理进度和最终结果的回调地址",
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ProcessBvhResponse(BaseModel):
    success: bool = Field(description="是否成功接收任务，不代表最终处理成功")
    task_id: str | None = Field(alias="taskId", description="异步任务 ID")
    message: str = Field(description="接收结果说明")

    model_config = ConfigDict(populate_by_name=True)


class HealthResponse(BaseModel):
    status: str
