from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator


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


class RetargetBvhRequest(BaseModel):
    action_id: str | None = Field(
        default=None,
        alias="actionId",
        min_length=1,
        description="业务后端的重定向记录 ID",
    )
    original_file_url: AnyHttpUrl = Field(
        alias="originalFileUrl",
        description="可直接下载 BVH 文件的 MinIO 地址",
    )
    robot_type: int = Field(
        alias="robotType",
        strict=True,
        ge=1,
        le=1,
        description="机器人类型：1 Unitree G1（Robot Retargeter）",
    )
    callback_url: AnyHttpUrl = Field(
        alias="callbackUrl",
        description="重定向 JSON 结果回调地址",
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class TrainBvhRequest(BaseModel):
    action_id: str | None = Field(
        default=None,
        alias="actionId",
        min_length=1,
        description="业务后端的训练记录 ID（可选）",
    )
    robot_type: int = Field(
        alias="robotType",
        strict=True,
        ge=1,
        le=1,
        description="机器人类型：1 Unitree G1",
    )
    algorithm_type: int = Field(
        alias="algorithmType",
        strict=True,
        ge=1,
        le=1,
        description="算法类型：当前仅支持 1 BeyondMimic",
    )
    npz_file_url: AnyHttpUrl = Field(
        alias="npzFileUrl",
        description="可直接下载重定向 NPZ 文件的 MinIO 地址",
    )
    domain_randomization: int = Field(
        alias="domainRandomization",
        strict=True,
        ge=2,
        le=2,
        description="域随机配置：当前仅支持 2（Whole Body Tracking 内置配置）",
    )
    return_type: int = Field(
        alias="returnType",
        strict=True,
        ge=1,
        le=2,
        description="回传类型：1 MP4 仿真视频，2 ONNX 模型",
    )
    gpu: int = Field(
        default=0,
        strict=True,
        ge=0,
        le=15,
        description="训练使用的物理 GPU 编号",
    )
    num_envs: int = Field(
        default=7168,
        alias="numEnvs",
        strict=True,
        ge=1,
        le=32768,
        description="Isaac Lab 并行环境数量",
    )
    max_iterations: int = Field(
        default=100000,
        alias="maxIterations",
        strict=True,
        ge=1,
        le=100000,
        description="BeyondMimic 最大训练迭代次数（GPU 控制服务上限 100000）",
    )
    seed: int = Field(
        default=42,
        strict=True,
        ge=0,
        le=2147483647,
        description="训练随机种子",
    )
    callback_url: AnyHttpUrl = Field(
        alias="callbackUrl",
        description="训练结果文件的回调地址",
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ProcessBvhResponse(BaseModel):
    success: bool = Field(description="是否成功接收任务，不代表最终处理成功")
    task_id: str | None = Field(alias="taskId", description="异步任务 ID")
    message: str = Field(description="接收结果说明")

    model_config = ConfigDict(populate_by_name=True)


class MergeBvhRequest(BaseModel):
    action_id: str = Field(
        alias="actionId",
        min_length=1,
        description="业务后端的动作记录 ID",
    )
    file_urls: list[AnyHttpUrl] = Field(
        alias="fileUrls",
        min_length=2,
        description="按合并顺序排列的 BVH 文件下载地址，至少两个",
    )
    intervals_seconds: list[float] = Field(
        alias="intervalsSeconds",
        description="相邻 BVH 文件之间的间隔秒数，数量必须比文件数量少一个",
    )
    bvh_motion_duration: list[float] = Field(
        alias="bvhMotionDuration",
        description="每个 BVH 文件用户拖动修改之后的动作时长，数量跟 bvh 文件数量一致",
    )
    callback_url: AnyHttpUrl = Field(
        alias="callbackUrl",
        description="合并完成后的结果回调地址",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_intervals(self) -> "MergeBvhRequest":
        expected_interval_count = len(self.file_urls) - 1
        if len(self.intervals_seconds) != expected_interval_count:
            raise ValueError(
                f"intervalsSeconds 必须包含 {expected_interval_count} 个间隔"
            )
        if any(interval < 0 for interval in self.intervals_seconds):
            raise ValueError("间隔秒数不能为负数")
        expected_duration_count = len(self.file_urls)
        if len(self.bvh_motion_duration) != expected_duration_count:
            raise ValueError(
                f"bvhMotionDuration 必须包含 {expected_duration_count} 个时长"
            )
        if any(duration < 0 for duration in self.bvh_motion_duration):
            raise ValueError("BVH 动作时长不能为负数")
        return self


class HealthResponse(BaseModel):
    status: str
