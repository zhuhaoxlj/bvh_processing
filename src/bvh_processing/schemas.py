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
        description="回传类型：1 MP4 仿真视频和 ONNX 模型，2 ONNX 模型",
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
    loss_callback_url: AnyHttpUrl | None = Field(
        default=None,
        alias="lossCallbackUrl",
        description="训练期间 GPU metrics JSON 的回调地址",
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ProcessBvhResponse(BaseModel):
    success: bool = Field(description="是否成功接收任务，不代表最终处理成功")
    task_id: str | None = Field(alias="taskId", description="异步任务 ID")
    message: str = Field(description="接收结果说明")

    model_config = ConfigDict(populate_by_name=True)


class MergeBvhSegment(BaseModel):
    segment_id: str = Field(
        alias="segmentId",
        min_length=1,
        description="时间轴片段唯一 ID",
    )
    action_id: int = Field(
        alias="actionId",
        strict=True,
        gt=0,
        description="动作库资源 ID",
    )
    action_url: AnyHttpUrl = Field(
        alias="actionUrl",
        description="源 BVH 文件下载地址",
    )
    source_in_seconds: float = Field(
        alias="sourceInSec",
        ge=0,
        description="片段在源 BVH 中的开始时间",
    )
    source_out_seconds: float | None = Field(
        alias="sourceOutSec",
        default=None,
        gt=0,
        description="片段在源 BVH 中的结束时间；null 表示文件末尾",
    )
    output_duration_seconds: float = Field(
        alias="outputDurationSec",
        gt=0,
        description="裁剪片段变速后的输出时长",
    )
    gap_after_seconds: float = Field(
        alias="gapAfterSec",
        ge=0,
        le=10,
        description="当前片段与下一片段之间的过渡时长",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_source_range(self) -> "MergeBvhSegment":
        if (
            self.source_out_seconds is not None
            and self.source_out_seconds <= self.source_in_seconds
        ):
            raise ValueError("sourceOutSec 必须大于 sourceInSec")
        return self


class MergeBvhRequest(BaseModel):
    action_id: str | None = Field(
        default=None,
        alias="actionId",
        min_length=1,
        description="合并任务回调关联 ID（业务后端调用时提供）",
    )
    dance_id: str | int | None = Field(
        default=None,
        alias="danceId",
        description="舞蹈 ID（直接提交舞蹈编排时提供）",
    )
    timeline_offset_seconds: float = Field(
        default=0,
        alias="timelineOffsetSec",
        ge=0,
        description="首动作相对时间轴零点的偏移；不写入输出 BVH",
    )
    segments: list[MergeBvhSegment] = Field(
        min_length=1,
        description="按最终合并顺序排列的 BVH 片段",
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
    def validate_identifier(self) -> "MergeBvhRequest":
        if self.action_id is None and self.dance_id is None:
            raise ValueError("actionId 和 danceId 至少提供一个")
        return self

    @property
    def callback_reference_id(self) -> str:
        if self.action_id is not None:
            return self.action_id
        return str(self.dance_id)


class HealthResponse(BaseModel):
    status: str
