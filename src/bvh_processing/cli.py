"""BVH Processing 服务启动命令。"""

import argparse
import os


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 BVH Processing API")
    parser.add_argument(
        "--mock",
        choices=(0, 1),
        default=1,
        type=int,
        help="训练接口模式：1 返回内置演示产物（默认），0 调用 GPU 控制服务",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9001, type=int)
    parser.add_argument("--reload", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    os.environ["BVH_MOCK"] = str(args.mock)

    import uvicorn

    uvicorn.run(
        "bvh_processing.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
