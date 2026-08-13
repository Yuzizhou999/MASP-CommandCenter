import uvicorn

from .settings import Settings


def main() -> None:
    settings = Settings.load()
    uvicorn.run(
        "command_center.api:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
