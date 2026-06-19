"""Domain error base. Subclasses carry a caller-safe message and HTTP status code."""


class AppError(Exception):
    status_code: int = 500

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
