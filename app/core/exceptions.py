from fastapi import HTTPException, status


class APIException(Exception):
    """API异常类"""

    pass


class ChatAppException(Exception):
    """基础异常类"""

    pass


class CredentialsException(HTTPException):
    """认证失败异常"""

    def __init__(self, detail: str = "认证凭证无效"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class PermissionDeniedException(HTTPException):
    """权限拒绝异常"""

    def __init__(self, detail: str = "没有足够的权限执行此操作"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


class NotFoundException(HTTPException):
    """资源不存在异常"""

    def __init__(self, detail: str = "请求的资源不存在"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class BadRequestException(HTTPException):
    """请求参数异常"""

    def __init__(self, detail: str = "请求参数有误"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class UserExistsException(BadRequestException):
    """用户已存在异常"""

    def __init__(self, detail: str = "该用户名或电子邮件已被使用"):
        super().__init__(detail=detail)


class LLMAPIException(HTTPException):
    """LLM API调用异常"""

    def __init__(self, detail: str = "大语言模型API调用失败"):
        super().__init__(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


class FileProcessingException(HTTPException):
    """文件处理异常"""

    def __init__(self, detail: str = "文件处理过程中出现错误"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail
        )


class InvalidFileTypeException(BadRequestException):
    """无效文件类型异常"""

    def __init__(self, detail: str = "不支持的文件类型"):
        super().__init__(detail=detail)


class FileTooLargeException(BadRequestException):
    """文件过大异常"""

    def __init__(self, detail: str = "文件大小超过限制"):
        super().__init__(detail=detail)
