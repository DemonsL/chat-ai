import random
import re
import string
import uuid
from typing import Optional


def generate_random_string(length: int = 10) -> str:
    """
    生成随机字符串，包含字母和数字
    """
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_random_password(length: int = 12) -> str:
    """
    生成随机密码，包含字母、数字和特殊字符
    """
    chars = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?"
    return "".join(random.choices(chars, k=length))


def generate_uuid() -> str:
    """
    生成UUID
    """
    return str(uuid.uuid4())


def slugify(text: str) -> str:
    """
    生成URL友好的slug
    """
    # 转换为小写
    text = text.lower()
    # 替换非字母数字字符为连字符
    text = re.sub(r"[^a-z0-9]+", "-", text)
    # 移除前后连字符
    text = text.strip("-")
    return text


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断文本到指定长度，添加后缀
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + suffix


def is_valid_email(email: str) -> bool:
    """
    验证邮箱格式是否有效
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def is_valid_phone_number(phone: str) -> bool:
    """
    验证手机号格式是否有效（简单验证）
    """
    pattern = r"^\+?[0-9]{10,15}$"
    return bool(re.match(pattern, phone))


def mask_email(email: str) -> str:
    """
    对邮箱地址进行部分遮蔽
    例如: example@example.com -> e***e@example.com
    """
    if not is_valid_email(email):
        return email

    parts = email.split("@")
    if len(parts) != 2:
        return email

    username, domain = parts
    if len(username) <= 2:
        masked_username = username
    else:
        masked_username = username[0] + "*" * (len(username) - 2) + username[-1]

    return f"{masked_username}@{domain}"


def camel_to_snake(name: str) -> str:
    """
    将驼峰命名转换为蛇形命名
    例如: camelCase -> camel_case
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def snake_to_camel(name: str) -> str:
    """
    将蛇形命名转换为驼峰命名
    例如: snake_case -> snakeCase
    """
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])
