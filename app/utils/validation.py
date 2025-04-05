import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


def validate_required_fields(
    data: Dict[str, Any], required_fields: List[str]
) -> List[str]:
    """
    验证必填字段

    Args:
        data: 要验证的数据字典
        required_fields: 必填字段列表

    Returns:
        缺失的字段列表，如果没有缺失字段则返回空列表
    """
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == "":
            missing_fields.append(field)
    return missing_fields


def validate_numeric(value: Any) -> bool:
    """
    验证值是否为数字类型或可转换为数字
    """
    if isinstance(value, (int, float)):
        return True

    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False

    return False


def validate_string_length(
    value: str, min_length: int = 0, max_length: Optional[int] = None
) -> bool:
    """
    验证字符串长度是否在指定范围内
    """
    if not isinstance(value, str):
        return False

    if len(value) < min_length:
        return False

    if max_length is not None and len(value) > max_length:
        return False

    return True


def validate_date_format(date_str: str, format_str: str = "%Y-%m-%d") -> bool:
    """
    验证日期字符串格式是否符合要求
    """
    try:
        datetime.strptime(date_str, format_str)
        return True
    except ValueError:
        return False


def validate_email(email: str) -> bool:
    """
    验证邮箱格式
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_password_strength(password: str) -> Dict[str, Union[bool, str]]:
    """
    验证密码强度

    要求:
    - 至少8个字符
    - 至少包含一个小写字母
    - 至少包含一个大写字母
    - 至少包含一个数字
    - 至少包含一个特殊字符

    Returns:
        验证结果字典: {"valid": bool, "message": str}
    """
    # 长度检查
    if len(password) < 8:
        return {"valid": False, "message": "密码长度必须至少为8个字符"}

    # 检查小写字母
    if not re.search(r"[a-z]", password):
        return {"valid": False, "message": "密码必须包含至少一个小写字母"}

    # 检查大写字母
    if not re.search(r"[A-Z]", password):
        return {"valid": False, "message": "密码必须包含至少一个大写字母"}

    # 检查数字
    if not re.search(r"[0-9]", password):
        return {"valid": False, "message": "密码必须包含至少一个数字"}

    # 检查特殊字符
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return {"valid": False, "message": "密码必须包含至少一个特殊字符"}

    return {"valid": True, "message": "密码强度良好"}


def validate_phone_number(phone: str, country_code: str = "CN") -> bool:
    """
    验证手机号格式

    目前仅支持简单验证，未区分国家/地区
    """
    patterns = {
        "CN": r"^1[3-9]\d{9}$",  # 中国大陆手机号
        "US": r"^(\+?1)?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$",  # 美国手机号
        # 可根据需要添加其他国家/地区的验证规则
    }

    pattern = patterns.get(country_code, r"^\+?[0-9]{10,15}$")  # 默认使用通用规则
    return bool(re.match(pattern, phone))


def validate_url(url: str) -> bool:
    """
    验证URL格式
    """
    pattern = r"^(http|https)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$"
    return bool(re.match(pattern, url))
