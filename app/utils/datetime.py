from datetime import datetime, timedelta, timezone
from typing import Optional


def now_utc() -> datetime:
    """
    获取当前UTC时间
    """
    return datetime.now(timezone.utc)


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    格式化日期时间
    """
    return dt.strftime(format_str)


def parse_datetime(
    dt_str: str, format_str: str = "%Y-%m-%d %H:%M:%S"
) -> Optional[datetime]:
    """
    解析日期时间字符串
    """
    try:
        return datetime.strptime(dt_str, format_str)
    except ValueError:
        return None


def add_days(dt: datetime, days: int) -> datetime:
    """
    添加天数
    """
    return dt + timedelta(days=days)


def add_hours(dt: datetime, hours: int) -> datetime:
    """
    添加小时数
    """
    return dt + timedelta(hours=hours)


def get_start_of_day(dt: datetime) -> datetime:
    """
    获取某天的开始时间 (00:00:00)
    """
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def get_end_of_day(dt: datetime) -> datetime:
    """
    获取某天的结束时间 (23:59:59)
    """
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def get_date_range(start_date: datetime, end_date: datetime) -> list[datetime]:
    """
    获取日期范围，返回每一天的日期列表
    """
    days = (end_date - start_date).days + 1
    return [start_date + timedelta(days=i) for i in range(days)]


def is_future_date(dt: datetime) -> bool:
    """
    判断日期是否在未来
    """
    return dt > now_utc()


def get_month_start(dt: datetime) -> datetime:
    """
    获取月初日期
    """
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def get_month_end(dt: datetime) -> datetime:
    """
    获取月末日期
    """
    next_month = dt.replace(day=28) + timedelta(days=4)  # 跳到下个月
    return (next_month.replace(day=1) - timedelta(days=1)).replace(
        hour=23, minute=59, second=59, microsecond=999999
    )
