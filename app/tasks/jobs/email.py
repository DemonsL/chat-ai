import logging
from typing import Dict, List, Optional

from celery import shared_task
from pydantic import EmailStr

from app.core.config import settings

logger = logging.getLogger(__name__)


@shared_task
def send_email(
    email_to: str,
    subject: str,
    html_content: str,
    environment: Optional[Dict[str, str]] = None,
) -> None:
    """
    发送邮件的后台任务

    在实际实现中，这里需要集成实际的邮件服务（如SMTP或第三方服务如SendGrid）
    """
    logger.info(f"发送邮件到 {email_to}, 主题: {subject}")
    # 这里是邮件发送的实际实现逻辑
    # 例如:
    # from sendgrid import SendGridAPIClient
    # from sendgrid.helpers.mail import Mail
    # message = Mail(from_email=settings.EMAILS_FROM_EMAIL,
    #                to_emails=email_to,
    #                subject=subject,
    #                html_content=html_content)
    # sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
    # sg.send(message)

    # 示例日志
    logger.info(f"邮件已发送到 {email_to}")
