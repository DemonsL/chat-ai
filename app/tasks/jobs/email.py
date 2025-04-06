from typing import Dict, List, Optional

from loguru import logger

from app.core.config import settings
from app.tasks.base import async_task


@async_task(
    name="tasks.email.send_email",
    queue="email_tasks",
    max_retries=3,
    retry_backoff=True,
)
async def send_email_task(
    self,
    email_to: str,
    subject: str,
    html_content: str,
    environment: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    发送邮件任务

    参数:
        email_to: 收件人邮箱
        subject: 邮件主题
        html_content: HTML内容
        environment: 模板环境变量
    """
    logger.info(f"发送邮件到 {email_to}, 主题: {subject}")

    try:
        # 根据配置选择不同的邮件服务提供商
        if settings.EMAIL_PROVIDER == "sendgrid":
            # SendGrid实现
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail

            message = Mail(
                from_email=settings.EMAILS_FROM_EMAIL,
                to_emails=email_to,
                subject=subject,
                html_content=html_content,
            )
            sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
            response = sg.send(message)

            return {
                "success": 200 <= response.status_code < 300,
                "provider": "sendgrid",
                "status_code": response.status_code,
            }

        elif settings.EMAIL_PROVIDER == "smtp":
            # SMTP实现
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            msg = MIMEMultipart()
            msg["From"] = settings.EMAILS_FROM_EMAIL
            msg["To"] = email_to
            msg["Subject"] = subject
            msg.attach(MIMEText(html_content, "html"))

            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                server.starttls()
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.send_message(msg)

            return {"success": True, "provider": "smtp"}

        else:
            # 默认开发模式，仅记录不实际发送
            logger.warning(f"未配置邮件提供商，模拟发送邮件到 {email_to}")
            return {"success": True, "provider": "development", "sent": False}

    except Exception as e:
        logger.error(f"发送邮件失败: {str(e)}")
        raise e


@async_task(name="tasks.email.send_notification", queue="email_tasks", max_retries=3)
async def send_notification_task(
    self,
    subject: str,
    content: str,
    recipients: List[str],
    template: str = "notification",
) -> Dict:
    """
    发送通知邮件给多个收件人

    参数:
        subject: 主题
        content: 内容
        recipients: 收件人列表
        template: 邮件模板
    """
    results = []

    for recipient in recipients:
        result = await send_email_task(
            email_to=recipient,
            subject=subject,
            html_content=f"<div>{content}</div>",
        )
        results.append({"email": recipient, "result": result})

    return {"success": True, "results": results}
